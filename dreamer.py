import queue
import threading

import numpy
import torch
from torch.distributions import kl_divergence, Independent, OneHotCategoricalStraightThrough, Normal
from torch.amp import GradScaler
import imageio
import matplotlib
import matplotlib.image
import random
import time

from networks import RecurrentModel, PriorNet, PosteriorNet, RewardModel, ContinueModel, EncoderConv, DecoderConv, Actor, Critic, FBV_SM, FiLMLayer, PositionalEncoder, SmAuxiliaryDecoder
from utils import computeLambdaValues, Moments
from buffer import ReplayBuffer
import envs
from func import *


class Dreamer:
    def __init__(self, observationShape, actionSize, actionLow, actionHigh, dt, device, config, configFile):
        self.observationShape   = observationShape
        self.actionSize         = actionSize
        self.dt                 = dt
        self.config             = config
        self.configFile         = configFile
        self.device             = device
        self.smFrameQueue       = queue.Queue(maxsize=2)
        self.wmFrameQueue       = queue.Queue(maxsize=2)

        self.recurrentSize  = config.recurrentSize
        self.latentSize     = config.latentLength*config.latentClasses
        self.smLatentSize   = config.selfModel.d_filter // 4
        self.fullStateSize  = config.recurrentSize + self.latentSize + self.smLatentSize
        self.wmFullStateSize = config.recurrentSize + self.latentSize

        self.actor                             = Actor(self.fullStateSize, actionSize, actionLow, actionHigh, device,                                  config.actor          ).to(self.device)
        self.critic                            = Critic(self.fullStateSize,                                                                            config.critic         ).to(self.device)
        self.encoder                           = EncoderConv(observationShape, self.config.encodedObsSize,                                             config.encoder        ).to(self.device)
        self.decoder                           = DecoderConv(self.wmFullStateSize, observationShape,                                                   config.decoder        ).to(self.device)
        self.recurrentModel                    = RecurrentModel(config.recurrentSize, self.latentSize, actionSize,                 config.recurrentModel ).to(self.device)
        self.priorNet                          = PriorNet(config.recurrentSize, config.latentLength, config.latentClasses,                             config.priorNet       ).to(self.device)
        self.posteriorNet                      = PosteriorNet(config.recurrentSize + config.encodedObsSize, config.latentLength, config.latentClasses, config.posteriorNet   ).to(self.device)
        self.rewardPredictor                   = RewardModel(self.fullStateSize,                                                                       config.reward         ).to(self.device)
        if config.selfModel.positionalEncoder:
            encoder = PositionalEncoder(d_input=(config.selfModel.dof - 2) + 3, n_freqs=10, log_space=True                                                                   ).to(self.device)
        else:
            encoder = None
        self.selfModel                         = FBV_SM(config=config, encoder=encoder, d_input=(config.selfModel.dof - 2) + 3, d_filter=config.selfModel.d_filter, output_size=2           ).to(self.device)
        self.filmLayer                         = FiLMLayer(self.fullStateSize                                                                                                ).to(self.device)
        #self.smAuxDecoder                     = SmAuxiliaryDecoder(self.fullStateSize, self.smLatentSize                                                                    ).to(self.device)

        if config.useContinuationPrediction:
            self.continuePredictor  = ContinueModel(self.fullStateSize,                                                                                config.continuation   ).to(self.device)

        self.buffer         = ReplayBuffer(observationShape, actionSize, config.buffer, device)
        self.valueMoments   = Moments(device)

        self.worldModelParameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.recurrentModel.parameters()) +
                                     list(self.priorNet.parameters()) + list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()))
        if self.config.useContinuationPrediction:
            self.worldModelParameters += list(self.continuePredictor.parameters())

        self.worldModelOptimizer    = torch.optim.Adam(self.worldModelParameters,   lr=self.config.worldModelLR)
        self.actorOptimizer         = torch.optim.Adam(self.actor.parameters(),     lr=self.config.actorLR)
        self.criticOptimizer        = torch.optim.Adam(self.critic.parameters(),    lr=self.config.criticLR)
        self.selfModelOptimizer     = torch.optim.Adam(self.selfModel.parameters(), lr=self.config.selfModelLR)

        self.totalEpisodes       = 0
        self.totalEnvSteps       = 0
        self.totalGradientSteps  = 0
        self.totalSelfModelSteps = config.batchSize * (config.batchLength - 1)
        self.smMinLoss = np.inf


    def worldModelTraining(self, data, smLatentStates):
        encodedObservations = self.encoder(data.observations.view(-1, *self.observationShape)).view(self.config.batchSize, self.config.batchLength, -1)
        previousRecurrentState  = torch.zeros(self.config.batchSize, self.recurrentSize,    device=self.device)
        previousLatentState     = torch.zeros(self.config.batchSize, self.latentSize,       device=self.device)
        previousSmLatentState   = torch.zeros(self.config.batchSize, self.smLatentSize,       device=self.device) # not used in this version, was used for recurrent state (eze)

        recurrentStates, priorsLogits, posteriors, posteriorsLogits = [], [], [], []
        for t in range(1, self.config.batchLength):
            recurrentState              = self.recurrentModel(previousRecurrentState, previousLatentState, data.actions[:, t-1])
            _, priorLogits              = self.priorNet(recurrentState)
            posterior, posteriorLogits  = self.posteriorNet(torch.cat((recurrentState, encodedObservations[:, t]), -1))

            recurrentStates.append(recurrentState)
            priorsLogits.append(priorLogits)
            posteriors.append(posterior)
            posteriorsLogits.append(posteriorLogits)

            previousRecurrentState = recurrentState
            previousLatentState    = posterior
            previousSmLatentState  = smLatentStates[:, t-1, :]

        recurrentStates             = torch.stack(recurrentStates,              dim=1) # (batchSize, batchLength-1, recurrentSize)
        priorsLogits                = torch.stack(priorsLogits,                 dim=1) # (batchSize, batchLength-1, latentLength, latentClasses)
        posteriors                  = torch.stack(posteriors,                   dim=1) # (batchSize, batchLength-1, latentLength*latentClasses)
        posteriorsLogits            = torch.stack(posteriorsLogits,             dim=1) # (batchSize, batchLength-1, latentLength, latentClasses)
        fullStates                  = torch.cat((recurrentStates, posteriors), dim=-1) # (batchSize, batchLength-1, recurrentSize + latentLength*latentClasses)

        reconstructionMeans        =  self.decoder(fullStates.view(-1, self.wmFullStateSize)).view(self.config.batchSize, self.config.batchLength-1, *self.observationShape)
        reconstructionDistribution =  Independent(Normal(reconstructionMeans, 1), len(self.observationShape))
        reconstructionLoss         = -reconstructionDistribution.log_prob(data.observations[:, 1:]).mean()

        fullStates = torch.cat((fullStates, smLatentStates), dim=-1)

        rewardDistribution  =  self.rewardPredictor(fullStates)
        rewardLoss          = -rewardDistribution.log_prob(data.rewards[:, 1:].squeeze(-1)).mean()

        priorDistribution       = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits              ), 1)
        priorDistributionSG     = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits.detach()     ), 1)
        posteriorDistribution   = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits          ), 1)
        posteriorDistributionSG = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits.detach() ), 1)

        priorLoss       = kl_divergence(posteriorDistributionSG, priorDistribution  )
        posteriorLoss   = kl_divergence(posteriorDistribution  , priorDistributionSG)
        freeNats        = torch.full_like(priorLoss, self.config.freeNats)

        priorLoss       = self.config.betaPrior*torch.maximum(priorLoss, freeNats)
        posteriorLoss   = self.config.betaPosterior*torch.maximum(posteriorLoss, freeNats)
        klLoss          = (priorLoss + posteriorLoss).mean()

        worldModelLoss =  reconstructionLoss + rewardLoss + klLoss # I think that the reconstruction loss is relatively a bit too high (11k)s
        
        if self.config.useContinuationPrediction:
            continueDistribution = self.continuePredictor(fullStates)
            continueLoss         = nn.BCELoss(continueDistribution.probs, 1 - data.dones[:, 1:])
            worldModelLoss      += continueLoss.mean()

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        nn.utils.clip_grad_norm_(self.worldModelParameters, self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.worldModelOptimizer.step()

        klLossShiftForGraphing = (self.config.betaPrior + self.config.betaPosterior)*self.config.freeNats
        metrics = {
            "worldModelLoss"        : worldModelLoss.item() - klLossShiftForGraphing,
            "reconstructionLoss"    : reconstructionLoss.item(),
            "rewardPredictorLoss"   : rewardLoss.item(),
            "klLoss"                : klLoss.item() - klLossShiftForGraphing}

        return fullStates.view(-1, self.fullStateSize).detach(), metrics


    def selfModelTraining(self, data):
        totalGradientSteps = self.totalGradientSteps
        config = self.configFile
        scaler = GradScaler()
        sim_real = config.dreamer.selfModel.sim_real
        arm_ee = config.dreamer.selfModel.arm_ee
        seed_num = config.seed
        robotid = config.robotID

        # 0:OM, 1:OneOut, 2: OneOut with distance
        different_arch = 0
        #np.random.seed(seed_num)
        #random.seed(seed_num)
        #torch.manual_seed(seed_num)
        DOF = config.dreamer.selfModel.dof  # the number of motors  # dof4 apr03
        Flag_save_image_during_training = True

        if config.dreamer.selfModel.positionalEncoder:
            add_name = 'PE'
        else:
            add_name = 'no_PE'
        LOG_PATH = "train_log/%s_id%d_(%d)_%s(%s)_%s" % (sim_real, robotid, seed_num, add_name, arm_ee, config.runName)
        config = config.dreamer
        if different_arch != 0:
           LOG_PATH += 'diff_out_%d' % different_arch
        os.makedirs(LOG_PATH + "/image/", exist_ok=True)

        # Encoders
        """arm dof = 2+3; arm dof=3+3"""
        # Stratified sampling
        perturb = True  # If set, applies noise to sample positions
        inverse_depth = False  # If set, samples points linearly in inverse depth
        # Hierarchical sampling
        n_samples_hierarchical = 64  # Number of samples per ray  # again no use found, (eze)
        perturb_hierarchical = False  # If set, applies noise to sample positions  # again no use found, (eze)

        # Training
        tr = config.selfModel.tr  # training ratio
        batchSize = int(config.batchSize)
        batchLength = int(config.batchLength)
        training_imges_snapshot = data.nextSmObservations.clone()
        training_angles_snapshot = data.angles.clone()
        height, width = training_imges_snapshot[0, 0].shape[1:]
        training_imges_snapshot = training_imges_snapshot.reshape(batchSize, batchLength, -1, height, width)
        training_angles_snapshot = training_angles_snapshot.reshape(batchSize, batchLength, DOF)
        train_amount = int((batchLength - 1) * tr)
        loss_v_last = np.inf
        patience = 0
        min_loss = self.smMinLoss

        # Early Stopping
        warmup_iters = 400  # Number of iterations during warmup phase
        warmup_min_fitness = 10.0  # Min val PSNR to continue training at warmup_iters
        n_restarts = 1000  # Number of times to restart if training stalls

        record_file_train = open(LOG_PATH + "/log_train.txt", "a")
        record_file_val = open(LOG_PATH + "/log_val.txt", "a")
        Patience_threshold = config.selfModel.patienceThreshold  # original: 100, (eze)

        # pretrained_model_pth = 'train_log/real_train_1_log0928_%ddof_100(0)/best_model/'%num_data
        # pretrained_model_pth = 'train_log/real_id1_10000(1)_PE(arm)/best_model/'
        self.totalSelfModelSteps = (totalGradientSteps + 1) * config.batchSize * (config.batchLength - 1)

        for _ in range(n_restarts):
            model = self.selfModel
            optimizer = self.selfModelOptimizer

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True)
            latents = torch.zeros(batchSize, batchLength - 1, config.selfModel.d_filter // 4, device=device)  # batchLength -1 because WM ignores first fullstate, (eze)

            for t in range(batchLength - 1):
                one = time.time()
                angles = training_angles_snapshot[:, t]
                imges = training_imges_snapshot[:, t].permute(0, 2, 3, 1).cpu().numpy()  # permute because cv2 uses channel last, (eze)

                # Pick an image as the target. # RGB -> colourfilter -> binary, (eze)
                maskedImg = torch.zeros(config.batchSize, training_imges_snapshot.shape[3], training_imges_snapshot.shape[4], device=device, dtype=torch.float32)
                for j in range(len(imges)):
                    maskedImg[j] = color_filter(config, imges[j])
                target_img = crop_center(maskedImg)  # also downscales, (eze)
                #img = target_img[0, 0].cpu().numpy()
                target_img = target_img.reshape([batchSize, -1])
                #matplotlib.image.imsave(LOG_PATH + '/image/' + "test.png", img, cmap='gray')

                if t < train_amount:
                    model.train()
                    train = True
                else:
                    model.eval()
                    train = False

                # Run one iteration of TinyNeRF and get the rendered RGB image.
                with autocast("cuda"):
                    latents[:, t], outputs = self_model_forward(config=self.configFile,
                                                                model=model,
                                                                arm_angle=angles,
                                                                output_flag=different_arch,
                                                                observation_shape=self.observationShape)
                two = time.time()
                rgb_predicted = outputs['rgb_map']
                if train:
                    # Backprop!
                    modelLock = threading.Lock()
                    with modelLock:
                        optimizer.zero_grad(set_to_none=True)
                        with autocast("cuda"):
                            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        loss_train = loss.item()
                        three = time.time()
                else:
                    # Evaluate testing
                    torch.no_grad()
                    valid_psnr = []
                    valid_image = []

                    with autocast("cuda"):
                       v_loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
                    np_image = rgb_predicted.reshape(
                        [-1, int(height * 0.25), int(width * 0.25), 1]).detach().cpu().numpy()
                    valid_image.append(np_image[:6])

            loss_valid = np.mean(v_loss.item())
            #print("SM-Loss:", loss_valid, 'patience', patience)
            scheduler.step(loss_valid)

            # save test image
            np_image_combine = np.hstack(valid_image[0])
            np_image_combine = np.dstack((np_image_combine, np_image_combine, np_image_combine))
            np_image_combine = np.clip(np_image_combine, 0, 1)
            try:
                matplotlib.image.imsave(LOG_PATH + '/image/' + 'latest.png', np_image_combine)
                if Flag_save_image_during_training and totalGradientSteps % 50 == 0:  # note that it doesnt save every 50 steps bc it only trains every 4 steps, (eze)
                    matplotlib.image.imsave(
                        LOG_PATH + '/image/' + '%d.png' % (self.totalSelfModelSteps),
                        np_image_combine)

                record_file_train.write(str(loss_train) + "\n")
                record_file_val.write(str(loss_valid) + "\n")

                if min_loss > loss_valid:
                    """record the best image and model"""
                    min_loss = loss_valid
                    matplotlib.image.imsave(LOG_PATH + '/image/' + 'best.png', np_image_combine)
                    patience = 0
                    success = True
                elif loss_valid == loss_v_last:
                    print("restart")
                    success = False
                else:
                    patience += 1
                    success = True
            except:
                print("\nSelf-model store-exception!!\n")

            loss_v_last = loss_valid
            # os.makedirs(LOG_PATH + "epoch_%d_model" % i, exist_ok=True)
            # torch.save(model.state_dict(), LOG_PATH + 'epoch_%d_model/nerf.pt' % i)
            # torch.cuda.empty_cache()    # to save memory
            latents = latents.reshape(config.batchSize, config.batchLength - 1, latents.shape[-1])
            if patience > Patience_threshold:
                break
            if success:
                print('SelfModel-Training successful!')
                break

        record_file_train.close()
        record_file_val.close()
        metrics = {
            "smLoss" : v_loss.item()
        }
        return latents, loss_valid, metrics

    def behaviorTraining(self, fullState):
        recurrentState, latentState, smLatentState = torch.split(fullState, (self.recurrentSize, self.latentSize, self.smLatentSize), -1)
        fullStates, logprobs, entropies, auxLosses = [], [], [], []
        energy = torch.randint(0, 1000, (self.config.batchLength - 1, self.config.batchSize))
        for _ in range(self.config.imaginationHorizon):
            fullState = self.filmLayer(fullState, torch.tensor([energy/self.config.envReward.max_energy], device=self.device, dtype=torch.float32))
            action, logprob, entropy = self.actor(fullState.detach(), training=True)
            energy = energy - 1
            recurrentState = self.recurrentModel(recurrentState, latentState, action)
            latentState, _ = self.priorNet(recurrentState)

            fullState = torch.cat((recurrentState, latentState, smLatentState), -1)
            fullStates.append(fullState)
            logprobs.append(logprob)
            entropies.append(entropy)

        #first_layer_weights = self.actor.network[0].weight  # (hidden_size, 800) first layer, compare sm to wm Impact (eze)
        #sm_weights = first_layer_weights[:, -32:]
        #wm_weights = first_layer_weights[:, :768]
        #print("\nSm: ", sm_weights.abs().mean())
        #print(", Wm: ", wm_weights.abs().mean())

        fullStates  = torch.stack(fullStates,    dim=1) # (batchSize*batchLength, imaginationHorizon, recurrentSize + latentLength*latentClasses)
        logprobs    = torch.stack(logprobs[1:],  dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        entropies   = torch.stack(entropies[1:], dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        
        predictedRewards = self.rewardPredictor(fullStates[:, :-1]).mean
        print(predictedRewards[2])
        values           = self.critic(fullStates).mean
        continues        = self.continuePredictor(fullStates).mean if self.config.useContinuationPrediction else torch.full_like(predictedRewards, self.config.discount)

        lambdaValues     = computeLambdaValues(predictedRewards, values, continues, self.config.lambda_)

        _, inverseScale = self.valueMoments(lambdaValues)
        advantages      = (lambdaValues - values[:, :-1])/inverseScale

        actorLoss = -torch.mean(advantages.detach()*logprobs + self.config.entropyScale*entropies)

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.actorOptimizer.step()

        valueDistributions  =  self.critic(fullStates[:, :-1].detach())
        criticLoss          = -torch.mean(valueDistributions.log_prob(lambdaValues.detach()))

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.criticOptimizer.step()

        metrics = {
            "actorLoss"     : actorLoss.item(),
            "criticLoss"    : criticLoss.item(),
            "entropies"     : entropies.mean().item(),
            "logprobs"      : logprobs.mean().item(),
            "advantages"    : advantages.mean().item(),
            "criticValues"  : values.mean().item()}
        return metrics


    @torch.no_grad()
    def environmentInteraction(self, wmEnv, smEnv, numEpisodes, seed=None, evaluation=False, saveVideo=False, liveView=False, dreamerLiveView=False, filename="videos/unnamedVideo", fps=30, macroBlockSize=16):
        scores = []
        overalMovement = 0
        overalMovements = numpy.zeros(8)
        for i in range(numEpisodes):
            recurrentState, latentState = torch.zeros(1, self.recurrentSize, device=self.device), torch.zeros(1, self.latentSize, device=self.device)
            action = torch.zeros(1, self.actionSize).to(self.device)

            smObservation = smEnv.reset(seed= (seed + self.totalEpisodes if seed else None))
            wmObservation = wmEnv.reset(seed= (seed + self.totalEpisodes if seed else None)) if wmEnv else smObservation

            encodedObservation = self.encoder(torch.from_numpy(wmObservation).float().unsqueeze(0).to(self.device))
            angles = torch.as_tensor(smEnv.unwrapped.data.qpos.copy()[:self.config.selfModel.dof], device=self.device, dtype=torch.float32).unsqueeze(0)

            maxEnergy = self.config.envReward.max_energy
            energy = maxEnergy
            currentScore, stepCount, done, frames = 0, 0, False, []
            modelLock = threading.Lock()
            while not done:
                with modelLock:
                    smLatentState, smPrediction = self_model_forward(config=self.configFile, model=self.selfModel.eval(), arm_angle=angles, output_flag=0, observation_shape=self.observationShape)
                    smPrediction = smPrediction['rgb_map']
                    target_img = crop_center(torch.from_numpy(smObservation).unsqueeze(0)).mean(dim=-1 if smObservation.shape[-1] in (1, 3) else 1).to(device).reshape(1, -1)
                with autocast("cuda"):
                    sm_loss = torch.nn.functional.mse_loss(smPrediction, target_img)
                #print("smLatentStateSize: ", smLatentState.size(), "recurrentStateSize: ", recurrentState.size(), "latentStateSize: ", latentState.size())  # debuging, (eze)
                recurrentState                  = self.recurrentModel(recurrentState, latentState, action)
                latentState, _                  = self.posteriorNet(torch.cat((recurrentState, encodedObservation.view(1, -1)), -1))
                modulatedState                  = self.filmLayer(torch.cat((recurrentState, latentState, smLatentState * self.config.smToWmRatio), -1), torch.tensor([energy/maxEnergy], device=self.device, dtype=torch.float32))

                action          = self.actor(modulatedState)
                actionNumpy     = action.cpu().numpy().reshape(-1)

                if wmEnv:
                    nextWmObservation, reward, done = wmEnv.step(actionNumpy)
                    nextObservation, _, _ = smEnv.step(actionNumpy)
                    envtype = wmEnv
                else:
                    nextWmObservation, reward, done = smEnv.step(actionNumpy)
                    nextObservation = nextWmObservation
                    envtype = smEnv

                if envs.in_energy_zone(envtype):
                    energy += 50
                else:
                    energy -= 1
                #if envs.check_collision_with_obstacles(envtype):  # already present in the standard ant reward function (eze)
                #    reward -= 1
                if energy == 0:
                    done = True
                reward -= abs((800 - energy) * 0.005)  # small penalty for too much or too little energy (eze)

                l = 0
                movePenalty = 0
                for j in actionNumpy:  # Penalty for using one part too often (eze)
                    overalMovement += abs(j)
                    overalMovements[l] += abs(j)
                    if overalMovements[l] >= overalMovement * 0.2:
                        reward -= abs(j)
                        movePenalty = abs(j)
                    l += 1

                # Penalty for bad Vision/ too much angle of central body-part (eze)
                _, x, y, _ = envtype.unwrapped.data.qpos[3:7]  # (w, x, y, z) (eze)
                up_z = 1 - 2 * (x ** 2 + y ** 2)
                up_z_pen = 0
                if up_z < 0.5:
                    reward -= abs((1 - up_z) * 2)
                    up_z_pen = up_z

                if stepCount % 10 == 0:
                    print("Overall: ", reward, "   Energy: ", energy, "   MovementDist: ", movePenalty, "   Vision: ", up_z_pen)
                angles = torch.as_tensor(smEnv.unwrapped.data.qpos.copy()[:self.config.selfModel.dof], device=self.device, dtype=torch.float32)  # qpos from documentation, (eze)
                if not evaluation:
                    self.buffer.add(wmObservation, smObservation, actionNumpy, reward, nextWmObservation, nextObservation, done, angles)

                if saveVideo and i == 0:
                    frame = smEnv.render()
                    targetHeight = (frame.shape[0] + macroBlockSize - 1)//macroBlockSize*macroBlockSize # getting rid of imagio warning
                    targetWidth = (frame.shape[1] + macroBlockSize - 1)//macroBlockSize*macroBlockSize
                    frames.append(np.pad(frame, ((0, targetHeight - frame.shape[0]), (0, targetWidth - frame.shape[1]), (0, 0)), mode='edge'))

                if liveView and i == 0:
                    frame = smEnv.render()
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.smFrameQueue.put(frame_bgr)

                if dreamerLiveView and i == 0:
                    frame = wmEnv.render()
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.wmFrameQueue.put(frame_bgr)

                encodedObservation = self.encoder(torch.from_numpy(nextWmObservation).float().unsqueeze(0).to(self.device))
                angles = angles.unsqueeze(0)
                smObservation = nextObservation
                
                currentScore += reward
                stepCount += 1
                if done:
                    scores.append(currentScore)
                    if not evaluation:
                        self.totalEpisodes += 1
                        self.totalEnvSteps += stepCount

                    if saveVideo and i == 0:
                        finalFilename = f"{filename}_reward_{currentScore:.0f}.mp4"
                        with imageio.get_writer(finalFilename, fps=fps) as video:
                            for frame in frames:
                                video.append_data(frame)
                    break
        return sum(scores)/numEpisodes if numEpisodes else None, np.mean(sm_loss.item())
    

    def saveCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'

        checkpoint = {
            'encoder'               : self.encoder.state_dict(),
            'decoder'               : self.decoder.state_dict(),
            'recurrentModel'        : self.recurrentModel.state_dict(),
            'priorNet'              : self.priorNet.state_dict(),
            'posteriorNet'          : self.posteriorNet.state_dict(),
            'rewardPredictor'       : self.rewardPredictor.state_dict(),
            'actor'                 : self.actor.state_dict(),
            'critic'                : self.critic.state_dict(),
            'sefModel'              : self.selfModel.state_dict(),
            'worldModelOptimizer'   : self.worldModelOptimizer.state_dict(),
            'criticOptimizer'       : self.criticOptimizer.state_dict(),
            'actorOptimizer'        : self.actorOptimizer.state_dict(),
            'selfModelOptimizer'    : self.selfModelOptimizer.state_dict(),
            'totalEpisodes'         : self.totalEpisodes,
            'totalEnvSteps'         : self.totalEnvSteps,
            'totalGradientSteps'    : self.totalGradientSteps,
            'totalSelfModelSteps'   : self.totalSelfModelSteps}
        if self.config.useContinuationPrediction:
            checkpoint['continuePredictor'] = self.continuePredictor.state_dict()
        torch.save(checkpoint, checkpointPath)


    def loadCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpointPath}")
        
        checkpoint = torch.load(checkpointPath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.recurrentModel.load_state_dict(checkpoint['recurrentModel'])
        self.priorNet.load_state_dict(checkpoint['priorNet'])
        self.posteriorNet.load_state_dict(checkpoint['posteriorNet'])
        self.rewardPredictor.load_state_dict(checkpoint['rewardPredictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.selfModel.load_state_dict(checkpoint['selfModel'])
        self.worldModelOptimizer.load_state_dict(checkpoint['worldModelOptimizer'])
        self.criticOptimizer.load_state_dict(checkpoint['criticOptimizer'])
        self.actorOptimizer.load_state_dict(checkpoint['actorOptimizer'])
        self.selfModelOptimizer.load_state_dict(checkpoint['selfModelOptimizer'])
        self.totalEpisodes = checkpoint['totalEpisodes']
        self.totalEnvSteps = checkpoint['totalEnvSteps']
        self.totalGradientSteps = checkpoint['totalGradientSteps']
        self.totalSelfModelSteps = checkpoint['totalSelfModelSteps']
        if self.config.useContinuationPrediction:
            self.continuePredictor.load_state_dict(checkpoint['continuePredictor'])

