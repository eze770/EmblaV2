import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation
import torch
import argparse
from tqdm import tqdm
import os
from dreamer    import Dreamer
from utils      import loadConfig, seedEverything, plotMetrics
from envs       import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper
from utils      import saveLossesToCSV, ensureParentFolders
from func import selfmodelEvalForward
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch version:", torch.__version__)
print("device: ", device)


def main(configFile):
    threads = []
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName                 = f"{config.environmentName}_{config.runName}"
    checkpointToLoad        = os.path.join(config.folderNames.checkpointsFolder, f"{runName}/{config.checkpointToLoad}")
    metricsFilename         = os.path.join(config.folderNames.metricsFolder,        runName)
    plotFilename            = os.path.join(config.folderNames.plotsFolder,          runName)
    checkpointFilenameBase  = os.path.join(config.folderNames.checkpointsFolder,    runName)
    videoFilenameBase       = os.path.join(config.folderNames.videosFolder,         runName)
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)
    os.makedirs(checkpointFilenameBase, exist_ok=True)
    os.makedirs(videoFilenameBase, exist_ok=True)

    camMode = config.camMode
    smEnv             = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(AddRenderObservation(gym.make(config.environmentName, render_mode="rgb_array", max_episode_steps=config.maxEnvSteps, camera_name="track", forward_reward_weight=0), render_only=True), (64, 64))))
    if camMode == 1:
        camName = "ego_cam"
        wmEnv         = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(AddRenderObservation(gym.make(config.environmentName, render_mode="rgb_array", max_episode_steps=config.maxEnvSteps, camera_name=camName, forward_reward_weight=0), render_only=True), (64, 64))))
    elif camMode == 2:
        camName = "topdown_cam"
        wmEnv         = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(AddRenderObservation(gym.make(config.environmentName, render_mode="rgb_array", max_episode_steps=config.maxEnvSteps, camera_name=camName, forward_reward_weight=0), render_only=True), (64, 64))))
    else:
        wmEnv = None

    observationShape, actionSize, actionLow, actionHigh, dt = getEnvProperties(wmEnv if wmEnv else smEnv)
    print(f"envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}, dt {dt}")
    damageDetected = 0
    smLatestLoss = 2.0

    dreamer = Dreamer(observationShape, actionSize, actionLow, actionHigh, dt, device, config.dreamer, config)
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)

    selfmodelEvalForward(config=config, observationShape=observationShape, data=torch.as_tensor([0, 0, 0, 0, 0, 0, 0]), initializeLatents=True)  # save one random latent state for Dreamer start, (eze)
    dreamer.environmentInteraction(wmEnv, smEnv, config.episodesBeforeStart, seed=config.seed)

    iterationsNum = config.gradientSteps // config.replayRatio
    for _ in tqdm(range(iterationsNum), desc="OverallProgress", colour="green"):
        for i in tqdm(range(config.replayRatio), desc="Dream", colour="blue"):
            one = time.time()
            warmup = True if dreamer.totalGradientSteps < 500 else False
            sampledData                          = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength, damageDetected)
            if i % config.dreamer.smFreq == 0:
                if (config.dreamer.selfModel.nIters // ((config.dreamer.batchLength - 1) * config.dreamer.batchSize)) - (dreamer.totalGradientSteps - damageDetected) >= 0 or smLatestLoss > 1.0:
                    smLatentStates, smLatestLoss, smMetrics = dreamer.selfModelTraining(sampledData, config.resume)  # initialize SelfModel training, (eze)
                else:
                    damageDetected = 0  # reset so that buffer uses all data for wm again, (eze)
                    with torch.no_grad():
                        smLatentStates                      = selfmodelEvalForward(config=config, observationShape=observationShape, data=sampledData.angles)
                two = time.time()
                initialStates, worldModelMetrics            = dreamer.worldModelTraining(sampledData, smLatentStates * config.dreamer.smToWmRatio)  # initial states also contains SM Latents (used for continuationpredictor), (eze)
                three = time.time()
            if not warmup:  # Only start Actor training when SM training is finished, so that no wrong policy is learned, (eze)
                behaviorMetrics                             = dreamer.behaviorTraining(initialStates)
            four = time.time()
            dreamer.totalGradientSteps += 1
            #print(f"SM: {two-one} WM: {three-two} Actor: {four-three}")

            if dreamer.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
                suffix = f"{dreamer.totalGradientSteps/1000:.0f}k"
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}/{suffix}")
                evaluationScore, _ = dreamer.environmentInteraction(wmEnv, smEnv, config.numEvaluationEpisodes, seed=config.seed, evaluation=True, saveVideo=True, filename=f"{videoFilenameBase}/{suffix}")
                print(f"Saved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluationScore:>8.2f}")

        mostRecentScore, smLoss = dreamer.environmentInteraction(wmEnv, smEnv, config.numInteractionEpisodes, seed=config.seed)
        if smLoss > smLatestLoss*100:
            print("\n", "-" * 100, "\nDamage detected!", smLoss, "::", smLatestLoss, "\n", "-" * 100, "\n")
            damageDetected = dreamer.totalGradientSteps
        if config.saveMetrics and not warmup:
            metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "totalReward" : mostRecentScore}
            saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics | smMetrics)
            plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{config.environmentName}")

        #smEnv.close()
        #if wmEnv:
        #    wmEnv.close()
        #    wmEnv = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(AddRenderObservation(gym.make(config.environmentName, render_mode="rgb_array", max_episode_steps=config.maxEnvSteps, camera_name=camName), render_only=True), (64, 64))))
        #smEnv = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(AddRenderObservation(gym.make(config.environmentName, render_mode="rgb_array", max_episode_steps=config.maxEnvSteps), render_only=True), (64, 64))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="emblaAnt.yml")
    main(parser.parse_args().config)
