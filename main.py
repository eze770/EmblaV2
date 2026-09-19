import cv2
import gymnasium as gym
import traceback
import torch
import argparse
from tqdm import tqdm
import os
from dreamer    import Dreamer
from utils      import loadConfig, seedEverything, plotMetrics
from envs       import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper
from utils      import saveLossesToCSV, ensureParentFolders
from func import self_model_forward, init_envs
import time
import threading
import queue
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smError = queue.Queue()  # threadsafe
score = queue.Queue()
print("Torch version:", torch.__version__)
print("device: ", device)


def main(configFile):
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName                 = f"{config.environmentName}_{config.runName}"
    checkpointToLoad        = os.path.join(config.folderNames.checkpointsFolder, f"{runName}/{config.checkpointToLoad}")
    crashFilenameBase       = config.folderNames.crashFolder
    ensureParentFolders(crashFilenameBase)
    os.makedirs(crashFilenameBase, exist_ok=True)

    # Only for getting the parameters. Env needs to be created in the thread that renders it (eze)
    smEnv, wmEnv = init_envs(config)
    observationShape, actionSize, actionLow, actionHigh, dt = getEnvProperties(smEnv)
    print(f"envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}, dt {dt}")

    dreamer = Dreamer(observationShape, actionSize, actionLow, actionHigh, dt, device, config.dreamer, config)
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)
    dreamer.environmentInteraction(wmEnv, smEnv, config.episodesBeforeStart, seed=config.seed)  # gather first training-data (eze)
    smEnv.close()
    wmEnv.close()

    train_t = threading.Thread(target=train, args=(config, dreamer, runName, observationShape, crashFilenameBase))
    env_t = threading.Thread(target=envInteraction, args=(config, dreamer, runName, crashFilenameBase))

    train_t.start()
    env_t.start()

    try:
        while env_t.is_alive():
            if not dreamer.smFrameQueue.empty():
                frame = dreamer.smFrameQueue.get_nowait()
                cv2.imshow("Live View", frame)
            if not dreamer.wmFrameQueue.empty():
                frame = dreamer.wmFrameQueue.get_nowait()
                cv2.imshow("Embla View", frame)
            cv2.waitKey(1)
    except Exception as e:
        with open(os.path.join(crashFilenameBase, runName + ".txt"), "a") as f:
            f.write(f"[main thread] {type(e).__name__}: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n" + "=" * 80 + "\n")
        raise

    train_t.join()
    env_t.join()
    print("\nTraining finished!")

def envInteraction(config, dreamer, runName, crashFilenameBase):
    try:
        videoFilenameBase       = os.path.join(config.folderNames.videosFolder,         runName)
        ensureParentFolders(videoFilenameBase)
        os.makedirs(videoFilenameBase, exist_ok=True)
        suffix = f"{dreamer.totalGradientSteps / 1000:.0f}k"

        while dreamer.totalGradientSteps <= config.gradientSteps:
            smEnv, wmEnv = init_envs(config)
            mostRecentScore, smLoss = dreamer.environmentInteraction(wmEnv, smEnv, config.numInteractionEpisodes, seed=config.seed, evaluation=False, saveVideo=False, liveView=True, dreamerLiveView=True, filename=f"{videoFilenameBase}/{suffix}")
            smError.put(smLoss)
            score.put(mostRecentScore)
            smEnv.close()
            wmEnv.close()
        cv2.destroyAllWindows()
    except Exception as e:
        with open(os.path.join(crashFilenameBase, runName + ".txt"), "a") as f:
            f.write(f"[envInteraction thread] {type(e).__name__}: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*80 + "\n")
        raise


def train(config, dreamer, runName, observationShape, crashFilenameBase):
    try:
        checkpointFilenameBase  = os.path.join(config.folderNames.checkpointsFolder,    runName)
        metricsFilename         = os.path.join(config.folderNames.metricsFolder,        runName)
        plotFilename            = os.path.join(config.folderNames.plotsFolder,          runName)
        ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase)
        os.makedirs(checkpointFilenameBase, exist_ok=True)

        damageDetected = 0
        smLatestLoss = 2.0
        iterationsNum = config.gradientSteps // config.replayRatio
        for _ in tqdm(range(iterationsNum), desc="OverallProgress", colour="green"):
            for i in tqdm(range(config.replayRatio), desc="Dream", colour="blue"):
                one = time.time()
                warmup = True if dreamer.totalGradientSteps < 500 else False
                sampledData                          = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength, damageDetected)
                if i % config.dreamer.smFreq == 0:
                    if (config.dreamer.selfModel.nIters // ((config.dreamer.batchLength - 1) * config.dreamer.batchSize)) - (dreamer.totalGradientSteps - damageDetected) >= 0 or smLatestLoss > 1.0:
                        smLatentStates, smLatestLoss, smMetrics = dreamer.selfModelTraining(sampledData)  # initialize SelfModel training, (eze)
                    else:
                        damageDetected = 0  # reset so that buffer uses all data for wm again, (eze)
                        with torch.no_grad():
                            smLatentStates                      = self_model_forward(config=config, model=dreamer.selfModel.eval(), arm_angle=sampledData.angles, output_flag=4, observation_shape=observationShape)
                    two = time.time()
                    initialStates, worldModelMetrics            = dreamer.worldModelTraining(sampledData, smLatentStates * config.dreamer.smToWmRatio)  # initial states also contains SM Latents (used for continuationpredictor), (eze)
                    three = time.time()
                if not warmup:  # Only start Actor training when SM training is finished, so that no wrong policy is learned, (eze)
                    behaviorMetrics                             = dreamer.behaviorTraining(initialStates)
                four = time.time()
                dreamer.totalGradientSteps += 1
                #print(f"SM: {two-one} WM: {three-two} Actor: {four-three}")

                if dreamer.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
                    while not score.empty():
                        mostRecentScore = score.get_nowait()
                    suffix = f"{dreamer.totalGradientSteps / 1000:.0f}k"
                    dreamer.saveCheckpoint(f"{checkpointFilenameBase}/{suffix}")
                    print(f"Saved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {mostRecentScore:>8.2f}")

            while not smError.empty():
                smLoss = smError.get_nowait()
                if smLoss > smLatestLoss*100:
                    print("\n", "-" * 100, "\nDamage detected!", smLoss, "::", smLatestLoss, "\n", "-" * 100, "\n")
                    damageDetected = dreamer.totalGradientSteps

            if config.saveMetrics and not warmup:
                while not score.empty():
                    mostRecentScore = score.get_nowait()
                metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "totalReward": mostRecentScore}
                saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics | smMetrics)
                plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{config.environmentName}")
    except Exception as e:
        with open(os.path.join(crashFilenameBase, runName + ".txt"), "a") as f:
            f.write(f"[train thread] {type(e).__name__}: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*80 + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="emblaAnt.yml")
    main(parser.parse_args().config)
