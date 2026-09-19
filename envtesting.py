import gymnasium as gym
import imageio
from gymnasium.wrappers import AddRenderObservation
import numpy as np
import torch
import cv2
import matplotlib.image
import matplotlib.pyplot as plt
from torchvision.transforms import Resize
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = AddRenderObservation(gym.make("Ant-v5", render_mode="rgb_array", max_episode_steps=1000, camera_name="third_person"), render_only=True)
env.reset()
env2 = AddRenderObservation(gym.make("Pusher-v5", render_mode="rgb_array", max_episode_steps=200, camera_name="topdown_cam"), render_only=True)
env.reset()
#env.render()


def bla():
    while(True):
        cv2.waitKey(30)


th = threading.Thread(target=bla)
th.start()

#renderer = env.unwrapped.mujoco_renderer
#cam = renderer.viewer.cam
# cam.distance = 1.0
#print("Distance:", cam.distance)
#print("Azimuth:", cam.azimuth)
#print("Elevation:", cam.elevation)
#print("Lookat:", cam.lookat)

observationShape = env.observation_space.shape
actionSize = len(env.action_space.high.tolist())
print("actionSize: ", actionSize)
print(observationShape)
finalFilename = "test.mp4"
observations = np.empty((1000, *observationShape), dtype=np.float32)

idx = 0
terminated, truncated = False, False
while not terminated and not truncated:
    action = np.random.uniform(-1.5, 2.0, size=8)
    obs, reward, terminated, truncated, info = env.step(action)
    observations[idx] = obs
    qpos = env.unwrapped.data.qpos.copy()[:8]
    qvel = env.unwrapped.data.qvel.copy()
    idx = idx + 1
    #plt.imshow(obs)
    #plt.show()

    timestep = env.unwrapped.model.opt.timestep
    frame_skip = env.unwrapped.frame_skip

    dt = timestep * frame_skip

    frame = env.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("SM View", frame_bgr)
    cv2.waitKey(1)

#print("qpos: ", qpos, "qvel: ", qvel, "dt: ", dt, "reward: ", reward)

maskedObs = torch.zeros(1000, observationShape[0], observationShape[1])
unfilteredObs = torch.zeros(1000, observationShape[0], observationShape[1])
print(observations)
for t in range(len(observations)):
    hsvImg = cv2.cvtColor(observations[t], cv2.COLOR_RGB2HSV)
    #lower = np.array([140, 100, 30])
    #upper = np.array([170, 255, 60])
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 60])
    mask = cv2.inRange(hsvImg, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #if t == 0:
     #   matplotlib.image.imsave("test_afterMask.png", hsvImg)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    else:
        clean_mask = np.zeros_like(mask)
    unfilteredObs[t] = torch.from_numpy(clean_mask).float()
    maskedObs[t] = ((torch.from_numpy(clean_mask).float() / 255.0) > 0.5)
#observations = torch.as_tensor(observations[1], device=device).float()
#training_imges_snapshot = observations.view(-1, *observationShape)
#print(training_imges_snapshot)
print(maskedObs[1].shape)
matplotlib.image.imsave("test_afterContours.png", maskedObs[1])
#print(unfilteredObs[1, 240])
#print(maskedObs[1, 240])

with imageio.get_writer(finalFilename, fps=30) as video:
    maskedObs = torch.as_tensor(maskedObs)
    for ob in maskedObs:
        resize = Resize((32, 32))
        #ob = resize()
        matplotlib.image.imsave("test_32.png", ob)
        #video.append_data(ob)

env.close()
