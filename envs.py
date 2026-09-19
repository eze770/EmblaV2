import gymnasium as gym
import numpy as np

def getEnvProperties(env):
    assert isinstance(env.action_space, gym.spaces.Box), "Sorry, supporting only continuous action space for now"
    observationShape = env.observation_space.shape
    actionSize = env.action_space.shape[0]
    actionLow = env.action_space.low.tolist()
    actionHigh = env.action_space.high.tolist()

    timestep = env.unwrapped.model.opt.timestep
    frame_skip = env.unwrapped.frame_skip
    dt = timestep * frame_skip
    return observationShape, actionSize, actionLow, actionHigh, dt

def in_energy_zone(env):
    env = env.unwrapped
    ant_pos = env.data.qpos[:2]
    zone_pos = env.model.site("energy_zone_1").pos[:2]
    distance = np.linalg.norm(ant_pos - zone_pos)
    return distance < 0.5  # Radius der Zone


def check_collision_with_obstacles(env):
    env = env.unwrapped
    obstacle_geom_ids = [env.model.geom("obstacle_1").id,
                         env.model.geom("obstacle_2").id]

    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        if contact.geom1 in obstacle_geom_ids or contact.geom2 in obstacle_geom_ids:
            return True
    return False

class GymPixelsProcessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observationSpace = self.observation_space
        newObsShape = observationSpace.shape[-1:] + observationSpace.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=newObsShape, dtype=np.float32)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))/255.0
        return observation
    
class CleanGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs

