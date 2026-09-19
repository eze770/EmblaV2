import attridict
import numpy as np
import torch

# Code comes from SimpleDreamer repo, I only changed some formatting and names, but I should really remake it.
class ReplayBuffer(object):
    def __init__(self, observation_shape, actions_size, config, device):
        self.config = config
        self.device = device
        self.capacity = int(self.config.capacity)

        self.observations        = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.smObservations      = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.nextObservations    = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.nextSmObservations  = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.actions             = np.empty((self.capacity, actions_size), dtype=np.float32)
        self.rewards             = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones               = np.empty((self.capacity, 1), dtype=np.float32)
        self.angles              = np.empty((self.capacity, actions_size), dtype=np.float32)  # actions_size is equivalent to number of angles, (eze)
        #self.vel                 = np.empty((self.capacity, actions_size), dtype=np.float32)

        self.bufferIndex = 0
        self.full = False
        
    def __len__(self):
        return self.capacity if self.full else self.bufferIndex

    def add(self, observation, smObservation, action, reward, nextObservation, nextSmObservation, done, angle):
        self.observations[self.bufferIndex]       = observation
        self.smObservations[self.bufferIndex]     = smObservation
        self.actions[self.bufferIndex]            = action
        self.rewards[self.bufferIndex]            = reward
        self.nextObservations[self.bufferIndex]   = nextObservation
        self.nextSmObservations[self.bufferIndex] = nextSmObservation
        self.dones[self.bufferIndex]              = done
        self.angles[self.bufferIndex]             = angle.cpu()
        #self.vel[self.bufferIndex]                = vel.cpu()

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or self.bufferIndex == 0

    def sample(self, batchSize, sequenceSize, anomalyDetected):
        lastFilledIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastFilledIndex > batchSize), "not enough data in the buffer to sample"
        sampleIndex = np.random.randint(anomalyDetected, self.capacity if self.full else lastFilledIndex, batchSize).reshape(-1, 1)  # only uses postanomaly-data if damage is detected. Is 0 again after sm training, (eze)
        sequenceLength = np.arange(sequenceSize).reshape(1, -1)

        sampleIndex = (sampleIndex + sequenceLength) % self.capacity

        observations        = torch.as_tensor(self.observations[sampleIndex], device=self.device).float()
        smObservations      = torch.as_tensor(self.observations[sampleIndex], device=self.device).float()
        nextObservations    = torch.as_tensor(self.nextObservations[sampleIndex], device=self.device).float()
        nextSmObservations  = torch.as_tensor(self.nextSmObservations[sampleIndex], device=self.device).float()

        actions  = torch.as_tensor(self.actions[sampleIndex], device=self.device)
        rewards  = torch.as_tensor(self.rewards[sampleIndex], device=self.device)
        dones    = torch.as_tensor(self.dones[sampleIndex], device=self.device)
        angles   = torch.as_tensor(self.angles[sampleIndex], device=torch.device(self.device))
        #vel      = torch.as_tensor(self.vel[sampleIndex], device=torch.device(self.device))  # May be redundant in later versions with SM rollouts, (eze)

        sample = attridict({
            "observations"      : observations,
            "smObservations"    : smObservations,
            "actions"           : actions,
            "rewards"           : rewards,
            "nextObservations"  : nextObservations,
            "nextSmObservations": nextSmObservations,
            "dones"             : dones,
            "angles"            : angles})
        return sample


    def getIndex(self):
        return self.bufferIndex

