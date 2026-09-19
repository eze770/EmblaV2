import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits
from typing import Union, Tuple
from utils import sequentialModel1D


class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize, latentSize, actionSize, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)()

        self.linear = nn.Linear(latentSize + actionSize, self.config.hiddenSize)
        self.recurrent = nn.GRUCell(self.config.hiddenSize, recurrentSize)

    def forward(self, recurrentState, latentState, action):
        return self.recurrent(self.activation(self.linear(torch.cat((latentState, action), -1))), recurrentState)


class PriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)
    
    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits
    

class PosteriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)
    
    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits


class RewardModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)  # MLP

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))


class ContinueModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 1, self.config.activation)

    def forward(self, x):
        return Bernoulli(logits=self.network(x).squeeze(-1))


class EncoderConv(nn.Module):
    def __init__(self, inputShape, outputSize, config):
        super().__init__()
        self.config = config
        activation = getattr(nn, self.config.activation)()
        channels, height, width = inputShape
        self.outputSize = outputSize

        self.convolutionalNet = nn.Sequential(
            nn.Conv2d(channels,            self.config.depth*1, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*1, self.config.depth*2, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*2, self.config.depth*4, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*4, self.config.depth*8, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Flatten(),
            nn.Linear(self.config.depth*8*(height // (self.config.stride ** 4))*(width // (self.config.stride ** 4)), outputSize), activation)

    def forward(self, x):
        return self.convolutionalNet(x).view(-1, self.outputSize)


class DecoderConv(nn.Module):
    def __init__(self, inputSize, outputShape, config):
        super().__init__()
        self.config = config
        self.channels, self.height, self.width = outputShape
        activation = getattr(nn, self.config.activation)()

        self.network = nn.Sequential(
            nn.Linear(inputSize, self.config.depth*32),
            nn.Unflatten(1, (self.config.depth*32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(self.config.depth*32, self.config.depth*4, self.config.kernelSize,     self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*4,  self.config.depth*2, self.config.kernelSize,     self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*2,  self.config.depth*1, self.config.kernelSize + 1, self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*1,  self.channels,       self.config.kernelSize + 1, self.config.stride))

    def forward(self, x):
        return self.network(x)


class SmAuxiliaryDecoder(nn.Module):
    def __init__(self, fullStateSize, smLatentSize):
        super().__init__()
        self.sm_auxiliary_decoder = nn.Sequential(
            nn.Linear(fullStateSize, 256),  # Fullstate size as Input
            nn.ReLU(),
            nn.Linear(256, smLatentSize)  # SM latent size as Output
        )

    def forward(self, x):
        return self.sm_auxiliary_decoder(x)


class Actor(nn.Module):
    def __init__(self, inputSize, actionSize, actionLow, actionHigh, device, config):
        super().__init__()
        actionSize *= 2
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, actionSize, self.config.activation)
        self.register_buffer("actionScale", ((torch.tensor(actionHigh, device=device) - torch.tensor(actionLow, device=device)) / 2.0))
        self.register_buffer("actionBias", ((torch.tensor(actionHigh, device=device) + torch.tensor(actionLow, device=device)) / 2.0))

    def forward(self, x, training=False):
        logStdMin, logStdMax = -5, 2
        mean, logStd = self.network(x).chunk(2, dim=-1)
        logStd = logStdMin + (logStdMax - logStdMin)/2*(torch.tanh(logStd) + 1) # (-1, 1) to (min, max)
        std = torch.exp(logStd)

        distribution = Normal(mean, std)
        sample = distribution.sample()
        sampleTanh = torch.tanh(sample)
        action = sampleTanh*self.actionScale + self.actionBias
        if training:
            logprobs = distribution.log_prob(sample)
            logprobs -= torch.log(self.actionScale*(1 - sampleTanh.pow(2)) + 1e-6)
            entropy = distribution.entropy()
            return action, logprobs.sum(-1), entropy.sum(-1)
        else:
            return action


class Critic(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1)) # twohot loss in normal dreamer


# scales and moves fullstates with energy
class FiLMLayer(nn.Module):
    def __init__(self, fullStateSize):
        super().__init__()
        self.gamma = nn.Linear(1, fullStateSize)
        self.beta = nn.Linear(1, fullStateSize)

    def forward(self, fullstate, energy):
        return self.gamma(energy) * fullstate + self.beta(energy)


# ----------------------------------------------------------------
# From here on are models for SM, (eze)
# ----------------------------------------------------------------
class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(
            self,
            d_input: int,
            n_freqs: int,
            log_space: bool = False
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(
            self,
            x
    ) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class FBV_SM(nn.Module):
    def __init__(self,
                 config,
                 encoder=None,
                 d_input: int = 5,
                 d_filter: int = 128,
                 output_size: int = 2):
        super(FBV_SM, self).__init__()

        self.d_input = d_input
        self.act = nn.functional.relu
        self.encoder = encoder
        self.config = config
        self.d_filter = d_filter

        # Initialize layers
        if self.encoder ==None:
            pos_encoder_d = 3
            cmd_encoder_d = d_input-3
            self.feed_forward = nn.Sequential(
                nn.Linear(d_filter * 2+d_input, d_filter),
                nn.ReLU(),
                nn.Linear(d_filter, d_filter // 4)
            )
        else:
            n_freqs = self.encoder.n_freqs
            pos_encoder_d = (n_freqs*2+1)*3
            cmd_encoder_d = (n_freqs*2+1)*(d_input-3)
            self.feed_forward = nn.Sequential(
                nn.Linear(d_filter*2, d_filter),
                nn.ReLU(),
                nn.Linear(d_filter,d_filter//4)
            )

        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_encoder_d, d_filter),
            nn.ReLU(),
            nn.Linear(d_filter,d_filter),
        )

        self.cmd_encoder = nn.Sequential(
            nn.Linear(cmd_encoder_d, d_filter),
            nn.ReLU(),
            nn.Linear(d_filter,d_filter),
        )

        self.output = nn.Linear(d_filter//4, output_size)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.encoder !=None:
            x_pos = self.encoder(x[:,:3])
            x_cmd = self.encoder(x[:,3:])
            x_pos = self.pos_encoder(x_pos)
            x_cmd = self.cmd_encoder(x_cmd)
            x = self.feed_forward(torch.cat((x_pos,x_cmd),dim=1))
        else:
            x_pos = self.pos_encoder(x[:,:3])
            x_cmd = self.cmd_encoder(x[:,3:])
            x = self.feed_forward(torch.cat((x_pos, x_cmd,x), dim=1,))
        return self.output(x), x.detach()  # x = latent, (eze)

