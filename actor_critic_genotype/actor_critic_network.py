from turtle import forward
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from typing import List

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        """
        Network that represents the actor 
        args:
            obs_dim: dimension of the observation in input to the newtork
            act_dim: number of actuators in output
        """
        super().__init__()
        self.mean_layer = nn.Linear(obs_dim, act_dim)
        #self.std_layer = nn.Parameter(torch.zeros(1, act_dim))
        self.std_layer = nn.Linear(obs_dim, act_dim)

    def forward(self, obs):
        mean = self.mean_layer(obs)
        #std = torch.exp(self.std_layer)
        std = torch.exp(self.std_layer(obs))
        action_prob = Normal(mean, std)
        action = action_prob.sample()
        logp = action_prob.log_prob(action).sum(-1, keepdim=True)
        return action, logp

class Critic(nn.Module):
    def __init__(self, obs_dim):
        """
        Network that represents the Critic
        args:
            obs_dim: dimension of the observation in input to the newtork
        """
        super().__init__()
        self.critic_layer = nn.Linear(obs_dim,1)

    def forward(self, obs):
        return self.critic_layer(obs)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        """
        args:
            obs_dim: dimension of the observation in input to the newtork
            act_dim: number of actuators in output
        """
        super().__init__()
        self.val_encoder = ObservationEncoder(obs_dim=obs_dim)
        self.pi_encoder = ObservationEncoder(obs_dim=obs_dim)
        self.actor = Actor(obs_dim=64, act_dim=act_dim)
        self.critic = Critic(obs_dim=64)
    
    def forward(self, obs: List[float]):
        val_obs = self.val_encoder(obs)
        pi_obs = self.pi_encoder(obs)
        action, logp = self.actor(pi_obs)
        value = self.critic(val_obs)
        return value, action, logp

class ObservationEncoder(nn.Module):
    def __init__(self, obs_dim):
        """
        Encoder for the observations
        TODO for now we only have the hinges position as observation
        args:
            obs_dim: dimension of the observation in input to the newtork
        """
        super().__init__()
        dims = [obs_dim] + [64,64]
        
        self.encoder = nn.Sequential()
        for n, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.encoder.add_module(name=f"observation_encoder_{n}", module=nn.Linear(in_features=dim_in, out_features=dim_out))

    def forward(self, obs):
        return self.encoder(obs)
