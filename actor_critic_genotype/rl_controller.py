from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt
import torch
from torch.optim import Adam

from revolve2.actor_controller import ActorController
from revolve2.serialization import SerializeError, StaticData

from interaction_buffer import Buffer
from .actor_critic_network import Actor, ActorCritic, Critic, ObservationEncoder


class RLcontroller(ActorController):
    _num_input_neurons: int
    _num_output_neurons: int
    _dof_ranges: npt.NDArray[np.float_]

    def __init__(
        self,
        actor_critic: Actor,
        dof_ranges: npt.NDArray[np.float_],
    ):
        """
        First num_output_neurons will be dof targets
        """
        self._actor_critic = actor_critic
        self._dof_ranges = dof_ranges
        self.device = torch.device("cuda:0")
        #self._actor_critic.to(self.device)
        params = [p for p in self._actor_critic.parameters() if p.requires_grad]
        self.optimizer = Adam(params, lr=1e-4)
        self._iteration_num = 1

    def get_dof_targets(self, observation) -> List[float]:
        observation = torch.tensor(observation)
        value, action, logp = self._actor_critic(observation)
        return list(
            np.clip(
                action[0],
                a_min=-self._dof_ranges,
                a_max=self._dof_ranges,
            )
        ), value, logp

    
    def train(self, buffer: Buffer):
        eps = 0.2
        print(f"\nITERATION NUM: {self._iteration_num}")
        self._iteration_num += 1
        for epoch in range(4):
            batch_sampler = buffer.get_sampler()

            ppo_losses = []
            val_losses = []
            losses = []
            for obs, val, act, logp_old, rew, adv, ret in batch_sampler:
                logp_old = logp_old.detach()
                adv = adv.detach()
                ret = ret.detach()
                value, action, logp = self._actor_critic(obs)
                ratio = torch.exp(logp - logp_old)
                obj1 = ratio * adv
                obj2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
                ppo_loss = -torch.min(obj1, obj2).mean()
                val_loss = 0.5 * (ret - value).pow(2).mean()
                #print(f"ret = {ret.mean()}, value = {value.mean()}")
                
                self.optimizer.zero_grad()
                loss = 0.5*val_loss + ppo_loss
                loss.backward()
                self.optimizer.step()
                ppo_losses.append(ppo_loss.item())
                val_losses.append(val_loss.item())
                losses.append(loss.item())

            print(f"EPOCH {epoch + 1} loss ppo:  {np.mean(ppo_losses)}, loss val: {np.mean(val_losses)}, final loss: {np.mean(losses)}")



    # TODO
    def step(self, dt: float):
        return

    def serialize(self) -> StaticData:
        return {
            "num_input_neurons": self._num_input_neurons,
            "num_output_neurons": self._num_output_neurons,
            "dof_ranges": self._dof_ranges.tolist(),
        }

    @classmethod
    def deserialize(cls, data: StaticData) -> RLcontroller:
        if (
            not type(data) == dict
            or not "actor_state" in data
            or not "critic_state" in data
            or not "encoder_state" in data
            or not "num_input_neurons" in data
            or not "num_output_neurons" in data
            or not "dof_ranges" in data
            or not all(type(r) == float for r in data["dof_ranges"])
        ):
            raise SerializeError()

        in_dim = data["num_input_neurons"]
        out_dim = data["num_output_neurons"]
        actor = Actor(in_dim, out_dim)
        actor.load_state_dict(data["actor_state"])
        critic = Critic(in_dim, out_dim)
        critic.load_state_dict(data["critic_state"])
        encoder = ObservationEncoder(in_dim)
        encoder.load_state_dict(data["encoder_state"])
        network = ActorCritic(in_dim, out_dim)
        network.actor = actor
        network.critic = critic
        network.encoder = encoder
        return RLcontroller(
            network,
            np.array(data["dof_ranges"]),
        )
