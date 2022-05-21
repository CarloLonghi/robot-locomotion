from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt
import torch

from revolve2.actor_controller import ActorController
from revolve2.serialization import SerializeError, StaticData
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
        self._actor = actor_critic.actor
        self._critic = actor_critic.critic
        self._obs_encoder = actor_critic.encoder
        self._dof_ranges = dof_ranges

    def get_dof_targets(self, observation) -> List[float]:
        observation = torch.tensor(observation)
        action, value = self._actor_critic(observation)
        action = action.sample()
        return list(
            np.clip(
                action[0],
                a_min=-self._dof_ranges,
                a_max=self._dof_ranges,
            )
        )

    # TODO
    def train_step(gradient):
        return

    # TODO
    def step(self, dt: float):
        return

    def serialize(self) -> StaticData:
        return {
            "actor_state": self._actor.state_dict(),
            "critic_state": self._critic.state_dict(),
            "encoder_state": self._obs_encoder.state_dict(),
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
