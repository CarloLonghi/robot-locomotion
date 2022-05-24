import math
import pickle
from random import Random
from typing import List, Tuple

import sqlalchemy
import torch
from actor_critic_genotype.actor_critic_network import ActorCritic
from actor_critic_genotype.rl_brain import RLbrain
from rl_agent import Agent, develop
from pyrr import Quaternion, Vector3
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

import revolve2.core.optimization.ea.generic_ea.population_management as population_management
import revolve2.core.optimization.ea.generic_ea.selection as selection
from revolve2.actor_controller import ActorController
from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import FloatSerializer
from revolve2.core.optimization import ProcessIdGen
from revolve2.core.optimization.ea.generic_ea import EAOptimizer
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    Environment,
    PosedActor,
    Runner,
)
from rl_runner_train import LocalRunner


class RLOptimizer():

    _runner: Runner

    _controller: ActorController

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    def __init__(
        self,
        rng: Random,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
    ) -> None:
        
        self._init_runner()
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency

    def _init_runner(self) -> None:
        self._runner = LocalRunner(LocalRunner.SimParams(), headless=False)

    def _control(self, dt: float, control: ActorControl, observations):
        num_agents = len(observations)
        actions = torch.zeros(num_agents,8)
        values = torch.zeros(num_agents)
        logps = torch.zeros(num_agents)
        for control_i in range(num_agents):
            action, value, logp = self._controller.get_dof_targets(observations[control_i])
            control.set_dof_targets(control_i, 0, action)
            actions[control_i] = torch.tensor(action)
            values[control_i] = value
            logps[control_i] = logp
        return actions, values, logps

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:

        # distance traveled on the xy plane
        return float(
            math.sqrt(
                (begin_state.position[0] - end_state.position[0]) ** 2
                + ((begin_state.position[1] - end_state.position[1]) ** 2)
            )
        )

    async def train(self, agents):
        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
            control=self._control,
        )

        brain = RLbrain()

        for agent_idx, agent in enumerate(agents):
            agent.brain = brain
            actor, controller = develop(agent).make_actor_and_controller()
            if agent_idx == 0:
                self._controller = controller
            bounding_box = actor.calc_aabb()
            env = Environment()
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                )
            )
            batch.environments.append(env)
        

        states = await self._runner.run_batch(batch, self._controller, len(agents))
        
        return 