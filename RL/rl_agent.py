from __future__ import annotations

from dataclasses import dataclass

from revolve2.core.modular_robot import ModularRobot

from RL.rl_brain import RLbrain

from direct_tree.direct_tree_genotype import DirectTreeGenotype
from direct_tree.direct_tree_genotype import develop as body_develop

import robot_zoo

@dataclass
class Agent:
    body: DirectTreeGenotype
    brain: RLbrain

def make_agent() -> Agent:
    body = robot_zoo.make_gecko_body()
    brain = None

    return Agent(body, brain)

def develop(agent: Agent) -> ModularRobot:
    body = body_develop(agent.body)
    brain = agent.brain
    return ModularRobot(body, brain)
