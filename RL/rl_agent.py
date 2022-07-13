from __future__ import annotations

import sys
from typing import List

import multineat
import math

from dataclasses import dataclass

from revolve2.core.modular_robot import ModularRobot

from revolve2.genotypes.cppnwin import Genotype as CppnwinGenotype
from revolve2.core.database import IncompatibleError, Serializer
from revolve2.genotypes.cppnwin import GenotypeSerializer as CppnwinGenotypeSerializer
from revolve2.genotypes.cppnwin import crossover_v1 as cppnwin_crossover, mutate_v1 as cppnwin_mutate
from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_cpg_v1 import (
    random_v1 as brain_random,
)
from RL.rl_brain import RLbrain

from direct_tree.direct_tree_genotype import DirectTreeGenotype
from direct_tree.direct_tree_config import DirectTreeGenotypeConfig
from direct_tree.direct_tree_genotype import GenotypeSerializer as DirectTreeGenotypeSerializer
from direct_tree.direct_tree_mutation import mutate as direct_tree_mutate
from direct_tree.direct_tree_crossover import crossover as direct_tree_crossover
from direct_tree.random_tree_generator import generate_tree
from direct_tree.direct_tree_genotype import develop as body_develop

from revolve2.core.modular_robot import Core, Body, Module, Brick, ActiveHinge

import sqlalchemy
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

from direct_tree.direct_tree_utils import duplicate_subtree
from RL.actor_critic_network import ActorCritic

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
