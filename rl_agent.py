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
from actor_critic_genotype.rl_brain import RLbrain

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
from actor_critic_genotype.actor_critic_network import ActorCritic

def _make_direct_tree_config() -> DirectTreeGenotypeConfig:

    morph_single_mutation_prob = 0.2
    morph_no_single_mutation_prob = 1 - morph_single_mutation_prob  # 0.8
    morph_no_all_mutation_prob = morph_no_single_mutation_prob ** 4  # 0.4096
    morph_at_least_one_mutation_prob = 1 - morph_no_all_mutation_prob  # 0.5904

    brain_single_mutation_prob = 0.5

    tree_genotype_conf: DirectTreeGenotypeConfig = DirectTreeGenotypeConfig(
        max_parts=50,
        min_parts=10,
        max_oscillation=5,
        init_n_parts_mu=10,
        init_n_parts_sigma=4,
        init_prob_no_child=0.1,
        init_prob_child_block=0.4,
        init_prob_child_active_joint=0.5,
        mutation_p_duplicate_subtree=morph_single_mutation_prob,
        mutation_p_delete_subtree=morph_single_mutation_prob,
        mutation_p_generate_subtree=morph_single_mutation_prob,
        mutation_p_swap_subtree=morph_single_mutation_prob,
        mutation_p_mutate_oscillators=brain_single_mutation_prob,
        mutation_p_mutate_oscillator=0.5,
        mutate_oscillator_amplitude_sigma=0.3,
        mutate_oscillator_period_sigma=0.3,
        mutate_oscillator_phase_sigma=0.3,
    )

    return tree_genotype_conf


@dataclass
class Agent:
    body: DirectTreeGenotype
    brain: RLbrain

def make_super_ant_body() -> DirectTreeGenotype:
    body = Body()
    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = ActiveHinge(math.pi / 2)
    body.core.left.attachment.attachment = Brick(0.0)
    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = ActiveHinge(math.pi / 2)
    body.core.right.attachment.attachment = Brick(0.0)
    body.core.back = ActiveHinge(math.pi / 2)
    body.core.back.attachment = Brick(math.pi / 2)
    body.core.back.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.left.attachment = ActiveHinge(math.pi / 2)
    body.core.back.attachment.left.attachment.attachment = Brick(0.0)
    body.core.back.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.right.attachment = ActiveHinge(math.pi / 2)
    body.core.back.attachment.right.attachment.attachment = Brick(0.0)
    body.core.back.attachment.front = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment = Brick(math.pi / 2)
    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.left.attachment.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.right.attachment.attachment = Brick(0.0)

    return DirectTreeGenotype(body)

def make_ant_body() -> DirectTreeGenotype:
    body = Body()
    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)
    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)
    body.core.back = ActiveHinge(math.pi / 2)
    body.core.back.attachment = Brick(math.pi / 2)
    body.core.back.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.right.attachment = Brick(0.0)
    body.core.back.attachment.front = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment = Brick(math.pi / 2)
    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

    return DirectTreeGenotype(body)


def make_agent() -> Agent:
    body = make_ant_body()
    brain = None

    return Agent(body, brain)

def develop(agent: Agent) -> ModularRobot:
    body = body_develop(agent.body)
    brain = agent.brain
    return ModularRobot(body, brain)
