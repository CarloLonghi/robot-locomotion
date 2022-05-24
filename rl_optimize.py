from ast import arguments
import logging

import isaacgym
import torch

from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import ProcessIdGen
from rl_agent import make_agent
from rl_optimizer import RLOptimizer 
from random import Random

async def main() -> None:
    # number of initial mutations for body and brain CPPNWIN networks
    LEARNING_INTERACTIONS = 1e5
    SAMPLING_FREQUENCY = 8
    CONTROL_FREQUENCY = 8
    POPULATION_SIZE = 20
    SIMULATION_TIME = int(LEARNING_INTERACTIONS / (CONTROL_FREQUENCY * POPULATION_SIZE))

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    logging.info(f"Starting learning")

    # random number generator
    rng = Random()
    rng.seed(42)

    # database
    database = open_async_database_sqlite("./RLdatabases/test_1")

    # process id generator
    process_id_gen = ProcessIdGen()
    process_id = process_id_gen.gen()

    optimizer = RLOptimizer(
        rng=rng,
        sampling_frequency=SAMPLING_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        simulation_time=SIMULATION_TIME
    )

    # initialize agent population
    agents = [make_agent() for _ in range(POPULATION_SIZE)]
    
    logging.info("Starting learning process..")

    await optimizer.train(agents)

    logging.info(f"Finished learning.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
