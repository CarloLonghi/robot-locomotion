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
    SIMULATION_TIME = 60
    SAMPLING_FREQUENCY = 8
    CONTROL_FREQUENCY = 8

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    logging.info(f"Starting optimization")

    # random number generator
    rng = Random()
    rng.seed(5)

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

    # initialize agent
    agent = make_agent()

    logging.info("Starting optimization process..")

    await optimizer.train(agent)

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
