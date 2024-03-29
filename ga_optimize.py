import logging
from random import Random

import multineat
from GA.genotype import Genotype
from GA.optimizer import Optimizer

from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import ProcessIdGen
from GA.genotype import random as random_genotype

async def main() -> None:
    # number of initial mutations for body and brain CPPNWIN networks
    NUM_INITIAL_MUTATIONS = 10

    SIMULATION_TIME = 32
    SAMPLING_FREQUENCY = 4
    CONTROL_FREQUENCY = 4

    POPULATION_SIZE = 64
    OFFSPRING_SIZE = 32
    NUM_GENERATIONS = 100

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    logging.info(f"Starting optimization")

    # random number generator
    rng = Random()
    rng.seed(5)

    # database
    database = open_async_database_sqlite("./GA/databases/test")

    # process id generator
    process_id_gen = ProcessIdGen()

    # multineat innovation databases
    innov_db_brain = multineat.InnovationDatabase()
    innov_db_body = multineat.InnovationDatabase()

    initial_population = [
        random_genotype(innov_db_brain, rng, NUM_INITIAL_MUTATIONS)
        for _ in range(POPULATION_SIZE)
    ]

    process_id = process_id_gen.gen()
    maybe_optimizer = await Optimizer.from_database(
        database=database,
        process_id=process_id,
        innov_db_brain=innov_db_brain,
        rng=rng,
        process_id_gen=process_id_gen,
    )
    if maybe_optimizer is not None:
        optimizer = maybe_optimizer
    else:
        optimizer = await Optimizer.new(
            database=database,
            process_id=process_id,
            initial_population=initial_population,
            rng=rng,
            process_id_gen=process_id_gen,
            innov_db_brain=innov_db_brain,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            num_generations=NUM_GENERATIONS,
            offspring_size=OFFSPRING_SIZE,
        )

    logging.info("Starting optimization process..")

    await optimizer.run()

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
