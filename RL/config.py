
PPO_CLIP_EPS = 0.2
PPO_LAMBDA = 0.95
PPO_GAMMA = 0.99

ACTOR_LOSS_COEFF = 1
CRITIC_LOSS_COEFF = 0.25
ENTROPY_COEFF = 0.01

NUM_STEPS = 128

NUM_OBS_TIMES = 3

LR_ACTOR = 2e-3
LR_CRITIC = 1e-3

BATCH_SIZE = 2048
N_EPOCHS = 4

NUM_PARALLEL_AGENT = 64

NUM_OBSERVATIONS = 1
OBS_DIM = (8*NUM_OBS_TIMES,)