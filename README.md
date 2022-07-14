# Robot Locomotion
Collection of learning algorithms for robot locomotion using Revolve2

### Setup of the training environment

1. Download isaacgym from https://developer.nvidia.com/isaac-gym
2. pip install \<isaacgym path\>/python
3. git clone --branch v0.2.5-alpha3 https://github.com/ci-group/revolve2
4. pip install \<revolve2 path\>/core
5. pip install \<revolve2 path\>/runners/isaacgym
6. pip install \<revolve2 path\>/genotypes/cppnwin

### To train the robot using the Proximal Policy Optimization (PPO) algorithm:

1. Run rl_optimize.py (optional parameters are --visualize to make the simulation visible and
--from_checkpoint to restart the learning task from a previous checkpoint)
2. Run plot_statistics.py to visualize the mean action reward and state value for each iteration
3. Run rl_rerun_best.py to rerun the last agent

Check out the report on the implementation [here](report.pdf).

### To train the robot using a Genetic Algorithm (GA):
1. Run ga_optimize.py
2. Run ga_rerun_best.py to rerun the best agent

<img src="videos/video_readme.gif" width="800">
