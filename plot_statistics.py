import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('RL/model_states/statistics.csv')

plt.figure(figsize=(13,7))
plt.subplot(1,2,1)
plt.plot(data['mean_rew'])
plt.subplot(1,2,2)
plt.plot(data['mean_val'])
plt.show()