import pandas as pd
import numpy as np

from RL_Agent1 import Autofeature_agent
from RL_environment1 import ISOFAEnvironment

dataset = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Drop the first column 'Id' since it just has serial numbers. Not useful in the prediction process.
dataset = dataset.iloc[:, 1:]
dataset["Cover_Type"] = dataset["Cover_Type"]  - 1

model_target = 0
max_try_num = 25
print("entra")
env = ISOFAEnvironment(dataset, 'Cover_Type', max_try_num)

# Parameters for the agent
learning_rate = 0.05
reward_decay = 0.9
e_greedy = 1
update_freq = 50
mem_cap = 400
BDQN_batch_size = 6

autodata = Autofeature_agent(env, BDQN_batch_size, learning_rate, reward_decay, e_greedy, update_freq, mem_cap)

print("Agent Ready!")

# Train the workload
autodata.train()

#autodata.plot_performance_vs_gradients() # This plots the recorded gradients and performance metrics
#autodata.plot_loss()

