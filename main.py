import pandas as pd

from RL_Agent1 import Autofeature_agent
from RL_environment1 import ISOFAEnvironment

paths = ["data/Customer Flight Activity.csv", "data/Customer Loyalty History.csv"]
# temp#####

t_core: pd.DataFrame = pd.read_csv(paths[0])
t_core = t_core[t_core["Year"] == 2018].groupby("Loyalty Number").sum().reset_index()
t_core.drop(['Month', 'Year'], axis=1, inplace=True)
t_candidate = pd.read_csv(paths[1])
t_candidate["Cancellation Year"] = t_candidate["Cancellation Year"].apply(
    lambda x: 1 if pd.notna(x) else 0)

t_core = t_core.merge(t_candidate, on='Loyalty Number', how='left')

for feat in t_core:
    if t_core[feat].dtype == 'object':
        t_core[feat] = t_core[feat].fillna('Unknown')
        t_core[feat] = t_core[feat].astype('category')
    else:
        t_core[feat] = t_core[feat].fillna(-1)  # TEMP STRATEGY FOR NAN VALUES

t_core.drop(['Cancellation Month'], axis=1, inplace=True)


model_target = 0
max_try_num = 3
print("entra")
env = ISOFAEnvironment(t_core, 'Cancellation Year', max_try_num)

# Parameters for the agent
learning_rate = 0.05
reward_decay = 0.9
e_greedy = 1
update_freq = 50
mem_cap = 1000
BDQN_batch_size = 3

autodata = Autofeature_agent(env, BDQN_batch_size, learning_rate, reward_decay, e_greedy, update_freq, mem_cap)

print("Agent Ready!")

# Train the workload
autodata.train()

autodata.plot_performance_vs_gradients() # This plots the recorded gradients and performance metrics
autodata.plot_loss()
autodata.plot_gradients()

