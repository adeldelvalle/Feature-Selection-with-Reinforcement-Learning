import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


class Net(nn.Module):
    def __init__(self, feature_action_num):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feature_action_num, 256)
        self.fc2 = nn.Linear(256, 128)
        self.ft_fc = nn.Linear(128, feature_action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.ft_fc(x)
        return x


class Autofeature_agent:
    def __init__(self, env, learning_rate=0.05, gamma=0.9, epsilon=1.0, update_freq=50, mem_cap=100000, batch_size=32):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.memory_capacity = mem_cap
        self.mem = np.zeros((mem_cap, 2 * env.state_len + 2))  # state, action, reward, next_state
        self.mem_counter = 0
        self.learning_step_counter = 0

        self.eval_net = Net(env.state_len)
        self.target_net = Net(env.state_len)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        if np.random.uniform() > self.epsilon:
            print("choose by network")
            self.eval_net.eval()
            with torch.no_grad():
                action_values = self.eval_net(state)
            action = action_values.argmax().item()
        else:
            print("choose by random")
            action = random.randrange(self.env.state_len)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.mem_counter % self.memory_capacity
        self.mem[index, :] = transition
        self.mem_counter += 1

    def learn(self):
        if self.mem_counter < self.batch_size:
            return
        sample_index = np.random.choice(min(self.mem_counter, self.memory_capacity), self.batch_size)
        batch_memory = self.mem[sample_index, :]
        batch_state = torch.tensor(batch_memory[:, :self.env.state_len], dtype=torch.float)
        batch_action = torch.tensor(batch_memory[:, self.env.state_len].astype(int), dtype=torch.long)
        batch_reward = torch.tensor(batch_memory[:, self.env.state_len + 1], dtype=torch.float)
        batch_next_state = torch.tensor(batch_memory[:, -self.env.state_len:], dtype=torch.float)

        q_eval = self.eval_net(batch_state).gather(1, batch_action.unsqueeze(1))
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward.unsqueeze(1) + self.gamma * q_next.max(1)[0].unsqueeze(1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learning_step_counter % self.update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_step_counter += 1

    def train(self):
        cut_off = 30
        episode_num = 40
        const_a = math.pow(0.001, 1 / episode_num)

        for episode in range(episode_num):  # Adjust number of episodes as needed
            self.env.reset()
            state = self.env.cur_state
            total_reward = 0
            while True:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.store_transition(state, action, reward, next_state)
                total_reward += reward
                if done:
                    break
                if self.mem_counter > self.batch_size:
                    self.learn()

                state = next_state

            print("Final set of features:", self.env.current_training_set.columns)
            if self.epsilon >= episode_num - cut_off:
                self.epsilon = 0.0001
            else:
                self.epsilon = math.pow(const_a, episode)

            print(f"Episode {episode + 1}, Total reward: {total_reward}")
