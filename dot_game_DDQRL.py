# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:47:57 2020

@author: William Woodall
"""


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import os
import pickle

import the_dot_game

## Stats directory for training display
if not os.path.isdir('stats'):
    os.makedirs('stats')
    
    
env = the_dot_game.game
env.food_count = 1
env.enemy_count = 0


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, mem_size, input_shape):
        self._input_shape = input_shape
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.mem_size = mem_size
        self.states = np.zeros((mem_size, *self.input_shape))
        self.actions = np.zeros(mem_size)
        self.rewards = np.zeros(mem_size)
        self.states_ = np.zeros((mem_size, *self.input_shape))
        self.dones = np.zeros(mem_size)
        self.mem_counter = 0
        self.index = 0
        
    def update(self, state, action, reward, state_, done):       
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.states_[self.index] = state_
        self.dones[self.index] = done
        self.mem_counter += 1
        self.index = self.mem_counter % self.mem_size
        
    def fetch_batch(self, batch_size):
        mem_max = min(self.mem_size, self.mem_counter)
        batch_indecies = np.random.choice(mem_max, batch_size, replace=False)
        states = self.states[batch_indecies]
        actions = self.actions[batch_indecies]
        rewards = self.rewards[batch_indecies]
        states_ = self.states_[batch_indecies]
        dones =  self.dones[batch_indecies]
        
        return states, actions, rewards, states_, dones
 
    def discount_rewards(self, steps, gamma, tao=0.9):
        last_step = self.index - 1
        first_step = last_step - steps
        reward = 0
        for i in reversed(range(first_step, last_step+1)):
            reward = (reward + self.rewards[i] * gamma) * tao
            self.rewards[i] = reward
        
    def memory_clear(self):
        self.__init__(self.mem_size, self._input_shape)
        
class DQNetwork(nn.Module):
    def __init__(self, inout_shape, num_actions, size_x=env.size_x, size_y=env.size_y):
        super(DQNetwork, self).__init__()
        self.conv_out_dims = 32*(size_x-3)*(size_y-3)
        self.fc1_dims = 512
        self.fc2_dims = 256
        
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(2,2), stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(2,2), stride=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(2,2), stride=1)
        self.batch_norm = nn.BatchNorm2d(16)
        
        self.fc1 = nn.Linear(self.conv_out_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        self.value = nn.Linear(self.fc2_dims, 1)
        self.advantage = nn.Linear(self.fc2_dims, num_actions)
        
    def forward(self, x):
        x = self.conv1(x)
        #x = F.dropout(x, 0.2)
        x = F.relu(x)
        x = self.conv2(x)
        #x = F.dropout(x, 0.2)
        x = F.relu(x)
        x = self.conv3(x)
        #x = F.dropout(x, 0.2)
        x = F.relu(x)
        
        x = x.view(-1, self.conv_out_dims)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        value = self.value(x)
        advantage = self.advantage(x)
        
        q_vals = value + (advantage - torch.mean(advantage))
        return advantage

  
class DQAgent:
    def __init__(self, input_shape, num_actions, mem_size=100_000, new_model=True):
        self.num_actions = num_actions
        self.lr = 1e-3
        self.batch_size = 256
        self.gamma = 0.99
        self.epsilon = 0.4
        self.epsilon_decay = 7e-7
        self.epsilon_min = 0.05
        self.learn_step = 0
        self.update_freq = 1000
        self.tau = .05
        
        self.mem = ReplayBuffer(mem_size, input_shape)
        
        if new_model:
            self.model = DQNetwork(input_shape, num_actions).to(device)
            self.target_model = DQNetwork(input_shape, num_actions).to(device)
            self.target_model.load_state_dict(self.target_model.state_dict())
        else:
            self.model = torch.load('dot_model.pth')
            self.target_model = torch.load('dot_model.pth')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def choose_action(self, state):       
        if random.random() < self.epsilon:
            action = np.random.choice(self.num_actions)
            q_vals = []
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            q_vals = self.model(state)
            action = torch.argmax(q_vals).item()
            
        return action
    
    def remember(self, state, action, reward, state_, done):
        self.mem.update(state,action,reward,state_, done)
    
    def process_img(self, img):
        img = img / 255
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
        return img
    
    def update_target_params(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) 
            
    def learn(self):
        if self.mem.mem_counter < self.batch_size:
            return [], [], []
    
        states, actions, rewards, states_, dones = self.mem.fetch_batch(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        states_ = torch.tensor(states_, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)
        
        batch_indecies = np.arange(self.batch_size)
        
        q_vals = self.model(states)
        q_vals_ = self.model(states_)
        target_vals = self.target_model(states_)
        
        old_q_vals = q_vals[batch_indecies, actions]
        future_actions = torch.max(q_vals_, dim=1)[1]
        
        q_targets = target_vals[batch_indecies, future_actions]
        q_targets[dones] = 0.0 
        
        q_targets = (q_targets - q_targets.mean())/q_targets.std() 
        
        q_targets = rewards + q_targets * self.gamma
        
        td = q_targets - old_q_vals
        
        self.optimizer.zero_grad()
        loss = (td**2).mean()
        loss.backward()
        self.optimizer.step()
        
        if not self.learn_step % self.update_freq and False:
            print('... updateing weights...')
            self.target_model.load_state_dict(self.model.state_dict())
            self.update_target_params()
        self.update_target_params()
        #print('.', end='')
        self.learn_step += 1
        
        return loss.item(), td.mean().item(), np.mean(target_vals.detach().cpu().numpy(), axis=0)    



agent = DQAgent(env.observation_space, 9)  

stats_dict = {'episode': [],
              'min': [],
              'max': [],
              'average': [],
              'moving average': [],
              'epsilon': [],
              'learn_rate': [],
              'model_loss': [],
              'temporal_difference': [],
              }

q_val_dict = {}
for i in range(env.action_space):
    q_val_dict[f"{i}"] = []


stats_update = 5
high_score = -1_000_000
score_history = []
loss_history = []
save_score = -.5

episode = 0
while True:
    
    loss_ = 0
    done = False
    state = env.reset()
    state = agent.process_img(state)
    score = 0
    steps = 0
    q_val_track = []
    while not done:
        env.render() 
        action = agent.choose_action(state)
        state_, reward, done = env.step(action)
        state_ = agent.process_img(state_)
        agent.remember(state, action, reward, state_, done)
        
        loss, td, q_vals = agent.learn()
        if len(q_vals) > 0:
            q_val_track.append(q_vals)
        if done:
           #agent.mem.discount_rewards(steps, agent.gamma)
           q_val_track = np.array(q_val_track)
           q_means = np.mean(q_val_track, axis=0)
           for i in range(len(q_val_dict)):
               q_val_dict[f'{i}'].append(q_means[i])
                    
        state = state_
        score += reward
        steps += 1
        
        agent.epsilon -= agent.epsilon_decay       
        agent.epsilon = max(agent.epsilon, agent.epsilon_min)
        #time.sleep(.01) 
   
    score_history.append(score)  
    avg_score = np.mean(score_history[-50:])
    high_score = max(high_score, avg_score)

        
    if not episode % stats_update:
        if td == []:
            td = 0
        if loss == []:
            loss = 0
        stats_dict['episode'].append(episode)
        stats_dict['moving average'].append(avg_score)
        stats_dict['min'].append(np.min(score_history[-stats_update:]))
        stats_dict['max'].append(np.max(score_history[-stats_update:]))
        stats_dict['average'].append(np.mean(score_history[-stats_update:]))
        stats_dict['epsilon'].append(agent.epsilon)
        stats_dict['learn_rate'].append(agent.lr)
        stats_dict['model_loss'].append(loss)
        stats_dict['temporal_difference'].append(td)   
        
        with open('./stats/training_hist.pkl', 'wb') as f:
            pickle.dump(stats_dict, f)
        with open('./stats/q_val.pkl', 'wb') as f:
            pickle.dump(q_val_dict, f)
        print(f"Episode:{episode}  Avg Reward:{np.round(avg_score,2)},  Max:{np.round(high_score,2)}, Epsilon:{np.round(agent.epsilon,3)},  LearnRate:{agent.lr},  Replay_size: {min(agent.mem.mem_size, agent.mem.mem_counter)}")
    if episode > 500 and high_score > save_score:
        save_score = high_score
        torch.save(agent.model, f'dot_model_{high_score}.pth')
    episode += 1



    
    
    
    