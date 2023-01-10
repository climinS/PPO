import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import argparse
from collections import namedtuple
from vmas import make_env

class CPPO():
    def __init__(self):
        super(CPPO, self).__init__()
        self.policy_net = Actor(num_state, 64, 64, num_action).float()
        self.critic_net = Critic(num_state, 64, 64).float()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(num_action)))
        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 4e-3)
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity = 1000
    batch_size = 2000
    def select_action(self, state):
        action_all = []
        action_log_prob_all = []
        for i in range(2):
            state_n = state[i]
            action_mean = self.policy_net.layer(state_n)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample()
            action = action.clamp(-1, 1)
            f_prob = probs.log_prob(action).sum(1), probs.entropy().sum(1)
            action_log_prob_all.append(f_prob)
            action_all.append(action)
        return action_all, action_log_prob_all
    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()
    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0
    def update(self):
        self.training_step += 1
        state = torch.tensor([t.state.detach().numpy() for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action.detach().numpy() for t in self.buffer], dtype=torch.float).view(-1,1)
        reward = torch.tensor([t.reward.detach().numpy() for t in self.buffer], dtype=torch.float).view(-1,1)
        next_state = torch.tensor([t.next_state.detach().numpy() for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1,1)
        reward = (reward - reward.mean()) / (reward.std() + 1e-10)
        with torch.no_grad():
            target_v = reward + args.gamma * self.critic_net.layer(next_state)
        advantage = (target_v - self.critic_net.layer(state)).detach()
        for _ in range(self.ppo_epoch):  # iteration ppo_epoch
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size,True):
                action_mean = self.policy_net.layer(state[index])
                action_logstd = self.actor_logstd.expand_as(action_mean)
                action_std = torch.exp(action_logstd)
                probs = Normal(action_mean, action_std)
                action_log_prob = probs.log_prob(action[index]).sum(1), probs.entropy().sum(1)
                action_log_prob = torch.tensor([item.cpu().detach().numpy() for item in action_log_prob],dtype=torch.float).view(-1, 1)
                ratio = torch.exp(action_log_prob - old_action_log_prob)
                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[index]
                action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
        del self.buffer[:]

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])
CPPO_reward=[]

class Actor(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, num_outputs):
        super(Actor, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, n_hidden_1),nn.ReLU(True),nn.Linear(n_hidden_1, n_hidden_2),nn.ReLU(True),nn.Linear(n_hidden_2, num_outputs))

class Critic(nn.Module):
    def __init__(self, in_dim, n_hidden_1,n_hidden_2):
        super(Critic, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, n_hidden_1),nn.ReLU(True),nn.Linear(n_hidden_1, n_hidden_2),nn.ReLU(True),nn.Linear(n_hidden_2, 1))

def run(scenario_name: str = "transport",n_steps: int = 200,n_envs: int = 32,env_kwargs: dict = {},device: str = "cpu",):
    env = make_env(scenario_name=scenario_name,num_envs=n_envs,device=device,continuous_actions=True,wrapper=None,**env_kwargs,)
    global num_state
    global num_action
    num_state = env.observation_space[0].shape[0]
    num_action = env.action_space[0].shape[0]
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    agent_num = 2
    agent=CPPO()
    running_reward = -1000
    for i_epoch in range(300):
        score = 0
        state = env.reset()
        for t in range(n_steps):
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            for j in range(agent_num):
                trans = Transition(state[j], action[j], reward[j], action_log_prob[j], next_state[j])
                if agent.store_transition(trans):
                    agent.update()
            score += sum(reward) / len(reward)
            state = next_state
        running_reward = running_reward * 0.9 + score * 0.1
        CPPO_reward.append(running_reward.item())

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=True, help='render the environment')
parser.add_argument('--log-interval',type=int,default=10,metavar='N',help='interval between training status logs (default: 10)')
args = parser.parse_args()
run(scenario_name="transport",heuristic=1,n_envs=1,n_steps=200,)
    