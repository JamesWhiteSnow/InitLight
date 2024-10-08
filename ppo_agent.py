import numpy as np
import torch.nn as nn
import os
import torch 
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    def sample(self):
        batch_step = np.arange(0, len(self.states), self.batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_step]
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def push(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class Actor(nn.Module):
    def __init__(self,state_dim, action_dim,
            hidden_dim):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
        )
    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim,hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, 1)
        )
    def forward(self, state):
        value = self.critic(state)
        return value

class PPOAgent:
    def __init__(self,intersection_id, state_dim, action_dim,cfg,phase_list):
        self.intersection_id = intersection_id
        self.phase_list = phase_list
        self.gamma = cfg.gamma
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device
        self.actor = Actor(state_dim, action_dim,cfg.hidden_dim).to(self.device)
        self.critic = Critic(state_dim,cfg.hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PPOMemory(cfg.batch_size)
        self.loss = 0

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, probs, value

    def remember(self,state, action, prob, val, reward, done):
        self.memory.push(state, action, prob, val, reward, done)

    def update(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.sample()
            values = vals_arr

            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)

            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss
                self.loss  = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()  
    def save(self,path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(path, 'ppo_critic.pt')
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)
    def load(self,path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(path, 'ppo_critic.pt')
        self.actor.load_state_dict(torch.load(actor_checkpoint)) 
        self.critic.load_state_dict(torch.load(critic_checkpoint))  

class MPPOAgent(object):
    def __init__(self,
                 intersection,
                 state_size,
                 cfg,
                 phase_list
                 ):

        self.intersection = intersection
        self.agents = {}
        self.make_agents(state_size, cfg, phase_list)

    def make_agents(self, state_size, cfg, phase_list):
        for id_ in self.intersection:
            self.agents[id_] = PPOAgent(intersection_id=id_,
                                        state_dim=state_size,
                                        action_dim=len(phase_list[id_]),
                                        cfg=cfg,
                                        phase_list=phase_list[id_],
                                        )

    def remember(self, state, action, prob, val, reward, done):
        for id_ in self.intersection:
            self.agents[id_].remember(state[id_],
                                      action[id_],
                                      prob[id_],
                                      val[id_],
                                      reward[id_],
                                      done[id_]
                                      )

    def choose_action(self, state):
        action = {}
        prob={}
        val={}
        for id_ in self.intersection:
            action[id_],prob[id_], val[id_] = self.agents[id_].choose_action(state[id_])
        return action,prob,val

    def replay(self):
        for id_ in self.intersection:
            self.agents[id_].update()

    def load(self, name):
        for id_ in self.intersection:
            self.agents[id_].load(name)
        print("\nloading model successfully!\n")

    def save(self,path):
        for id_ in self.intersection:
            self.agents[id_].save(path + "/")
