import numpy as np
import scipy
import torch
import rclpy
import os
import copy

from torch import nn
from torch.nn import functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, max_action):
        super().__init__()
        self.fa1 = nn.Linear(state_size, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        self.fa3 = nn.Linear(hidden_size, action_size)

        self.max_action = max_action

    def forward(self, states, visualize=False):
        x1 = torch.relu(self.fa1(states))
        x2 = torch.relu(self.fa2(x1))
        action = torch.tanh(self.fa3(x2)) * self.max_action

        return action

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.l1 = nn.Linear(state_size + action_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, states, actions):
        x = torch.cat((states, actions), dim=1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_size=512,
        device="cpu",
        gamma=0.99,
        tau=0.003,
        policy_noise=0.2,
        noise_clip=0.5,
        exploration_noise=0.1,
        policy_delay=2,
        batch_size=64,
        actor_lr=0.003,
        critic_lr=0.003,
    ):
        self.device = device
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_delay = policy_delay
        self.batch_size = batch_size

        self.actor = Actor(state_dim, action_dim, hidden_size, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic_1 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.total_it = 0
    
    def get_action(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]

        if training:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise

        action = np.clip(action, -self.max_action, self.max_action)
        return action
    
    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train(self, replay_buffer):
        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)

            target = reward + self.gamma * (1 - done) * target_q

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        critic_1_loss = F.mse_loss(current_q1, target)
        critic_2_loss = F.mse_loss(current_q2, target)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        actor_loss = None

        
        if self.total_it % self.policy_delay == 0:  
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic_1, self.critic_1_target)
            self.soft_update(self.critic_2, self.critic_2_target)

        return {
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else None
        }