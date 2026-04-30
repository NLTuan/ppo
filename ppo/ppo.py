import gymnasium as gym

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


env = gym.make("CartPole-v1", render_mode="human")

class Policy(nn.Module):
    def __init__(self, env, hidden_dim):
        super().__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, env.action_space.n)
        )

    def forward(self, x):
        return self.policy_network(x)

    def get_action(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.sample(), dist.log_prob(dist.sample()) 


policy = Policy(env, hidden_dim=128)


for _ in range(10):
    state, info = env.reset()
    done = False
    truncated = False
    
    while not done and not truncated:
        action = policy.get_action(torch.tensor(state))[0].item()
        state, reward, done, truncated, info = env.step(action)
        env.render()

env.close()