import gymnasium as gym

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical

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
        action = dist.sample()
        return action, dist.log_prob(action)


env = gym.make("CartPole-v1")
state, info = env.reset()
done = False
truncated = False
    
policy = Policy(env, hidden_dim=256)
opt = optim.Adam(policy.parameters(), lr=0.004)



total_steps = 100000
batch_size = 1000
num_iterations = total_steps // batch_size

num_update_epochs = 1

gamma = 0.99



for i in range(num_iterations):

    log_probs = []
    eps_rewards = []

    returns = []
    for j in range(batch_size):
        if done or truncated:
            state, info = env.reset()


            eps_returns = []
            G = 0
            for r in reversed(eps_rewards):
                G = r + gamma * G
                eps_returns.insert(0, G)
            returns += eps_returns

            print(f"Episodic rewards: {sum(eps_rewards)}")
            eps_rewards = []

            done = False
            truncated = False


        action, log_prob = policy.get_action(torch.tensor(state, dtype=torch.float32))

        state, reward, done, truncated, info = env.step(action.item())

        eps_rewards.append(reward)
        log_probs.append(log_prob)

    if eps_rewards:
        state, info = env.reset()

        eps_returns = []
        G = 0
        for r in reversed(eps_rewards):
            G = r + gamma * G
            eps_returns.insert(0, G)
        returns += eps_returns

        # print(f"Episodic rewards: {sum(eps_rewards)}")
        eps_rewards = []

        done = False
        truncated = False


    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs = torch.stack(log_probs)

    opt.zero_grad()
    loss = -(log_probs * returns).mean()
    loss.backward()
    opt.step()

env.close()