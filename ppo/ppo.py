import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical

env_id            = "CartPole-v1"
num_envs          = 4
hidden_dim        = 256
lr                = 3e-4
gamma             = 0.99
total_timesteps   = 100_000
num_steps         = 128
num_update_epochs = 1
capture_video     = False
run_name          = "ppo_cartpole"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size     = num_steps * num_envs
num_iterations = total_timesteps // batch_size

class Agent(nn.Module):
    def __init__(self, envs, hidden_dim):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = envs.single_action_space.n

        self.policy_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )

        self.value_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.policy_network(x)

    def get_action(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_value(self, x):
        return self.value_network(x)

    def get_action_and_value(self, x, action=None):
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.get_value(x)

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

envs = gym.vector.SyncVectorEnv([make_env(env_id, i, capture_video, run_name) for i in range(num_envs)])
obs_dim = np.array(envs.single_observation_space.shape).prod()

state, info = envs.reset()
done = False
truncated = False

agent = Agent(envs, hidden_dim)
opt = optim.Adam(policy.parameters(), lr=lr)

obs_buf   = torch.zeros(num_steps, num_envs, obs_dim)
act_buf   = torch.zeros(num_steps, num_envs)
rew_buf   = torch.zeros(num_steps, num_envs)
done_buf  = torch.zeros(num_steps, num_envs)
logp_buf  = torch.zeros(num_steps, num_envs)
val_buf   = torch.zeros(num_steps, num_envs)

obs = torch.Tensor(state).to(device)
done = torch.zeros(num_envs).to(device)

global_step = 0

for i in range(1, num_iterations + 1):

    for step in range(num_steps):
        global_step += num_envs

        obs_buf[step] = obs
        done_buf[step] = done

        # sample action 
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs)
            val_buf[step] = value.flatten()

        actions[step] = action
        logp_buf[step] = logprob

        obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)
        rew_buf[step] = torch.tensor(reward).to(device).view(-1)
        obs, done = torch.Tensor(obs).to(device), torch.Tensor(done).to(device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"step={global_step}, episodic_return={info['episode']['r']}")
        



    if eps_rewards:
        state, info = envs.reset()

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

envs.close()