from dataclasses import dataclass

@dataclass
class PPOConfig:
    env_id: str
    num_envs: int = 4
    hidden_dim: int = 64
    lr: float = 0.002
    gamma: float = 0.99
    gae_lambda: float = 0.95
    total_timesteps: int = 200_000
    num_steps: int = 128
    num_minibatches: int = 4
    num_update_epochs: int = 4
    clip_coeff: float = 0.2
    entropy_coeff: float = 0.00
    vf_coeff: float = 0.5
    max_grad_norm: float = 0.5
    capture_video: bool = True
    run_name: str = "ppo"

CONFIGS = {
    "CartPole-v1": PPOConfig(
        env_id="CartPole-v1",
        run_name="ppo_cartpole",
    ),
    "Pendulum-v1": PPOConfig(
        env_id="Pendulum-v1",
        num_steps=256,
        lr=0.0003,
        entropy_coeff=0.02,
        clip_coeff=0.1,
        run_name="ppo_pendulum",
    ),
    "HalfCheetah-v5": PPOConfig(
        env_id="HalfCheetah-v5",
        hidden_dim=128,
        lr=0.0003,
        total_timesteps=1_000_000,
        num_steps=2048,
        num_minibatches=32,
        num_update_epochs=10,
        clip_coeff=0.2,
        entropy_coeff=0.00,
        run_name="ppo_halfcheetah",
    )
}
