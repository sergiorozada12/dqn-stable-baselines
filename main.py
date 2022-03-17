from src.environment import get_env
from src.model import get_model
from src.buffer import get_buffer
from src.trainer import DQNTrainer
from src.config import NAME, POLICY, ALPHA

import torch


if __name__ == '__main__':
    env = get_env(NAME)
    model = get_model(env=env, hidden_size=100,)
    buffer = get_buffer(1_000_000, 0)

    trainer = DQNTrainer(
        env=env,
        model=model,
        episodes=10,
        max_steps=100,
        epsilon=1.0,
        alpha=.01,
        gamma=.99,
        buffer=buffer,
        batch_size=32,
        decay=.99999,
    )

    trainer.train(run_greedy_frequency=1)
