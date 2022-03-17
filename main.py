from src.environment import get_env
from src.model import get_model
from src.buffer import get_buffer
from src.trainer import DQNTrainer
from src.config import (
    BATCH_SIZE,
    NAME,
    ALPHA,
    HIDDEN_SIZE,
    BUFFER_SIZE,
    SEED,
    EPISODES,
    MAX_STEPS,
    EPSILON,
    GAMMA,
    DECAY,
)

if __name__ == '__main__':
    env = get_env(NAME)
    model = get_model(env=env, hidden_size=HIDDEN_SIZE,)
    buffer = get_buffer(BUFFER_SIZE, SEED)

    trainer = DQNTrainer(
        env=env,
        model=model,
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        epsilon=EPSILON,
        alpha=ALPHA,
        gamma=GAMMA,
        buffer=buffer,
        batch_size=BATCH_SIZE,
        decay=DECAY,
    )

    trainer.train(run_greedy_frequency=1)
    reward_final = trainer.greedy_cumulative_reward[-1]
    
    print(f'Final reward - {reward_final}')
