from src.environment import get_env
from src.model import get_model
from src.config import NAME, POLICY, ALPHA

if __name__ == '__main__':
    env = get_env(NAME)
    model = get_model(
        policy=POLICY,
        env=env,
        alpha=ALPHA,
        buffer_size=100,
        batch_size=32,
        gamma=0.99,
        epsilon=1.0,
        net_arch=[10,10]
    )
    #print(model)