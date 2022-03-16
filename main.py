from src.environment import get_env
from src.config import NAME

if __name__ == '__main__':
    env = get_env(NAME)
    print(env)