from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.type_aliases import Schedule


from torch.nn.modules.activation import ReLU
from torch.optim.adam import Adam

def get_model(
    policy,
    env,
    alpha,
    buffer_size,
    batch_size,
    gamma,
    epsilon,
    net_arch
    ):

    def lr_schedule(_):
        return alpha

    model = DQNPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lr_schedule,
        net_arch=net_arch,
        activation_fn=ReLU,
        optimizer_class=Adam
    )

    print(model.forward(env.reset()))

    print(model)

    return DQN(
        policy=policy,
        env=env,
        learning_rate=alpha,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_initial_eps=epsilon,
        verbose=1
    )