import torch


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output


def get_model(env, hidden_size):
    input_size = env.observation_space.high.shape[0]
    output_size = env.action_space.n
    return MLP(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )
