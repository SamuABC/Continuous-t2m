import torch.nn as nn

class MotionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_layers=3):
        super().__init__()
        layers = []
        dim_in = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.ReLU())
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, input_dim))  # output_dim = input_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, D) oder (B*T, D)
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
        out = self.net(x)
        if len(orig_shape) == 3:
            out = out.reshape(B, T, -1)
        return out