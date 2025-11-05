import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], 2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
