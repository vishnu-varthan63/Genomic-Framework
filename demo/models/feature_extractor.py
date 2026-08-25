import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * input_dim, hidden_dim)
        self.activation = nn.ReLU()
    def forward(self, x):
        # x: (batch, embed_dim)
        x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        x = self.conv(x)    # (batch, 16, embed_dim)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc(x))
        return x  # (batch, hidden_dim)
