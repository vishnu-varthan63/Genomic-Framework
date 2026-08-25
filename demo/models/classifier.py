import torch
import torch.nn as nn

class Classifier(nn.Module):
    """Small MLP classifier for the demo.
    Configurable via a dictionary passed on init.
    Expected config keys:
        - embed_dim: dimension of fused representation (matches attention output)
        - hidden_dim: hidden layer size
        - num_classes: number of output classes
    """
    def __init__(self, embed_dim=64, hidden_dim=64, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        probs = self.softmax(logits)
        return probs
