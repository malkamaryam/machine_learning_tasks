"""CNN architecture used to learn fixed-size speaker embeddings from MFCCs."""

import torch.nn as nn
import torch.nn.functional as F


class VoiceEmbeddingNet(nn.Module):
    """CNN that maps an MFCC array to a fixed-size embedding vector.

    Input is expected to be shaped ``(batch, 1, n_mfcc, time_frames)``.
    """

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
