"""Unit tests for src.losses."""

import torch

from src.losses import triplet_loss


def test_triplet_loss_is_zero_for_well_separated_embeddings():
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[-1.0, 0.0]])

    loss = triplet_loss(anchor, positive, negative, margin=0.2)

    assert loss.item() == 0.0


def test_triplet_loss_is_positive_when_negative_is_close():
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[0.99, 0.0]])

    loss = triplet_loss(anchor, positive, negative, margin=0.2)

    assert loss.item() > 0.0
