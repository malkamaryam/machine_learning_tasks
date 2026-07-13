"""Unit tests for src.models."""

import torch

from src.models import VoiceEmbeddingNet


def test_voice_embedding_net_output_shape():
    model = VoiceEmbeddingNet(embedding_dim=128)
    dummy_input = torch.randn(4, 1, 40, 98)  # batch=4, matches MFCC shape

    output = model(dummy_input)

    assert output.shape == (4, 128)


def test_voice_embedding_net_output_is_normalized():
    model = VoiceEmbeddingNet(embedding_dim=64)
    dummy_input = torch.randn(2, 1, 40, 98)

    output = model(dummy_input)
    norms = torch.norm(output, p=2, dim=1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
