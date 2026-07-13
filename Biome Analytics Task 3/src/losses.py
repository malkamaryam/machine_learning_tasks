"""Triplet loss used to train the speaker embedding models."""

import torch.nn.functional as F


def triplet_loss(anchor, positive, negative, margin=0.2):
    """Compute the triplet loss for a batch of embeddings.

    Parameters
    ----------
    anchor, positive, negative : torch.Tensor
        Batches of embeddings of shape ``(batch, embedding_dim)``.
    margin : float
        Minimum desired gap between the positive and negative distances.

    Returns
    -------
    torch.Tensor
        Scalar mean triplet loss over the batch.
    """
    dist_pos = F.pairwise_distance(anchor, positive, p=2)
    dist_neg = F.pairwise_distance(anchor, negative, p=2)
    losses = F.relu(dist_pos - dist_neg + margin)
    return losses.mean()
