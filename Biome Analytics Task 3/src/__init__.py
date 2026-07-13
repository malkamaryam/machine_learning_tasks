"""Speaker verification/recognition utilities for Biome Analytics Task 3.

This package contains reusable building blocks extracted from the
experiment notebooks in ``notebooks/``:

- ``features``: audio loading, chunking, and MFCC feature extraction.
- ``models``: the CNN-based ``VoiceEmbeddingNet`` architecture.
- ``losses``: the triplet loss used to train the embedding models.
- ``ecapa_utils``: helpers for extracting embeddings with the
  pretrained ECAPA-TDNN model (speechbrain).
- ``evaluation``: FAR/FRR/EER computation for verification systems.
"""
