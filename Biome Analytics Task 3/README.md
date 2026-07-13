# Biome Analytics — Task 3: Speaker Recognition/Verification

This task explores speaker recognition and verification using two approaches:

1. **CNN-based triplet-loss embeddings**, trained from scratch on:
   - A 4-speaker dataset (`notebooks/CNN-4_Speakers.ipynb`)
   - A 50-speaker dataset (`notebooks/CNN-50_Speakers.ipynb`)
2. **ECAPA-TDNN**, a pretrained speaker-embedding model from SpeechBrain,
   evaluated (and lightly fine-tuned) on the 50-speaker dataset
   (`notebooks/ECAPA_TDNN.ipynb`).

## Structure

```
Biome Analytics Task 3/
├── notebooks/          # Original experiment notebooks
│   ├── CNN-4_Speakers.ipynb
│   ├── CNN-50_Speakers.ipynb
│   └── ECAPA_TDNN.ipynb
├── src/                # Reusable modules extracted from the notebooks
│   ├── features.py     # MFCC extraction, audio chunking
│   ├── models.py        # VoiceEmbeddingNet (CNN) architecture
│   ├── losses.py         # Triplet loss
│   ├── ecapa_utils.py     # ECAPA-TDNN embedding helper
│   └── evaluation.py       # FAR / FRR / EER metrics
├── tests/               # Unit tests for src/ (pytest)
├── docs/                # Additional notes/results write-ups
└── requirements.txt
```

## Approach summary

- Audio is converted to MFCC features (for the CNN models) or fed directly
  to the pretrained ECAPA-TDNN encoder.
- Models are evaluated using genuine vs. impostor verification trials,
  reporting False Acceptance Rate (FAR), False Rejection Rate (FRR), and
  the Equal Error Rate (EER) threshold.
- The 50-speaker CNN model processes audio in fixed 3-second chunks to
  handle longer recordings efficiently.

## Setup

```bash
pip install -r requirements.txt
```

## Running tests

```bash
pytest tests/
```

## Linting

This task follows the repo-wide flake8 configuration (max line length 120).

```bash
flake8 src/ tests/
```

## Notes

- Dataset download previously relied on a Kaggle API token that was
  hardcoded in the notebooks. This has been removed — set your own token
  as the `KAGGLE_API_TOKEN` environment variable (or via
  `~/.kaggle/access_token`) before re-running the download cells.
