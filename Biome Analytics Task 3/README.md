# Biome Analytics — Task 3: Speaker Recognition/Verification

This task explores speaker recognition and verification using two approaches:

1. **CNN-based triplet-loss embeddings**, trained from scratch on:
   - A 4-speaker dataset (`notebooks/CNN-4_Speakers.ipynb`)
   - A 50-speaker dataset (`notebooks/CNN-50_Speakers.ipynb`)
2. **ECAPA-TDNN**, a pretrained speaker-embedding model from SpeechBrain,
   evaluated (and lightly fine-tuned) on the 50-speaker dataset
   (`notebooks/ECAPA_TDNN.ipynb`).

## Structure
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
├── Voice Biometric Flask/  # Flask web app (see below)
└── requirements.txt

## Approach summary
- Audio is converted to MFCC features (for the CNN models) or fed directly
  to the pretrained ECAPA-TDNN encoder.
- Models are evaluated using genuine vs. impostor verification trials,
  reporting False Acceptance Rate (FAR), False Rejection Rate (FRR), and
  the Equal Error Rate (EER) threshold.
- The 50-speaker CNN model processes audio in fixed 3-second chunks to
  handle longer recordings efficiently.
- **Final model decision:** the zero-shot pretrained ECAPA-TDNN
  (no fine-tuning) was selected for production use, achieving ~2.3% EER —
  substantially better than the custom CNN (~6.8–8% EER) and with no
  meaningful gain from lightweight fine-tuning (~2.2% EER, within
  run-to-run noise).

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

---

## Flask Web Application (`Voice Biometric Flask/`)

A simple Flask-based web interface demonstrating the voice biometric
system above in an interactive, browser-based form. Built as a follow-up
task to integrate the final ECAPA-TDNN model into a usable application.

### Structure
Voice Biometric Flask/
├── app.py                  # Flask routes: home, register, authenticate
├── utils/
│   └── voice_auth.py       # Embedding extraction, enrollment, authentication logic
├── templates/               # HTML pages (home, register, authenticate)
├── static/
│   └── style.css            # App styling
├── test_data/                # Sample voice recordings for manual testing
├── uploads/                   # Uploaded audio saved at runtime (gitignored)
└── voiceprints.pkl              # Stored voice embeddings (gitignored, generated at runtime)

### Features
- **Register**: enter a username, upload one or more voice samples, and
  generate a stored voiceprint (averaged, L2-normalized ECAPA-TDNN embedding).
- **Authenticate**: select a registered username, upload a new voice
  sample, and get an Authenticated/Not Authenticated result along with
  the cosine similarity confidence score and threshold used.
- Handles invalid/unsupported audio formats gracefully, including
  automatic transcoding of mislabeled files (e.g. browser-recorded
  Ogg/WebM audio saved with a `.wav` extension) via `pydub`/`ffmpeg`.

### Setup
```bash
cd "Voice Biometric Flask"
pip install -r requirements.txt
```
`ffmpeg` must also be installed and available (either on your system PATH,
or its path set directly in `utils/voice_auth.py` via `AudioSegment.converter`).

### Running the app
```bash
python app.py
```
Then visit `http://127.0.0.1:5000` in your browser.

### Test data
Sample voice recordings for manual testing are provided in `test_data/`,
including genuine (same-speaker) and impostor (different-speaker) samples
<<<<<<< HEAD
used during development to validate registration and authentication behavior.
=======
used during development to validate registration and authentication behavior.
>>>>>>> ff9fa85c9a6926b86ce2ed68cef5b269af06266a
