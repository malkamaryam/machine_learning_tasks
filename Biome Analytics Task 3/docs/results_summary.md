# Results Summary — Task 3

## CNN (4 speakers)
Triplet-loss CNN trained from scratch on MFCC features from a 4-speaker
dataset. See `notebooks/CNN-4_Speakers.ipynb` for training curves and the
FAR/FRR/EER sweep.

## CNN (50 speakers)
Same architecture, extended to a 50-speaker dataset using chunked
(3-second) audio segments to handle longer recordings efficiently. See
`notebooks/CNN-50_Speakers.ipynb`, including a learning-rate comparison
sweep.

## ECAPA-TDNN (50 speakers)
Pretrained `speechbrain/spkrec-ecapa-voxceleb` encoder evaluated
zero-shot, then lightly fine-tuned on the same 50-speaker dataset. See
`notebooks/ECAPA_TDNN.ipynb` for the full comparison against the CNN
baseline, including live-microphone enrollment/verification cells.

