"""Helpers for extracting speaker embeddings with a pretrained ECAPA-TDNN
model from speechbrain.
"""

import torch
import torchaudio


def get_ecapa_embedding(file_path, ecapa_model, target_sr=16000):
    """Load an audio file and return its ECAPA-TDNN embedding.

    Parameters
    ----------
    file_path : str
        Path to the audio file on disk.
    ecapa_model : speechbrain.inference.speaker.EncoderClassifier
        A loaded ECAPA-TDNN encoder classifier instance.
    target_sr : int
        Sample rate the model expects (16 kHz for spkrec-ecapa-voxceleb).

    Returns
    -------
    np.ndarray
        Flat NumPy array containing the speaker embedding.
    """
    signal, sr = torchaudio.load(file_path)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        signal = resampler(signal)

    with torch.no_grad():
        embedding = ecapa_model.encode_batch(signal)

    return embedding.squeeze().cpu().numpy()
