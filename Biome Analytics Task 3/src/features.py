"""Audio loading, chunking, and MFCC feature extraction utilities."""

import librosa
import numpy as np


def extract_mfcc(file_path, n_mfcc=40, n_fft=400, hop_length=160, max_len=98):
    """Load one audio file and convert it into a fixed-size MFCC array.

    Parameters
    ----------
    file_path : str
        Path to the audio file on disk.
    n_mfcc : int
        Number of MFCC coefficients to compute.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length between successive frames.
    max_len : int
        Number of time frames to pad/truncate to, for a fixed-size output.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_mfcc, max_len)``.
    """
    waveform, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc


def chunk_audio_file(file_path, chunk_duration=3.0, sr_target=16000):
    """Load one audio file and split it into fixed-duration chunks.

    Parameters
    ----------
    file_path : str
        Path to the audio file on disk.
    chunk_duration : float
        Duration in seconds of each chunk.
    sr_target : int
        Sample rate to resample to before chunking.

    Returns
    -------
    list[np.ndarray]
        List of waveform chunks, each of length
        ``int(chunk_duration * sr_target)``.
    """
    waveform, sr = librosa.load(file_path, sr=sr_target)
    chunk_len = int(chunk_duration * sr_target)
    n_chunks = len(waveform) // chunk_len

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_len
        end = start + chunk_len
        chunks.append(waveform[start:end])
    return chunks


def get_chunk_waveform_efficient(
    file_path, chunk_idx, chunk_duration=3.0, sr_target=16000
):
    """Read only the requested chunk of audio directly from disk.

    Avoids loading the entire file into memory when only one chunk is
    needed, which is significantly faster for large datasets.
    """
    import soundfile as sf

    info = sf.info(file_path)
    native_sr = info.samplerate
    chunk_len_native = int(chunk_duration * native_sr)
    start = chunk_idx * chunk_len_native

    waveform, sr = sf.read(
        file_path, start=start, frames=chunk_len_native, dtype="float32"
    )

    if sr != sr_target:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sr_target)

    return waveform
