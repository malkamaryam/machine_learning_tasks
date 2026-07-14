import torch
import torchaudio
import numpy as np
import pickle
import os
import soundfile as sf
from speechbrain.inference.speaker import EncoderClassifier
from pydub import AudioSegment

AudioSegment.converter = r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ecapa_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_ecapa"
)

VOICEPRINTS_PATH = "voiceprints.pkl"
THRESHOLD = 0.60


def load_voiceprints():
    if os.path.exists(VOICEPRINTS_PATH):
        with open(VOICEPRINTS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_voiceprints(voiceprints):
    with open(VOICEPRINTS_PATH, "wb") as f:
        pickle.dump(voiceprints, f)


def get_embedding(file_path, target_sr=16000):
    """
    Loads an audio file (any common format/container, regardless of its
    extension), converts it to a proper mono WAV via ffmpeg, resamples to
    16kHz if needed, and returns its ECAPA-TDNN embedding as a normalized
    NumPy array. Raises a ValueError with a clear message if the file
    can't be processed.
    """
    # Step 1: Normalize the file into a real WAV using ffmpeg/pydub,
    # regardless of what its extension claims to be. This fixes files
    # that are actually Ogg/WebM/etc. but saved with a .wav extension
    # (common with browser-based voice recorders).
    converted_path = file_path + "_converted.wav"
    try:
        audio = AudioSegment.from_file(file_path)  # ffmpeg auto-detects real format
        audio = audio.set_channels(1)  # force mono
        audio.export(converted_path, format="wav")
    except Exception as e:
        raise ValueError(
            f"Could not read audio file. Please upload a valid .wav or .mp3 file. ({e})"
        )

    # Step 2: Read the guaranteed-clean WAV with soundfile
    try:
        waveform, sr = sf.read(converted_path, dtype="float32")
    except Exception as e:
        raise ValueError(f"Could not process converted audio file. ({e})")
    finally:
        if os.path.exists(converted_path):
            os.remove(converted_path)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if len(waveform) < sr * 0.5:  # less than half a second of audio
        raise ValueError(
            "Audio sample is too short. Please upload at least 1-2 seconds of speech."
        )

    signal = torch.from_numpy(waveform).unsqueeze(0)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        signal = resampler(signal)

    with torch.no_grad():
        embedding = ecapa_model.encode_batch(signal).squeeze().cpu().numpy()

    return embedding / np.linalg.norm(embedding)


def enroll_user(username, file_paths):
    embeddings = [get_embedding(path) for path in file_paths]
    voiceprint = np.mean(embeddings, axis=0)
    voiceprint = voiceprint / np.linalg.norm(voiceprint)
    voiceprints = load_voiceprints()
    voiceprints[username] = voiceprint
    save_voiceprints(voiceprints)
    return True


def authenticate_user(username, file_path):
    voiceprints = load_voiceprints()
    if username not in voiceprints:
        return {"error": f"'{username}' is not registered."}
    embedding = get_embedding(file_path)
    stored_voiceprint = voiceprints[username]
    similarity = float(np.dot(embedding, stored_voiceprint))
    decision = similarity >= THRESHOLD
    return {
        "username": username,
        "authenticated": decision,
        "confidence": round(similarity, 4),
        "threshold": THRESHOLD,
    }


def get_registered_users():
    voiceprints = load_voiceprints()
    return list(voiceprints.keys())
