import numpy as np
import librosa
from PIL import Image

SR = 22050
CLIP_DUR = 3
N_FFT = 512
HOP_LENGTH = 512
N_MELS = 128
IMG_SIZE = 128


def preprocess_audio(audio_file):
    y, _ = librosa.load(audio_file, sr=SR, mono=True)

    target_len = SR * CLIP_DUR

    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        center=False
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    img = Image.fromarray(mel_db).resize(
        (IMG_SIZE, IMG_SIZE),
        Image.BILINEAR
    )

    mel_img = np.array(img, dtype=np.float32)

    min_val = mel_img.min()
    max_val = mel_img.max()

    if max_val - min_val > 1e-6:
        mel_img = (mel_img - min_val) / (max_val - min_val)
    else:
        mel_img = np.zeros_like(mel_img)

    mel_img = mel_img[..., np.newaxis]

    return mel_img