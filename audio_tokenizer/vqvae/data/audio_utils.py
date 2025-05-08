import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F_audio
import random
import librosa
import matplotlib.pyplot as plt
from typing import Optional


class Mel_to_wf(nn.Module):
    """Transform log mel spectogram to waveform, use Griffin Lim."""

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.mel_to_spec = T.InverseMelScale(
            n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate
        )
        self.griffin_lim = T.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            power=1.0,
            n_iter=32,
        )
        # self.griffin_lim = F_audio.GriffinLim(n_fft=n_fft, hop_length=hop_length,win_length=n_fft)

    def forward(self, log_mel):
        # back to Amplitude
        power_mel = F_audio.DB_to_amplitude(log_mel, ref=1.0, power=0.5)

        spec = self.mel_to_spec(power_mel)
        print(f"Spec shape : {spec.shape}")
        waveforms = []
        for i in range(spec.size(0)):
            spec_unbatch = spec[i].squeeze(0)
            wave = self.griffin_lim(spec_unbatch)
            waveforms.append(wave)

        return torch.stack(waveforms)


class Wf_to_mel(nn.Module):
    """Transform waveform to logmel spectogram"""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        segment_duration: float = 2.0,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        self.sr = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Number of sample per data point
        self.segment_samples = int(sample_rate * segment_duration)

        # Create MelSpectogram from raw audio signal
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,  # Sample rate of audio signal
            n_fft=n_fft,  # Size of FFt
            hop_length=hop_length,  # Length of Hop between STFT windows
            n_mels=n_mels,  # Number of mel filterbanks
        )

        # Turn tensor from amplitude scale to decibel scale
        self.amplitude_to_db = T.AmplitudeToDB()

    def forward(self, waveform):
        """Transform waveform to logmel spectogram

        Parameters:
            waveform (Tensor) : Tensor containing the waveform
        Return:
            mel_db (Tensor) : Mel spectogram
            mean (Tensor) : Mean
            std (Tensor)  : Std
        """

        # Get mel spectogram
        mel_spec = self.mel_transform(waveform)
        # Log spectogram
        mel_db = self.amplitude_to_db(mel_spec)
        # Normalize
        mean = mel_db.mean()
        std = mel_db.std()
        mel_db = (mel_db - mean) / (std + 1e-6)
        return mel_db.squeeze(1), mean, std


def visualize_waveform(y, sr, path: Optional[str] = None):
    """Take audio time series (np.ndarray) and plot waweform"""
    y = y.numpy()
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Audio Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def visualize_spectogram(mel_features, sr, path: Optional[str] = None):
    """Visualize Spectogram given mel features"""
    mel_features = mel_features.numpy()
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mel_features, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%2.0f dB")
    plt.title("Mel Spectrogram")

    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
