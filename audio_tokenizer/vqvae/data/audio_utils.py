import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F_audio
import torch.nn.functional as F
import random
import librosa
from pathlib import Path
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
    if mel_features.ndim == 3:
        mel_features = mel_features.squeeze(0)
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

def save_waveform_batch(
    originals,
    reconstructions,
    sample_rate,
    batch_index,
    batch_size,
    output_dir="audio_tokenizer/vqvae/data/inference",
):
    """
    Saves original and reconstructed waveforms for each item in the batch.

    Args:
        originals (Tensor): shape (batch_size, num_channels, num_samples)
        reconstructions (Tensor): shape (batch_size, num_channels, num_samples)
        sample_rate (int): sampling rate (e.g., 16000)
        batch_index (int): index of the batch
        output_dir (str or Path): root folder for output
    """
    batch_folder = Path(output_dir) / f"batch_{batch_index}"
    batch_folder.mkdir(parents=True, exist_ok=True)

    # Detach tensor and put on CPU
    orig = originals.detach().cpu()
    recon = reconstructions.detach().cpu()

    for i in range(batch_size):
        # Extract wf from batch
        recon_wf_i = recon[i]
        orig_wf_i = orig[i]
        # Create path
        orig_path = batch_folder / f"original_{i}.wav"
        recon_path = batch_folder / f"reconstruction_{i}.wav"
        orig_wf_path = batch_folder / f"original_{i}.png"
        recon_wf_path = batch_folder / f"reconstruction_{i}.png"
        # Save
        torchaudio.save(
            orig_path.as_posix(),
            orig_wf_i,
            sample_rate,
            encoding="PCM_F",
            bits_per_sample=32,
        )
        torchaudio.save(
            recon_path.as_posix(),
            recon_wf_i,
            sample_rate,
            encoding="PCM_F",
            bits_per_sample=32,
        )
        visualize_waveform(orig_wf_i, sample_rate, orig_wf_path.as_posix())
        visualize_waveform(recon_wf_i, sample_rate, recon_wf_path.as_posix())
        print(f"Saved item {i} of batch {batch_index} at {orig_path}")
        
def stft_loss(predicted_audio, target_audio, n_fft=1024, hop_length=256):
    predicted_audio = predicted_audio.squeeze(1)  # Remove the singleton dimension
    target_audio = target_audio.squeeze(1)  # Remove the singleton dimension
    # Compute the STFT (magnitude spectrogram)
    stft_pred = torch.stft(predicted_audio, n_fft=n_fft, hop_length=hop_length,return_complex=True)
    stft_target = torch.stft(target_audio, n_fft=n_fft, hop_length=hop_length,return_complex=True)

    # Magnitude of the complex spectrogram
    mag_pred = torch.abs(stft_pred)
    mag_target = torch.abs(stft_target)

    # Compute loss as the L1 distance between magnitudes
    return F.l1_loss(mag_pred, mag_target)