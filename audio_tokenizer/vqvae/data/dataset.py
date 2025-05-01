import torch
import torchaudio
import torchaudio.transforms as at
from torch.utils.data import Dataset
from typing import Union
from pathlib import Path
import random


class LibriSpeechMelDataset(Dataset):
    """Dataset of log mel spectogram from Librispeech"""

    def __init__(
        self,
        root: Union[str, Path],
        url: str = "dev-clean",
        sr: int = 44100,
        n_mels: int = 80,
        segment_duration: float = 2.0,
    ):
        """Init dataset class

        Parameters:
            root ([Str,Path]) : Path of dataset if local
            url (str) : Subset of the dataset
            sr (int) : sample rate
            n_mels (int) : Number of mel filterbanks
            segment_duration (float) : Duration of each training sample
        """
        # Load Librispeech from root or download it
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=url, folder_in_archive="LibriSpeech", download=True
        )
        self.sr = sr
        self.n_mels = n_mels

        # Number of sample per data point
        self.segment_samples = int(sr * segment_duration)

        # Create MelSpectogram from raw audio signal
        self.mel_transform = at.MelSpectrogram(
            sample_rate=sr,  # Sample rate of audio signal
            n_fft=1024,  # Size of FFt
            hop_length=256,  # Length of Hop between STFT windows
            n_mels=n_mels,  # Number of mel filterbanks
        )

        # Turn tensor from amplitude scale to decibel scale
        self.amplitude_to_db = at.AmplitudeToDB()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return log mel spectogram of a raw waveform from Librispeech

        Parameters:
            idx : Index of raw audio in LibriSpeech
        """
        # Get data sample
        waveform, sr, _, _, _, _ = self.dataset[idx]

        # Convert to mono
        waveform = waveform.mean(dim=0, keepdim=True)

        # Pad clip if too short, crop if too long
        sample_size = waveform.shape[1]
        if sample_size < self.segment_samples:
            pad_size = self.segment_samples - sample_size
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        else:
            # Generate rdm number to choose start of sequence
            start = random.randint(0, sample_size - self.segment_samples)
            waveform = waveform[:, start : start + self.segment_samples]

        # Get mel spectogram
        mel_spec = self.mel_transform(waveform)
        # Log spectogram
        mel_db = self.amplitude_to_db(mel_spec)
        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return mel_db.squeeze(0)
