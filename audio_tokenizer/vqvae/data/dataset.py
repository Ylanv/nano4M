import torchaudio
from torch.utils.data import Dataset
from typing import Union
from pathlib import Path
import torch
import random


class LibriSpeechMelDataset(Dataset):
    """Dataset of log mel spectogram from Librispeech"""

    def __init__(
        self,
        root: Union[str, Path],
        url: str = "dev-clean",
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
        self.segment_duration = segment_duration
        # Load Librispeech from root or download it
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=url, folder_in_archive="LibriSpeech", download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return log mel spectogram of a raw waveform from Librispeech

        Parameters:
            idx (int): Index of raw audio in LibriSpeech

        Return:
            waveform (Tensor) : Waveform
            sr (int) : sampling rate of data point
            txt (str) : Transcribe of data point
        """
        # Get data sample
        waveform, sr, txt, _, _, _ = self.dataset[idx]

        # Convert to mono
        waveform = waveform.mean(dim=0, keepdim=True)

        n_frame = int(sr * self.segment_duration)
        # Pad clip if too short, crop if too long.
        frame_wf = waveform.shape[1]
        if frame_wf < n_frame:
            pad_size = n_frame - frame_wf
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        else:
            # Generate rdm number to choose start of sequence
            start = random.randint(0, frame_wf - n_frame)
            waveform = waveform[:, start : start + n_frame]
        return waveform, sr, txt
