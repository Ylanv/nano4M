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
        segment_duration: float = 3.0,
        stride_duration: float = 1.0,
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
        self.stride_duration = stride_duration
        # Load Librispeech from root or download it
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=url, folder_in_archive="LibriSpeech", download=True
        )

        self.segment = []
        # Calcule n
        sr = 0
        for idx in range(len(self.dataset)):
            wf, sr, *_ = self.dataset[idx]
            num_frames = wf.shape[1]
            seg_len = int(sr * segment_duration)
            stride = int(sr * stride_duration)

            # If waveform is shorter than a single segment skip it
            if num_frames < seg_len:
                continue

            # Append the start of each sample
            for start in range(0, num_frames - seg_len + 1, stride):
                self.segment.append((idx, start))

        self.segment_length = int(sr * segment_duration)

    def __len__(self):
        # Return the number of segment
        return len(self.segment)

    def __getitem__(self, idx):
        """Return log mel spectogram of a raw waveform from Librispeech

        Parameters:
            idx (int): Index of raw audio in LibriSpeech

        Return:
            waveform (Tensor) : Waveform
            sr (int) : sampling rate of data point
            txt (str) : Transcribe of data point
        """
        index, start = self.segment[idx]

        # Get data sample
        waveform, sr, txt, _, _, _ = self.dataset[index]

        # Convert to mono
        waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform[:, start : start + self.segment_length]

        return waveform, sr, txt
