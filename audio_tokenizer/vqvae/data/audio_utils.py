import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F_audio

class MelToAudio(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.mel_to_spec = T.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate
        )

        self.griffin_lim = T.GriffinLim(n_fft=n_fft, hop_length=hop_length,length=48000)

    def forward(self, log_mel):

        log_mel = log_mel 
        power_mel = F_audio.DB_to_amplitude(log_mel, ref=1.0, power=1.0)  # back to power scale
        spec = self.mel_to_spec(power_mel)
        waveform = self.griffin_lim(spec)
        return waveform  # [B, waveform_len]
