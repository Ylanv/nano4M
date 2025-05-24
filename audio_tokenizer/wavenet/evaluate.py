import os
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

from audio_tokenizer.vqvae.models.vqvae import RawVQVAE
from wavenet_vocoder.wavenet import WaveNet
from audio_tokenizer.vqvae.data.audio_utils import stft_loss
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset

# === Config ===
BATCH_SIZE = 1  # must be 1 for autoregressive inference
SEGMENT_DURATION = 2.0
SAMPLE_RATE = 16000
T_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)
SAVE_EXAMPLES = True  # save waveform output for inspection
N_EXAMPLES = 5        # how many samples to save
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Paths ===
DATASET_PATH = "/work/com-304/snoupy/librispeech/"
CHECKPOINT_VQVAE = "/work/com-304/snoupy/weights/vqvae/vqvae_epoch30.pt"
CHECKPOINT_WAVENET = "/work/com-304/snoupy/weights/wavenet/wavenet_epoch_3.pt"
OUTPUT_DIR = Path("audio_tokenizer/vqvae/data/inference/test_clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_models():
    vqvae = RawVQVAE().to(DEVICE)
    vqvae.load_state_dict(torch.load(CHECKPOINT_VQVAE, map_location=DEVICE))
    vqvae.eval()

    wavenet = WaveNet(
        out_channels=30,
        layers=20,
        stacks=2,
        residual_channels=64,
        gate_channels=128,
        skip_out_channels=64,
        kernel_size=3,
        dropout=0.05,
        cin_channels=128,
        scalar_input=True,
        upsample_conditional_features=True,
        upsample_scales=[4, 4, 4, 1]
    ).to(DEVICE)
    wavenet.load_state_dict(torch.load(CHECKPOINT_WAVENET, map_location=DEVICE))
    wavenet.eval()

    return vqvae, wavenet

def evaluate(vqvae, wavenet, dataloader):
    recon_loss_fn = torch.nn.L1Loss()
    total_l1, total_stft = 0.0, 0.0
    count = 0

    for i, (waveform, _, _) in enumerate(tqdm(dataloader)):
        waveform = waveform.to(DEVICE)  # [B, 1, T]
        waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)

        with torch.no_grad():
            z_q = vqvae.encode(waveform)
            pred = wavenet.incremental_forward(
                initial_input=None,
                c=z_q,
                T=waveform.shape[-1],
                softmax=False,
                quantize=False
            )
            pred = pred.squeeze(0)  # [1,T]
            target = waveform.squeeze(0)  # [1,T]

            l1 = recon_loss_fn(pred, target)
            stft = stft_loss(pred.unsqueeze(0), target.unsqueeze(0))

            total_l1 += l1.item()
            total_stft += stft.item()
            count += 1

            if SAVE_EXAMPLES and i < N_EXAMPLES:
                torchaudio.save(OUTPUT_DIR / f"original_{i}.wav", target.unsqueeze(0).cpu(), SAMPLE_RATE)
                torchaudio.save(OUTPUT_DIR / f"reconstructed_{i}.wav", pred.unsqueeze(0).cpu(), SAMPLE_RATE)

    print("\nEvaluation on test-clean:")
    print(f"  Avg L1 Loss   : {total_l1 / count:.4f}")
    print(f"  Avg STFT Loss : {total_stft / count:.4f}")

def main():
    vqvae, wavenet = load_models()

    dataset = LibriSpeechMelDataset(
        root=Path(DATASET_PATH),
        url="test-clean",
        segment_duration=SEGMENT_DURATION,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    evaluate(vqvae, wavenet, dataloader)

if __name__ == "__main__":
    main()
