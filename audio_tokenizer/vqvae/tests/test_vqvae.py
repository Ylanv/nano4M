from audio_tokenizer.vqvae.models.vqvae import VQVAE
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset
from audio_tokenizer.vqvae.data.audio_utils import MelToAudio
import torch 
import torchaudio
import pytest 
from pathlib import Path 
from torch.utils.data import DataLoader
from time import perf_counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@pytest.fixture
def vqvae():
    model = VQVAE().to(device)
    return model 

@pytest.fixture
def loader():
    dataset_path = Path("audio_tokenizer/vqvae/data")
    dataset = LibriSpeechMelDataset(root=dataset_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    return loader

@pytest.fixture
def melToAudio():
    return MelToAudio().to(device)

def test_vqvae_forward(vqvae,loader,melToAudio):

    for batch in loader:
        batch = batch.to(device)
        print(f"Batch Size : {batch.shape}")
        start = perf_counter()
        x_recon, vq_loss, indices = vqvae(batch)
        time_duration = perf_counter() - start 
        print(f"Reconstruction shape : {x_recon.shape}")
        print(f"VQ loss : {vq_loss}")
        print(f"Time : {time_duration:.3f} seconds")
        #wf = melToAudio(x_recon).detach().cpu()
        #torchaudio.save("audio_tokenizer/vqvae/tests/output/wf.wav",wf[0].unsqueeze(0),sample_rate=24000)
        #print("Waveform saved")
        break