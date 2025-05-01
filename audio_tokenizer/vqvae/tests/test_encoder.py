from audio_tokenizer.vqvae.models.encoder import VQEncoder
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset
import torch 
import pytest 
from pathlib import Path 
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@pytest.fixture
def encoder():
    model = VQEncoder().to(device)
    return model 

@pytest.fixture
def loader():
    dataset_path = Path("audio_tokenizer/vqvae/data")
    dataset = LibriSpeechMelDataset(root=dataset_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    return loader

def test_encoder_forward(encoder,loader ):

    for batch in loader:
        batch = batch.to(device)
        embeddings = encoder.forward(batch)
        print(embeddings.shape)
        break