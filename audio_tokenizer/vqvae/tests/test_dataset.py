from torch.utils.data import DataLoader
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset
from pathlib import Path
import pytest


@pytest.fixture
def loader():
    dataset_path = Path("audio_tokenizer/vqvae/data")
    dataset = LibriSpeechMelDataset(root=dataset_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    return loader


def test_iterate_loader(loader):
    print(len(loader))
    for batch in loader:
        print(batch.shape)
        assert batch.ndim == 3
        break
