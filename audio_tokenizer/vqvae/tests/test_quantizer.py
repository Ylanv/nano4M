from audio_tokenizer.vqvae.models.quantizer import VQuantizer
import torch
import pytest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def quantizer():
    model = VQuantizer(num_embeddings=512, embedding_dim=64).to(device)
    return model


def test_quantizer_forward(quantizer):
    z = torch.randn(4, 64, 10, 23).to(device)
    z_q, loss, indices = quantizer(z)
    print(z_q.shape)
    print(loss)
    print(indices.shape)
