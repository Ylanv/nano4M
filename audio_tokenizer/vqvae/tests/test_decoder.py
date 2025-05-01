from audio_tokenizer.vqvae.models.decoder import VQDecoder
import torch 
import pytest 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@pytest.fixture
def decoder():
    model = VQDecoder().to(device)
    return model 



def test_encoder_forward(decoder):
    z_q = torch.randn(4,64,10,23).to(device)
    recon = decoder(z_q)
    print(recon.shape)