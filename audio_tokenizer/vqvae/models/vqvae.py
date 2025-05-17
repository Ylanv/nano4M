import torch
import torch.nn as nn
from audio_tokenizer.vqvae.models.encoder import VQEncoder,RawEncoder
from audio_tokenizer.vqvae.models.decoder import VQDecoder,RawDecoder
from audio_tokenizer.vqvae.models.quantizer import VQuantizer


class VQVAE(nn.Module):
    """VQ-VAE with Mirror CNN encoder/decoder
    and Vector Quantizer using L2 distance
    """

    def __init__(
        self,
        embedding_dim=64,
        num_embeddings=512,
        commitment_cost=0.25,
        encoder_hidden_dims=[64, 128, 256],
        decoder_hidden_dims=[256, 128, 64],
    ):
        """
        Parameters:
            embedding_dim (int) :
            num_embeddings (int) :
            commitment_cost (float) :
            encoder_hidden_dims (list[int]) :
            decoder_hidden_dims (list[int]) :
        """
        super().__init__()

        self.encoder = VQEncoder(
            in_channels=1, hidden_dims=encoder_hidden_dims, embedding_dim=embedding_dim
        )

        self.quantizer = VQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        self.decoder = VQDecoder(
            embedding_dim=embedding_dim, hidden_dims=decoder_hidden_dims, out_channels=1
        )

    def forward(self, x):
        # Encoder
        z = self.encoder(x)
        # Quantizer
        z_q, vq_loss, indices = self.quantizer(z)
        # Decoder
        # [4, 10, 23, 64])
        x_recon = self.decoder(z_q.permute(0, 3, 1, 2)).squeeze(1)
        return x_recon, vq_loss, indices


class RawVQVAE(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        num_embeddings=512,
        commitment_cost=0.25,
        encoder_hidden_dims=128,
        decoder_hidden_dims=128,
    ):
        """
        Parameters:
            embedding_dim (int) :
            num_embeddings (int) :
            commitment_cost (float) :
            encoder_hidden_dims (list[int]) :
            decoder_hidden_dims (list[int]) :
        """
        super().__init__()

        self.encoder = RawEncoder(
            in_channels=1, hidden_dim=encoder_hidden_dims, num_layers=6
        )

        self.quantizer = VQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        self.decoder = RawDecoder(
            embedding_dim=embedding_dim, hidden_dim=decoder_hidden_dims, out_channels=1,num_layers=6
        )

    def forward(self, x):
        # Encoder
        z_e = self.encoder(x)
        #print(f"z_e shape:{z_e.shape}")
        # Quantizer
        z_q, vq_loss = self.quantizer(z_e)
        #print(f"z_q shape:{z_q.shape}")
        # Decoder
        x_hat = self.decoder(z_q)
        #print(f"x_hat shape:{x_hat.shape}")
        return x_hat, vq_loss
    
    def encode(self,x):
        # Encoder
        z_e = self.encoder(x)
        #print(f"z_e shape:{z_e.shape}")
        # Quantizer
        z_q, vq_loss = self.quantizer(z_e)
        #print(f"z_q shape:{z_q.shape}")
        return z_q
    
if __name__ == "__main__":
        x = torch.rand((16,1,32000))
        model = RawVQVAE()
        x_hat,vq_loss = model(x)
        #print(f"Reconstruction shape : {x_hat.shape}")
        #print(f"VQ LOSS : {vq_loss}")    