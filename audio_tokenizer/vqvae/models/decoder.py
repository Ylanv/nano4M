import torch
import torch.nn as nn
from audio_tokenizer.vqvae.models.residual import ResidualStack

class VQDecoder(nn.Module):
    """Decoder, mirror of the encoder, use ConvTranspose2d for upsampling"""

    def __init__(self, embedding_dim=64, hidden_dims=[256, 128, 64], out_channels=1):
        """Init the VQDecoder

        Parameters:
            embedding_dim (int) : Dimension of each embeddings
            hidden_dims (list[int]) : Dimensions of hidden layer
            out_channels (int) :
        """
        super().__init__()

        layers = []
        current_channels = embedding_dim
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        current_channels, h_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(inplace=True),
                )
            )
            current_channels = h_dim

        # Final layer to get back to mel-spectrogram shape
        layers.append(
            nn.ConvTranspose2d(
                current_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,  # Keep output resolution
            )
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class RawDecoder(nn.Module):
    def __init__(self,embedding_dim=128, hidden_dim=128, out_channels=1, num_layers=6):
        super().__init__()
        layers = []
        layers.append(
                ResidualStack(
                    in_dim= embedding_dim,
                    h_dim= embedding_dim,
                    res_h_dim= hidden_dim,
                    n_res_layers= 2,
                )
        )
        for i in range(num_layers):
            in_channels = embedding_dim if i == 0 else hidden_dim
            out_ch = out_channels if i == num_layers - 1 else hidden_dim

            layers.append(nn.ConvTranspose1d(
                in_channels,
                out_ch,
                kernel_size=4,
                stride=2,
                padding=1
            ))

            
            if i != num_layers - 1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())  # to bound output in [-1, 1]
        self.decoder = nn.Sequential(*layers)
    def forward(self,x):
        return self.decoder(x)
    
if __name__ == "__main__":
    x = torch.rand((16,128,500))
    model = RawDecoder()
    x_hat = model(x)
    print(f"Reconstruction shape : {x_hat.shape}")