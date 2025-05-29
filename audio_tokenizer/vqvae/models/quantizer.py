import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
Vector Quantizer, code is inspired from Misha Laskin VQ-VAE pytorch implementation
Github Repository : https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
"""


class VQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        """Init the quantizer module

        Parameters:
            num_embeddings (int) : Number of token in the vocabulary
            embedding_dim (int) : Dimension of each embedding
            commitement_cost (float) : Balance how strongly the encoder is encouraged to commit to the nearest codebook vector
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitement_cost = commitment_cost

        # Codebook use for VQ
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize weight uniformily between [-1/K,1/K] , K = voc_size
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        """Map z to a discrete one-hot vector that is index of the closest embeddings
        Parameters:
            z (Tensor) : Output of the encoder module
        """
        device = z.device
        
        # Reshape : [B,emb_d,T//64] => [B,T//64,emb_d]
        z_e = z.permute(0, 2, 1).contiguous()
        
        # Flatten :[B,n_tok,emb_d] => [B*n_tok,embd_d]
        z_flat = z_e.view(-1, self.embedding_dim)
        

        # Calculate L2 distance between z and embeddings in the codebook
        z2 = torch.sum(z_flat**2, dim=1, keepdim=True)
        e2 = torch.sum(torch.pow(self.codebook.weight, 2), dim=1)
        ez = torch.matmul(z_flat, self.codebook.weight.t())
        distances = z2 + e2 - 2 * ez

        # Get encoding with closest dist [num_tok,1]
        e_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        
        # Convert one hot
        encoding_one_hot = torch.zeros(e_indices.size(0), self.num_embeddings, device=device)
        encoding_one_hot = encoding_one_hot.scatter_(1, e_indices, 1)  # [B*T',num_emb]
        
        
        # Quantize
        z_q = torch.matmul(encoding_one_hot, self.codebook.weight) 
        #[BT',D] =>  [B,T',D] 
        z_q = z_q.view(z_e.shape)  
        
        
        # Loss
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        embedding_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = commitment_loss * self.commitement_cost + embedding_loss
        
        z_q = z_e + (z_q - z_e).detach()
    
        return z_q.permute(0,2,1).contiguous(),vq_loss

if __name__ == "__main__":
    x = torch.rand((16,128,500))
    model = VQuantizer(num_embeddings=512,embedding_dim=128)
    z_q,vq_loss = model(x)
    print(f"z_q shape :{z_q.shape}")
    print(f"VQ_loss : {vq_loss}")