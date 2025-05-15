# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Some functions are based on the timm and 4M code bases
# https://github.com/huggingface/pytorch-image-models
# https://github.com/apple/ml-4m
# --------------------------------------------------------

from typing import Optional,Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class LayerNorm(nn.Module):
    """Custom implementation of LayerNorm with the option to disable the bias term."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_buffer("bias", torch.zeros(normalized_shape))

        # Normalized shape must be a tuple for F.layer_norm
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps=self.eps)


class Mlp(nn.Module):
    """
    MLP module with GELU activation.

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features (optional)
        out_features: Number of output features (optional)
        bias: Whether to include bias in the linear layers
    """
    def __init__(self, 
            in_features: int, 
            hidden_features: Optional[int] = None, 
            out_features: Optional[int] = None, 
            bias: bool = False,
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.flatten = nn.Flatten()

        # GELU(XW1 + b1)*W2 + b2
        self.GELU_Layer = nn.Sequential(
            nn.Linear(in_features = in_features,out_features = hidden_features,bias = bias), # XW1^T + b1
            nn.GELU(), # GELU
            nn.Linear(in_features = hidden_features,out_features = out_features,bias = bias), # *W2^T + b2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.GELU_Layer(x) # Pass GELU_Layer
        return logits
    


class Attention(nn.Module):
    """
    Multi-head self-attention module.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        qkv_bias: Whether to include bias in the QKV linear layers
        proj_bias: Whether to include bias in the attention output projection
    """
    def __init__(self, dim: int, head_dim: int = 64, qkv_bias: bool = False, proj_bias: bool = False):
        super().__init__()

        num_heads = dim // head_dim
        assert dim % num_heads == 0, "Dim of embeddings is not divisible by the number of heads"
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        # TODO: Define here the linear layer(s) producing K, Q, V from the input x
        # Hint: Do you need to define three different projections, or can you use a single one for all three?
        self.packed_proj = nn.Linear(in_features =dim , out_features = 3*dim ,bias = qkv_bias)

        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor, mask : Optional[torch.Tensor] = None, rotary_emb:
                Optional[nn.Module] = None, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape # Batch size, sequence length, and dimension

        # TODO: Compute the keys K, queries Q, and values V from x. Each should be of shape [B num_heads L head_dim].
        result = self.packed_proj(x)
        q, k, v = torch.chunk(result, 3, dim=-1)
        
    
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        q = q.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        k = k.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        v = v.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)

        assert q.shape == (B,self.num_heads,L,self.head_dim), f"Shape of Q is not correct should be ({B},{self.num_heads},{L},{self.head_dim})"
        assert k.shape == (B,self.num_heads,L,self.head_dim), f"Shape of K is not correct should be ({B},{self.num_heads},{L},{self.head_dim})"
        assert v.shape == (B,self.num_heads,L,self.head_dim), f"Shape of V is not correct should be ({B},{self.num_heads},{L},{self.head_dim})"
        
        if rotary_emb is not None:
            q = rearrange(rotary_emb(rearrange(q, "b h l d -> b l h d"), input_pos=input_pos), "b l h d -> b h l d")
            k = rearrange(rotary_emb(rearrange(k, "b h l d -> b l h d"), input_pos=input_pos), "b l h d -> b h l d")
        
        # TODO: Compute the attention matrix (pre softmax) and scale it by 1/sqrt(d_k). It should be of shape [B num_heads L L].
        # Hint: Use the already defined self.scale
        
        
        attn = q @ k.transpose(-2, -1) * self.scale
        if mask is not None:
            mask = rearrange(mask, "b n m -> b 1 n m") # Unsqueeze for multi-head attention
            # TODO: Apply the optional attention mask. Wherever the mask is False, replace the attention 
            # matrix value by negative infinity → zero attention weight after softmax.
            attn = attn.masked_fill_(mask.logical_not(), float('-inf'))  # Set masked positions to -inf
        
        
         
        # TODO: Compute the softmax over the last dimension
        attn = torch.softmax(attn, dim=-1)
        assert attn.shape == (B,self.num_heads,L,L),f"After softmax : {attn.shape} , should be ({B},{self.num_heads},{L},{L})"
        # TODO: Weight the values V by the attention matrix and concatenate the different attention heads
        # Make sure to reshape the output to the original shape of x, i.e. [B L D]
        x = attn @ v

        attn = x.transpose(1, 2).flatten(2, 3)

        assert attn.shape == (B,L,D), f"attn @ V is not reshaped as orginal shape : ({B},{L},{D}) , Attn shape : {attn.shape}"
        
        
        x = self.attn_out_proj(attn)
        return x

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention module.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        qkv_bias: Whether to include bias in the QKV linear layers
        proj_bias: Whether to include bias in the attention output projection
    """
    def __init__(self, dim: int, head_dim: int = 64, qkv_bias: bool = False, proj_bias: bool = False):
        super().__init__()
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        # TODO: Define here the linear layer producing Q from the input x
        self.linear_q = nn.Linear(in_features=dim,out_features=dim,bias=qkv_bias)

        # TODO: Define here the linear layers producing K, V from the context
        # Hint: Do you need to define two different projections, or can you use a single one for both?
        self.linear_kv = nn.Linear(in_features=dim,out_features=2*dim,bias=qkv_bias)

        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None,
                rotary_emb: Optional[nn.Module] = None, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape # Batch size, x sequence length (N), and dimension
        _, M, _ = context.shape # _, context sequence length (M), _

        # TODO: Compute the queries Q from x. It should be of shape [B num_heads N head_dim].
        q = self.linear_q(x)
        # TODO: Compute the keys K and values V from the context. Each should be of shape [B num_heads M head_dim].
        result = self.linear_kv(context)
        k, v = torch.chunk(result, 2, dim=-1)
        
        # reshape query, key, value to separate by head
        q = q.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        k = k.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        v = v.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        
        assert q.shape == (B,self.num_heads,N,self.head_dim), \
            f"Q wrong shape : {q.shape}"
        assert k.shape == (B,self.num_heads,M,self.head_dim), \
            f"K wrong shape : {k.shape}"
        assert v.shape == (B,self.num_heads,M,self.head_dim), \
            f"K wrong shape : {v.shape}"

        if rotary_emb is not None:
            q = rearrange(rotary_emb(rearrange(q, "b h l d -> b l h d"), input_pos=input_pos), "b l h d -> b h l d")
            k = rearrange(rotary_emb(rearrange(k, "b h l d -> b l h d"), input_pos=input_pos), "b l h d -> b h l d")
            
        # TODO: Compute the attention matrix (pre softmax) and scale it by 1/sqrt(d_k). It should be of shape [B num_heads N M].
        # Hint: Use the already defined self.scale
        attn = q @ k.transpose(-2, -1) * self.scale
        assert attn.shape == (B,self.num_heads,N,M), \
            f"Att wrong shape: {attn.shape}"
            
        if mask is not None:
            mask = rearrange(mask, "b n m -> b 1 n m") # Unsqueeze for multi-head attention
            # TODO: Apply the optional attention mask. Wherever the mask is False, replace the attention 
            # matrix value by negative infinity → zero attention weight after softmax.
            attn = attn.masked_fill_(mask.logical_not(), float('-inf'))

        # TODO: Compute the softmax over the last dimension
        attn = torch.softmax(attn, dim=-1)


        # TODO: Weight the values V by the attention matrix and concatenate the different attention heads
        # Make sure to reshape the output to the original shape of x, i.e. [B N D]
        x = attn @ v

        attn = x.transpose(1, 2).flatten(2, 3)
        assert attn.shape == (B,N,C), f"attn @ V is not reshaped as orginal shape : ({B},{N},{C}) , Attn shape : {attn.shape}"
        
    
        
        # Output projection
        x = self.attn_out_proj(attn)

        return x

class Block(nn.Module):
    """
    Basic transformer block with a multi-head self-attention mechanism and a feed-forward MLP.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(self, dim: int, head_dim: int = 64, mlp_ratio: float = 4., use_bias: bool = False):
        super().__init__()
        # LayerNorm 
        self.norm1 = LayerNorm(normalized_shape=dim,bias=use_bias) 
        self.attn = Attention(dim=dim,head_dim=head_dim,qkv_bias=use_bias,proj_bias=use_bias) 
        self.norm2 = LayerNorm(normalized_shape=dim,bias=use_bias) 
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim,bias=use_bias) 

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, rotary_emb:
                Optional[nn.Module] = None, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.dim() == 3, f"Expected input tensor x to have 3 dimensions (B, L, D), got {x.dim()}"
        B,L,D = x.shape
        
        n1 = self.norm1(x)
        assert n1.shape == (B,L,D), f"LayerNom1 changed the Input shape"
        
        x_a = x + self.attn(n1, mask, rotary_emb, input_pos)
        assert x_a.shape == (B,L,D), f"Attn layer changed the Input shape"
        
        

        n2 = self.norm2(x_a)
        assert n2.shape == (B,L,D), f"LayerNom2 changed the Input shape"
        

        r_mlp = self.mlp(n2)

        
        x_b = x_a + r_mlp
        assert x_b.shape == (B,L,D), f"Mlp layer changed the Input shape"

        return x_b

class DecoderBlock(nn.Module):
    """
    Basic transformer decoder block with a multi-head self-attention, 
    a multi-head cross-attention, and a feed-forward MLP layer.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(self, dim: int, head_dim: int = 64, mlp_ratio: float = 4., use_bias: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(normalized_shape=dim,bias=use_bias) 
        self.query_norm = LayerNorm(normalized_shape=dim,bias = use_bias) # TODO (use the LayerNorm defined above)
        self.context_norm = LayerNorm(normalized_shape=dim,bias=use_bias) # TODO (use the LayerNorm defined above)
        self.norm2 = LayerNorm(normalized_shape=dim,bias=use_bias) # TODO (use the LayerNorm defined above)

        self.self_attn = Attention(dim=dim,head_dim=head_dim,qkv_bias=use_bias,proj_bias=use_bias) # TODO Attention layer
        self.cross_attn = CrossAttention(dim=dim,head_dim=head_dim,qkv_bias=use_bias,proj_bias=use_bias) # TODO CrossAttention layer

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim,bias=use_bias) # TODO MLP layer

    def forward(self, 
            x: torch.Tensor, 
            context: torch.Tensor, 
            sa_mask: Optional[torch.Tensor] = None, # Self-attention mask
            xa_mask: Optional[torch.Tensor] = None, # Cross-attention mask
            rotary_emb: Optional[nn.Module] = None,
            input_pos: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:

        # Self-attention, then cross-attention, then MLP
        # Make sure to apply the self-attention mask (sa_mask) to the self-attention layer,
        # and the cross-attention mask (xa_mask) to the cross-attention layer.
        # Don't forget to add the residual connections after each layer, and
        # to apply the normalizations on the inputs of each layer.
        xa = x + self.self_attn(self.norm1(x), sa_mask, rotary_emb, input_pos)
        xb = xa + self.cross_attn(self.query_norm(xa), self.context_norm(context), xa_mask, rotary_emb, input_pos)
        xc = xb + self.mlp(self.norm2(xb))
        return xc
        

class TransformerTrunk(nn.Module):
    """Basic Transformer trunk definition that can be used for encoder-only,
    decoder-only and prefixLM models, depending on the attention mask applied.

    Args:
        dim: Transformer dimension
        depth: Number of transformer layers
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(
        self,
            dim: int = 512,
            depth: int = 8,
            head_dim: int = 64,
            mlp_ratio: float = 4.0,
            use_bias: bool = False,
        ):
        super().__init__()

        self.blocks = nn.ModuleList([Block(dim=dim,head_dim=head_dim,mlp_ratio=mlp_ratio,use_bias=use_bias) for _ in range(depth)])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, rotary_emb:
                Optional[nn.Module] = None, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        for block in self.blocks:
            x = block(x, mask, rotary_emb, input_pos)
        return x

class TransformerDecoderTrunk(nn.Module):
    """Basic Transformer decoder with interleaved self- and cross-attention, that can
    be used as the decoder for encoder-decoder models.

    Args:
        dim: Transformer dimension
        depth: Number of transformer layers
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(
        self,
            dim: int = 512,
            depth: int = 8,
            head_dim: int = 64,
            mlp_ratio: float = 4.0,
            use_bias: bool = False,
        ):
        super().__init__()
        blocks = []
        for i in range(depth):
            if i % 2 == 0:
                blocks.append(Block(dim=dim, head_dim=head_dim, mlp_ratio=mlp_ratio, use_bias=use_bias))
            else:
                blocks.append(DecoderBlock(dim=dim, head_dim=head_dim, mlp_ratio=mlp_ratio, use_bias=use_bias))
        self.blocks = nn.ModuleList(blocks)
        
    def forward(
            self, 
            x: torch.Tensor, 
            context: torch.Tensor, 
            sa_mask: Optional[torch.Tensor] = None, # Self-attention mask
            xa_mask: Optional[torch.Tensor] = None, # Cross-attention mask
            rotary_emb: Optional[nn.Module] = None,
            input_pos: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        
        for block in self.blocks:
            if isinstance(block, DecoderBlock):
                x = block(x, context, sa_mask, xa_mask, rotary_emb, input_pos)
            else:  # Block (self-attn only)
                x = block(x, sa_mask, rotary_emb, input_pos)
        return x
