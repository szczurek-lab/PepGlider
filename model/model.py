from typing import Literal, OrderedDict, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from model.constants import CLS_TOKEN, PAD_TOKEN, SEQ_LEN, VOCAB_SIZE
from model.layer import EmbeddingPositionalEncoding, TransformerLayer
from torch import nn, cuda, backends, manual_seed, tensor

DEVICE = torch.device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')

class EncoderRNN(nn.Module):
    _EPS = 1e-5

    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        latent_dim: int,
        pos_enc_type: Literal["add", "concat"],
        dropout: float = 0.1,
        layer_norm: bool = True,
    ):
        super().__init__()
        # + pad and cls token
        self.embedding = nn.Embedding(VOCAB_SIZE+1, latent_dim, padding_idx=PAD_TOKEN)
        self.pos_encoder = EmbeddingPositionalEncoding(
            transform=pos_enc_type, num_tokens=SEQ_LEN + 1, embedding_dim=latent_dim
        )
        self.attention_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"att{i}",
                        TransformerLayer(num_heads, latent_dim, dropout, layer_norm),
                    )
                    for i in range(num_layers)
                ]
            )
        )
        self.linear = nn.Linear(latent_dim, 2 * latent_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _add_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        cls_tensor = (
            torch.ones(1, x.shape[1], dtype=torch.long, device=x.device) * CLS_TOKEN
        )
        return torch.concatenate([cls_tensor, x], dim=0)

    def compute_representations(self, src: torch.Tensor) -> torch.Tensor:
        """(seq_len, batch_size) -> (seq_len + 1, batch_size, latent_dim)"""
        x = self._add_cls_token(src)
        assert not (torch.isnan(x).all() ), f"add_cls_token contains all NaN values: {x}"
        assert not (torch.isinf(x).all() ), f"add_cls_token contains all Inf values: {x}"
        x = self.embedding(x)
        assert not (torch.isnan(x).all() ), f"embedding contains all NaN values: {x}"
        assert not (torch.isinf(x).all() ), f"embedding contains all Inf values: {x}"
        x = self.pos_encoder(x)
        assert not (torch.isnan(x).all() ), f"pos_encoder contains all NaN values: {x}"
        assert not (torch.isinf(x).all() ), f"pos_encoder contains all Inf values: {x}"
        x = self.attention_layers(x)
        assert not (torch.isnan(x).all() ), f"transformer contains all NaN values: {x}"
        assert not (torch.isinf(x).all() ), f"transformer contains all Inf values: {x}"
        return x

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(seq_len, batch_size) -> tuple[(batch_size, latent_dim), (batch_size, latent_dim)]"""
        x = self.compute_representations(src)[0]  # cls token is first
        x = self.linear(x) #czy on zwraca nan assert
        assert not (torch.isnan(x).all() ), f"linear contains all NaN values: {x}"
        assert not (torch.isinf(x).all() ), f"linear contains all Inf values: {x}"
        mu, std_out = torch.chunk(x, 2, dim=1)
        std = F.softplus(std_out) + self._EPS
        return mu, std

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """(seq_len, batch_size) -> (batch_size, latent_dim)"""
        mu, std = self.forward(src)
        return mu

# Decoder
# ------------------------------------------------------------------------------

# Decode from Z into sequence

class DecoderRNN(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        latent_dim: int,
        pos_enc_type: Literal["add", "concat"],
        dropout: float = 0.1,
        layer_norm: bool = True,
        zero_pad_value: Union[float, None] = 0,
    ):
        super().__init__()
        self.pos_encoder = EmbeddingPositionalEncoding(
            transform=pos_enc_type, num_tokens=SEQ_LEN, embedding_dim=latent_dim
        )
        self.attention_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"att{i}",
                        TransformerLayer(num_heads, latent_dim, dropout, layer_norm),
                    )
                    for i in range(num_layers)
                ]
            )
        )
        out_dim = VOCAB_SIZE #if zero_pad_value is None else VOCAB_SIZE
        self.linear = nn.Linear(latent_dim, out_dim)  # NOTE: 1 dim less
        self.zero_pad_value = zero_pad_value
        self._init_weights() 

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _create_sequence(x: torch.Tensor) -> torch.Tensor:
        """(batch_size, latent_dim) -> (seq_len, batch_size, latent_dim)"""
        return x.unsqueeze(0).repeat(SEQ_LEN, 1, 1)
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Return unnormalized values for each token of the sequence.

        (batch_size, latent_dim) -> (seq_len, batch_size, vocab_size)
        """
        x = self._create_sequence(src)
        x = self.pos_encoder(x)  # S x B x L
        x = self.attention_layers(x)  # S x B x L

        S, B, L = x.shape
        model_out = self.linear(x.reshape(S * B, L)).reshape(S, B, -1)
        return model_out

    def decode(self, src: torch.Tensor) -> torch.Tensor:
        """Return predicted tokens.

        (batch_size, latent_dim) -> (seq_len, batch_size)
        """
        logits = self.forward(src)
        return logits.argmax(dim=2)
    
    def generate(self, batch_size, latent_dim):
        input = torch.randn(batch_size, latent_dim).to(DEVICE)
        #(batch_size, latent_dim) -> (seq_len, batch_size, vocab_size)
        return self.forward(input)
    
    def generate_from(self, batch_size, latent_dim, dim1, shift_value):
        # shift_value = 2.0 # Wartość przesunięcia (średnia będzie 2)
        std_dev = 1.0     # Odchylenie standardowe (jak w standardowym rozkładzie normalnym)
        # x1 = (torch.randn(200) * std_dev + shift_value).to(DEVICE)
        # x2 = (torch.randn(200) * std_dev + shift_value).to(DEVICE)
        # z1, z2 = torch.meshgrid([x1, x2], indexing='ij')
        # num_points = z1.size(0) * z1.size(1) 
        # mod_dim = torch.randn(batch_size, 1) + shift_value

        # print(f"Kształt z1 po meshgrid (przed view): {z1.shape}")
        # print(f"Kształt z2 po meshgrid (przed view): {z2.shape}")
        # print(f"Przykładowa średnia z1: {z1.mean().item():.2f}")
        # print(f"Przykładowa średnia z2: {z2.mean().item():.2f}")
        z = torch.randn(batch_size, latent_dim).to(DEVICE) # Generowanie losowego wektora z
        z[:, dim1] = (z[:, dim1] + shift_value).to(DEVICE)
        # z = z.repeat(num_points, 1).to(DEVICE) # Powielenie go do rozmiaru num_points x z_dim
        # z[:, dim1] = z1.to(DEVICE).contiguous().view(-1) # Spłaszcz z1 do 1D i przypisz do kolumny d
        # z[:, dim2] = z2.to(DEVICE).contiguous().view(-1)
        #(batch_size, latent_dim) -> (seq_len, batch_size, vocab_size)
        return self.forward(z)
    