from typing import Literal, OrderedDict, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from model.constants import CLS_TOKEN, PAD_TOKEN, SEQ_LEN, VOCAB_SIZE
from model.layer import EmbeddingPositionalEncoding, TransformerLayer

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')

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
        self.embedding = nn.Embedding(VOCAB_SIZE + 1, latent_dim, padding_idx=PAD_TOKEN)
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
        out_dim = VOCAB_SIZE + 1 #if zero_pad_value is None else VOCAB_SIZE
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

        (batch_size, latent_dim) -> (seq_len, batch_size, vocab_size + 1)
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
        #(batch_size, latent_dim) -> (seq_len, batch_size, vocab_size + 1)
        return self.forward(input)
# Container
# ------------------------------------------------------------------------------

# class VAE(L.LightningModule):
#     def __init__(self, encoder, decoder, n_steps=None):
#         super(VAE, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.automatic_optimization = False
#         self.register_buffer('steps_seen', torch.tensor(0, dtype=torch.long))
#         self.register_buffer('kld_max', torch.tensor(1.0, dtype=torch.float))
#         self.register_buffer('kld_weight', torch.tensor(0.0, dtype=torch.float))
#         if n_steps is not None:
#             self.register_buffer('kld_inc', torch.tensor((self.kld_max - self.kld_weight) / (n_steps // 2), dtype=torch.float))
#         else:
#             self.register_buffer('kld_inc', torch.tensor(0, dtype=torch.float))
#     # def loss_function(self, x_hat, x, mean, log_var):
#     #     reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
#     #     KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
#     #     return reproduction_loss + KLD, reproduction_loss, KLD
    
#     def training_step(self, batch, batch_idx):
#         # training_step defines the train loop.
#         # it is independent of forward
#         # temperature=1.0
#         x, y = batch
#         # print(x.shape)
#         #shape to encoder - 32x25
#         var = x.long()
#         m, l, z = self.encoder(var, DEVICE)
#         # print(m)
#         # print(l)
#         prior_distr = torch.distributions.Normal(torch.zeros_like(m), torch.ones_like(abs(l)))
#         q_distr = torch.distributions.Normal(m, abs(l+1e-6))
#         z_norm = q_distr.rsample()
#         # z_norm = torch.tensor(z_norm).to(DEVICE)
#         #print(z.shape)
#         #(32,100)
#         output, logits = self.decoder(z_norm)
#         #print(output.shape)
#         #(25,32)
#         #print(logits.shape)
#         #(25,32,21)
#         S, B, C = logits.shape
#         input_reshaped = logits.permute(1, 0, 2).reshape(B * S, C)
#         target_reshaped = var.reshape(B * S)
#         amino_acc, empty_acc = compare_tensors(var.t(), output)
#         loss_function = nn.CrossEntropyLoss()
#         # KLD = - 0.5 * torch.sum(1+ l - m.pow(2) - l.exp())
#         kl_loss = nn.KLDivLoss(reduction="batchmean", log_target = True)
#         input = F.log_softmax(z, dim=1)
#         target = F.log_softmax(z_norm, dim=1)
#         KLD = kl_loss(input, target)
#         # Kullback Leibler divergence
#         log_qzx = q_distr.log_prob(z_norm).sum(dim=1)
#         log_pz = prior_distr.log_prob(z_norm).sum(dim=1)

#         # KLD = (log_qzx - log_pz).sum()
#         # (wyjscie encoder, wyjscie po normal distribution)
#         reproduction_loss = loss_function(input_reshaped,
#             target_reshaped)
#         loss = reproduction_loss + KLD
#         # Logging to TensorBoard (if installed) by default
#         self.log("train_loss", loss)
#         self.log("amino_acc", amino_acc)
#         self.log("empty_acc", empty_acc)
#         self.log("reproduction_loss", reproduction_loss)
#         self.log("KLD", KLD)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         scheduler = StepLRScheduler(optimizer, decay_t = 15, decay_rate=0.5)
#         return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


#     def lr_scheduler_step(self, scheduler, metric):
#         scheduler.step(epoch=self.current_epoch)
