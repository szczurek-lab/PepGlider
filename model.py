import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from functools import wraps
import lightning as L
import numpy as np
from math import e
from timm.scheduler import StepLRScheduler
from metrics import compare_tensors

# if __package__ is None or __package__ == '':
#     from datasets import *
# else:
#     from pytorchtextvae.datasets import *

MAX_SAMPLE = False
TRUNCATED_SAMPLE = True
model_random_state = np.random.RandomState(1988)
torch.manual_seed(1999)
SOS_token = 0
EOS_token = 1

def _decorate(forward, module, name, name_g, name_v):
    @wraps(forward)
    def decorated_forward(*args, **kwargs):
        g = module.__getattr__(name_g)
        v = module.__getattr__(name_v)
        w = v*(g/torch.norm(v)).expand_as(v)
        module.__setattr__(name, w)
        return forward(*args, **kwargs)
    return decorated_forward

class Encoder(nn.Module):
    def sample(self, mu, logvar, device):
        eps = Variable(torch.randn(mu.size())).to(device)
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

# Encoder
# ------------------------------------------------------------------------------

# Encode into Z with mu and log_var

class EncoderRNN(Encoder):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.1, bidirectional=bidirectional)
        self.o2p = nn.Linear(hidden_size, output_size * 2)

    def forward(self, input, device):
        embedded = self.embed(input)
        embedded = embedded.permute(1, 0, 2)
        output, hidden = self.gru(embedded, None)
        output = output[-1]
        if self.bidirectional:
            output = output[:, :self.hidden_size] + output[: ,self.hidden_size:]
        else:
            output = output[:, :self.hidden_size]

        ps = self.o2p(output)
        mu, logvar = torch.chunk(ps, 2, dim=1)
        z = self.sample(mu, logvar, device)
        return mu, logvar, z

# Decoder
# ------------------------------------------------------------------------------

# Decode from Z into sequence

class DecoderRNN(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc = nn.Linear(latent_size, hidden_size, device=DEVICE)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, bidirectional=True, device=DEVICE)
        self.fc_output = nn.Linear(hidden_size*2, output_size, device=DEVICE)

    def forward(self, z):
        SEQ_LEN = 25
        z = z.unsqueeze(0).repeat(SEQ_LEN, 1, 1)
        #print(z.shape)
        #(25,32,100)
        S, B, L = z.shape
        z = self.fc(z)
        #(25,32,512)
        num_directions = 2  # For bidirectional LSTM
        hidden_size = self.gru.hidden_size
        z, _ = self.gru(z.to(DEVICE), torch.zeros(self.num_layers*2, B, hidden_size).to(DEVICE))
        #print(z.shape)
        #(25,32,1024)
        output = self.fc_output(z)
        #print(output.shape)
        #(25,32,21)
        return output.argmax(dim=2), output

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def generate(self, batch_size):
        input = torch.randn(batch_size, 100).to(DEVICE)
        return self.forward(input)

# Container
# ------------------------------------------------------------------------------
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
class VAE(L.LightningModule):
    def __init__(self, encoder, decoder, n_steps=None):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.automatic_optimization = False
        self.register_buffer('steps_seen', torch.tensor(0, dtype=torch.long))
        self.register_buffer('kld_max', torch.tensor(1.0, dtype=torch.float))
        self.register_buffer('kld_weight', torch.tensor(0.0, dtype=torch.float))
        if n_steps is not None:
            self.register_buffer('kld_inc', torch.tensor((self.kld_max - self.kld_weight) / (n_steps // 2), dtype=torch.float))
        else:
            self.register_buffer('kld_inc', torch.tensor(0, dtype=torch.float))
    def loss_function(self, x_hat, x, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD, reproduction_loss, KLD
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        # temperature=1.0
        x, y = batch
        # print(x.shape)
        #shape to encoder - 32x25
        m, l, z = self.encoder(x, DEVICE)
        #print(z.shape)
        #(32,100)
        output, logits = self.decoder(z)
        #print(output.shape)
        #(25,32)
        #print(logits.shape)
        #(25,32,21)
        S, B, C = logits.shape
        input_reshaped = logits.permute(1, 0, 2).reshape(B * S, C)
        target_reshaped = x.permute(1, 0).reshape(B * S)
        amino_acc, empty_acc = compare_tensors(x.t(), output)
        loss_function = nn.CrossEntropyLoss()
        KLD = - 0.5 * torch.sum(1+ l - m.pow(2) - l.exp())
        reproduction_loss = loss_function(input_reshaped,
            target_reshaped)
        loss = reproduction_loss + KLD
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("amino_acc", amino_acc)
        self.log("empty_acc", empty_acc)
        self.log("reproduction_loss", reproduction_loss)
        self.log("KLD", KLD)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = StepLRScheduler(optimizer, decay_t = 5, decay_rate=0.5)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
