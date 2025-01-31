

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


logger = logging.getLogger(__name__)

class PositionalEncoding2D(nn.Module):

    def __init__(self, channels, device):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.device = device
        channels = int(np.ceil(channels/4)*2)
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)
        self.cache_shape = None
        self.enc_cache = None

    def _from_cache(self, tensor):
        shape = list(tensor.size())[1:]
        if shape == self.cache_shape and self.enc_cache is not None:
            return self.enc_cache
        else:
            batch_size, x, y, orig_ch = tensor.shape
            pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
            pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
            sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
            sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
            emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
            emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
            emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(tensor.type())
            emb[:, :, :self.channels] = emb_x
            emb[:, :, self.channels:2 * self.channels] = emb_y
            self.enc_cache = emb
            self.cache_shape = shape
        return self.enc_cache

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape

        #emb = self._from_cache(tensor)
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y
        #emb = emb.bfloat16()
        if tensor.get_device() > -1 and tensor.get_device() != emb.get_device():
            emb = emb.to(tensor.get_device())
        return tensor + emb[None, :, :, :orig_ch].expand(batch_size, -1, -1, -1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if div_term.shape[0] % 2:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.transpose(0,1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if self.batch_first:
            x = x + self.pe[:, 0:x.size(1), :]
        else:
            x = x + self.pe[0:x.size(0), :, :]
        return self.dropout(x)


class MultitokenHead(nn.Module):
    """
    A cheap and easy way to have the decoder output multiple heads instead of concatenating them.
    """
    def __init__(self, embed_dim, heads, d_ff, p_dropout):
        super().__init__()
        self.heads = nn.ModuleList([nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=1,
            dim_feedforward=d_ff,
            dropout=p_dropout,
            batch_first=True,
            activation='gelu') for _ in range(heads)])
    
    def forward(self, tgt, mem, tgt_mask, tgt_key_padding_mask=None):
        head_outputs = []
        for head in self.heads:
            head_outputs.append(
                head(tgt, mem, tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            )

        return torch.stack(head_outputs, dim=1)


class VarTransformer(nn.Module):

    def __init__(self,
                 read_depth,
                 feature_count,
                 kmer_dim,
                 encoder_attention_heads,
                 decoder_attention_heads,
                 d_ff,
                 embed_dim_factor,
                 n_encoder_layers,
                 n_decoder_layers,
                 decoder_embed_dim,
                 p_dropout=0.1,
                 device='cpu'):
        super().__init__()

        self.device = device
        self.read_depth = read_depth
        self.kmer_dim = kmer_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.embed_dim = encoder_attention_heads * embed_dim_factor
        self.fc1_hidden = 12

        self.fc1 = nn.Linear(feature_count, self.fc1_hidden)
        self.fc2 = nn.Linear(self.read_depth * self.fc1_hidden, self.embed_dim)

        self.converter = nn.Linear(self.embed_dim, self.decoder_embed_dim)
        self.pos_encoder = PositionalEncoding2D(self.fc1_hidden, self.device)
        self.tgt_pos_encoder = PositionalEncoding(self.kmer_dim, batch_first=True, max_len=500).to(self.device)
        logger.info(f"tgt pos encoder: {self.tgt_pos_encoder.pe.shape}, embed dim: {self.decoder_embed_dim}")
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=encoder_attention_heads,
            dim_feedforward=d_ff,
            dropout=p_dropout,
            batch_first=True,
            activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_encoder_layers)

        decoder_layers = nn.TransformerDecoderLayer(
            d_model=self.decoder_embed_dim,
            nhead=decoder_attention_heads,
            dim_feedforward=d_ff,
            dropout=p_dropout,
            batch_first=True,
            activation='gelu')

        self.tgt_input_converter = nn.Linear(self.kmer_dim, self.decoder_embed_dim)
        self.decoder0 = nn.TransformerDecoder(decoder_layers, num_layers=n_decoder_layers)
        self.decoder1 = nn.TransformerDecoder(decoder_layers, num_layers=n_decoder_layers)

        n_tokens_to_predict = 4
        self.multihead0 = MultitokenHead(self.decoder_embed_dim, n_tokens_to_predict, d_ff, p_dropout)
        self.multihead1 = MultitokenHead(self.decoder_embed_dim, n_tokens_to_predict, d_ff, p_dropout)

        self.decode_output_converter0 = nn.Linear(self.decoder_embed_dim, self.kmer_dim)
        self.decode_output_converter1 = nn.Linear(self.decoder_embed_dim, self.kmer_dim)

        self.softmax = nn.LogSoftmax(dim=-1)
        self.emb_layernorm = nn.LayerNorm(self.embed_dim)
        self.emb_dropout = nn.Dropout(p_dropout)


    def encode(self, src):
        src = F.gelu(self.fc1(src)) # Operates on each "feature" (10 feature encoded base)
        src = self.pos_encoder(src)  # For 2D encoding we have to do this before flattening, right?
        src = src.flatten(start_dim=2)
        src = F.gelu(self.fc2(src)) # Operates on an entire alignment column
        src = self.emb_dropout(self.emb_layernorm(src))
        mem = self.encoder(src)
        return mem

    def decode(self, mem, tgt, tgt_mask, tgt_key_padding_mask=None):
        mem_proj = self.converter(mem)

        tgt0 = self.tgt_pos_encoder(tgt[:, 0, :, :])
        tgt1 = self.tgt_pos_encoder(tgt[:, 1, :, :])

        # Convert to decoder embedding (model dimension) size
        tgt0 = self.tgt_input_converter(tgt0)
        tgt1 = self.tgt_input_converter(tgt1)

        # The magic of DataParallel mistakenly modifies the first dimension of the tgt mask when running on multi-GPU setups
        # This hack just forces it to be a square again
        if tgt_mask.shape[0] != tgt_mask.shape[1]:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_mask.shape[1]).to(self.device)
            #logger.info(f"Forcing tgt mask shapre to be {tgt_mask.shape}, input enc shape is: {mem.shape}")
        h0 = self.decoder0(tgt0, mem_proj, tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        h1 = self.decoder1(tgt1, mem_proj, tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        # TODO: Probably need to adjust the tgt_mask? Or maybe not?
        h0toks = self.multihead0(tgt0, h0, tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        h1toks = self.multihead1(tgt1, h1, tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        h0 = self.decode_output_converter0(h0toks)
        h1 = self.decode_output_converter1(h1toks)

        h0 = self.softmax(h0)
        h1 = self.softmax(h1)
        return torch.stack((h0, h1), dim=1)

    def forward(self, src, tgt, tgt_mask, tgt_key_padding_mask=None):
        mem = self.encode(src)
        result = self.decode(mem, tgt.float(), tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return result

