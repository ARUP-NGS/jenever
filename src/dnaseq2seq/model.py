

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


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block:
      x -> [Linear_a, Linear_b] -> SiLU(a) * b -> Dropout -> Linear_out
    Default hidden size keeps params ~constant vs. standard FFN:
      hidden = int(2/3 * dim_feedforward)
    """
    def __init__(
        self,
        in_features: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        out_features: int | None = None,
        layer_norm_eps: float = 1e-5,
        init_xavier: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden = int(2 * dim_feedforward / 3)

        self.linear_a = nn.Linear(in_features, hidden)
        self.linear_b = nn.Linear(in_features, hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden, out_features)

        if init_xavier:
            for m in (self.linear_a, self.linear_b, self.linear_out):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.linear_a(x)
        b = self.linear_b(x)
        x = F.silu(a) * b
        x = self.dropout(x)
        return self.linear_out(x)

class TransformerEncoderLayerSwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.ff = SwiGLU(d_model, dim_feedforward, dropout=dropout, out_features=d_model)
        self.dropout_ff = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.norm_first = norm_first

    def _sa(self, x, attn_mask, key_padding_mask, is_causal):
        y, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal)
        return self.dropout_sa(y)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            x = x + self.dropout_ff(self.ff(self.norm2(x)))
        else:
            x = self.norm1(x + self._sa(x, src_mask, src_key_padding_mask, is_causal))
            x = self.norm2(x + self.dropout_ff(self.ff(x)))
        return x


class TransformerDecoderLayerSwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.ff = SwiGLU(d_model, dim_feedforward, dropout=dropout, out_features=d_model)
        self.norm_first = norm_first

    def _sa(self, x, attn_mask, key_padding_mask, is_causal):
        y, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal)
        return self.drop1(y)

    def _ca(self, x, mem, attn_mask, key_padding_mask):
        y, _ = self.cross_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.drop2(y)

    def forward(
        self, tgt, memory, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False
    ):
        x = tgt
        if self.norm_first:
            x = x + self._sa(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._ca(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self.drop3(self.ff(self.norm3(x)))
        else:
            x = self.norm1(x + self._sa(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._ca(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self.drop3(self.ff(x)))
        return x


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
                 cls_head_output=1,
                 device='cpu'):
        super().__init__()

        self.device = device
        self.read_depth = read_depth
        self.kmer_dim = kmer_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.embed_dim = encoder_attention_heads * embed_dim_factor
        self.fc1_hidden = 12
        self.cls_head_output = cls_head_output
        self.fc1 = nn.Linear(feature_count, self.fc1_hidden)
        self.fc2 = nn.Linear(self.read_depth * self.fc1_hidden, self.embed_dim)

        self.converter = nn.Linear(self.embed_dim, self.decoder_embed_dim)
        self.pos_encoder = PositionalEncoding2D(self.fc1_hidden, self.device)
        self.tgt_pos_encoder = PositionalEncoding(self.kmer_dim, batch_first=True, max_len=500).to(self.device)
        logger.debug(f"tgt pos encoder: {self.tgt_pos_encoder.pe.shape}, embed dim: {self.decoder_embed_dim}")
        encoder_layers = TransformerEncoderLayerSwiGLU(
            d_model=self.embed_dim,
            nhead=encoder_attention_heads,
            dim_feedforward=d_ff,
            dropout=p_dropout,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_encoder_layers)

        decoder_layers = TransformerDecoderLayerSwiGLU(
            d_model=self.decoder_embed_dim,
            nhead=decoder_attention_heads,
            dim_feedforward=d_ff,
            dropout=p_dropout,
            batch_first=True)

        self.tgt_input_converter = nn.Linear(self.kmer_dim, self.decoder_embed_dim)
        self.decoder0 = nn.TransformerDecoder(decoder_layers, num_layers=n_decoder_layers)
        self.decoder1 = nn.TransformerDecoder(decoder_layers, num_layers=n_decoder_layers)

        # Un-embedding layers
        self.decode_output_converter0 = nn.Linear(self.decoder_embed_dim, self.kmer_dim)
        self.decode_output_converter1 = nn.Linear(self.decoder_embed_dim, self.kmer_dim)

        self.softmax = nn.LogSoftmax(dim=-1)
        self.emb_layernorm = nn.LayerNorm(self.embed_dim)
        self.emb_dropout = nn.Dropout(p_dropout)
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.cls_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim //2),
            nn.GELU(),
            nn.Linear(self.embed_dim //2, self.cls_head_output),
        )


    def encode(self, src):
        src = F.gelu(self.fc1(src)) # Operates on each "feature" (10 feature encoded base)
        src = self.pos_encoder(src)  # For 2D encoding we have to do this before flattening, right?
        src = src.flatten(start_dim=2)
        src = F.gelu(self.fc2(src)) # Operates on an entire alignment column
        src = self.emb_dropout(self.emb_layernorm(src))
        
        # Append CLS token to the beginning of the sequence
        batch_size = src.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        src = torch.cat([cls_tokens, src], dim=1)  # (batch_size, seq_len + 1, embed_dim)
        
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

        h0 = self.decode_output_converter0(h0)
        h1 = self.decode_output_converter1(h1)

        h0 = self.softmax(h0)
        h1 = self.softmax(h1)
        return torch.stack((h0, h1), dim=1)

    def forward(self, src, tgt, tgt_mask, tgt_key_padding_mask=None):
        mem = self.encode(src)
        result = self.decode(mem, tgt.float(), tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return result

