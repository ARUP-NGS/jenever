

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import TransformerDecoder, TransformerSelfAttentionLayer, TransformerCrossAttentionLayer, MultiHeadAttention, FeedForward
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


class CLSClassifier(nn.Module):
    """
    A simple classifier for the CLS token.
    """
    def __init__(self, embed_dim, cls_output_classes):
        super().__init__()
        self.cls_predictor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, cls_output_classes)
        )

    def forward(self, x):
        return self.cls_predictor(x)


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
                 cls_output_classes=1,
                 device='cpu'):
        super().__init__()

        self.device = device
        self.cls_token = torch.zeros((1,feature_count)).to(device)
        self.cls_token[:, 0:4] = 1
        self.read_depth = read_depth
        self.kmer_dim = kmer_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.embed_dim = encoder_attention_heads * embed_dim_factor
        self.fc1_hidden = 12
        self.cls_classifier = CLSClassifier(self.embed_dim, cls_output_classes)
        self.hap0_ref_classifier = CLSClassifier(self.embed_dim, 1)
        self.hap1_ref_classifier = CLSClassifier(self.embed_dim, 1)
        self.hap0_hap1_classifier = CLSClassifier(self.embed_dim, 1)

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
        self.decode_output_converter0 = nn.Linear(self.decoder_embed_dim, self.kmer_dim)
        self.decode_output_converter1 = nn.Linear(self.decoder_embed_dim, self.kmer_dim)

        self.softmax = nn.LogSoftmax(dim=-1)
        self.emb_layernorm = nn.LayerNorm(self.embed_dim)
        self.emb_dropout = nn.Dropout(p_dropout)


    def encode(self, src):
        # Add CLS token to the beginning of the sequence, the dimensions of src are (batch_size, seq_len, read_depth, feature_count)
        src = torch.cat((self.cls_token.unsqueeze(0).repeat(src.shape[0], 1, src.shape[2], 1), src), dim=1)

        src = F.gelu(self.fc1(src)) # Operates on each "feature" (10 feature encoded base)
        src = self.pos_encoder(src)  # For 2D encoding we have to do this before flattening, right?
        src = src.flatten(start_dim=2)
        src = F.gelu(self.fc2(src)) # Operates on an entire alignment column
        src = self.emb_dropout(self.emb_layernorm(src))
        mem = self.encoder(src)

        cls_embed = mem[:, 0, :]
        cls_pred = self.cls_classifier(cls_embed)
        hap0_ref_pred = self.hap0_ref_classifier(cls_embed)
        hap1_ref_pred = self.hap1_ref_classifier(cls_embed)
        hap0_hap1_pred = self.hap0_hap1_classifier(cls_embed)
        return mem, cls_pred, hap0_ref_pred, hap1_ref_pred, hap0_hap1_pred

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
        mem, cls_pred, hap0_ref_pred, hap1_ref_pred, hap0_hap1_pred = self.encode(src)
        result = self.decode(mem, tgt.float(), tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return result, cls_pred, hap0_ref_pred, hap1_ref_pred, hap0_hap1_pred


class TransformerEncoderStack(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, num_kv_heads, dropout=0.1, ff_factor=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.ff_factor = ff_factor
        self.layers = nn.ModuleList([
            TransformerSelfAttentionLayer(
                attn=MultiHeadAttention(
                    embed_dim=embed_dim, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=embed_dim//num_heads,
                    q_proj = nn.Linear(embed_dim, embed_dim, bias=False),
                    k_proj = nn.Linear(embed_dim, embed_dim // (num_heads//num_kv_heads), bias=False),
                    v_proj = nn.Linear(embed_dim, embed_dim // (num_heads//num_kv_heads), bias=False),
                    output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                    attn_dropout=dropout,
                    is_causal=False,  # Encoder should use bidirectional attention
                    ),
                sa_norm=nn.LayerNorm(embed_dim),
                mlp=nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * ff_factor),
                    nn.GELU(),
                    nn.Linear(embed_dim * ff_factor, embed_dim),
                ),
                mlp_norm=nn.LayerNorm(embed_dim),
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ZeroModule(nn.Module):
    """A module that always returns zeros. Used to disable the MLP residual in attention layers."""
    def forward(self, x):
        return torch.zeros_like(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, dropout=0.1, ff_factor=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.ff_factor = ff_factor
        # Use ZeroModule for mlp to disable the MLP residual inside self-attention
        # (we handle the MLP separately with cross-attention feeding into it)
        self.self_attention_layer = TransformerSelfAttentionLayer(
            attn=MultiHeadAttention(
                embed_dim=embed_dim, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=embed_dim//num_heads,
                q_proj = nn.Linear(embed_dim, embed_dim, bias=False),
                k_proj = nn.Linear(embed_dim, embed_dim // (num_heads//num_kv_heads), bias=False),
                v_proj = nn.Linear(embed_dim, embed_dim // (num_heads//num_kv_heads), bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                attn_dropout=dropout, 
            ),
            sa_norm=nn.LayerNorm(embed_dim),
            mlp=ZeroModule(),
        )
        # Cross-attention layer with FeedForward included (standard decoder layer pattern)
        self.cross_attention_layer = TransformerCrossAttentionLayer(
            attn=MultiHeadAttention(
                embed_dim=embed_dim, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=embed_dim//num_heads,
                q_proj = nn.Linear(embed_dim, embed_dim, bias=False),
                k_proj = nn.Linear(embed_dim, embed_dim // (num_heads//num_kv_heads), bias=False),
                v_proj = nn.Linear(embed_dim, embed_dim // (num_heads//num_kv_heads), bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                attn_dropout=dropout,
                is_causal=False,  # Cross-attention should attend to ALL encoder positions
            ),
            ca_norm=nn.LayerNorm(embed_dim),
            mlp=FeedForward(gate_proj=nn.Linear(embed_dim, embed_dim * ff_factor),
                            up_proj=nn.Linear(embed_dim, embed_dim * ff_factor),
                            down_proj=nn.Linear(embed_dim * ff_factor, embed_dim)),
            mlp_norm=nn.LayerNorm(embed_dim),
        )
    
    def forward(self, x, encoder_input, mask=None, encoder_mask=None):
        x = self.self_attention_layer(x, mask=mask)
        x = self.cross_attention_layer(x, encoder_input=encoder_input, encoder_mask=encoder_mask)
        return x


class TransformerDecoderStack(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, num_kv_heads, dropout=0.1, ff_factor=2):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, num_kv_heads, dropout, ff_factor)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, encoder_input, mask=None, encoder_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_input, mask=mask, encoder_mask=encoder_mask)
        return x

class NewVarTransformer(nn.Module):
    def __init__(self,
                 read_depth: int,
                 feature_count: int,
                 encoder_embed_dim,
                 encoder_attention_heads: int,
                 encoder_num_kv_heads: int,
                 encoder_ff_factor: int,
                 decoder_embed_dim: int,
                 decoder_attention_heads: int,
                 decoder_num_kv_heads: int,
                 decoder_ff_factor: int,
                 kmer_dim: int,
                 n_encoder_layers: int,
                 n_decoder_layers: int,
                 cls_output_classes=1,
                 device='cpu',
    ):
        super().__init__()
        self.read_depth = read_depth
        self.feature_count = feature_count
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_num_kv_heads = encoder_num_kv_heads
        self.encoder_ff_factor = encoder_ff_factor
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_num_kv_heads = decoder_num_kv_heads
        self.decoder_ff_factor = decoder_ff_factor

        self.device = device
        self.cls_token = torch.zeros((1,feature_count)).to(device)
        self.cls_token[:, 0:4] = 1
        self.read_depth = read_depth
        self.kmer_dim = kmer_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.embed_dim = encoder_embed_dim
        self.fc1_hidden = 12
        self.cls_classifier = CLSClassifier(self.embed_dim, cls_output_classes)
        self.hap0_ref_classifier = CLSClassifier(self.embed_dim, 1)
        self.hap1_ref_classifier = CLSClassifier(self.embed_dim, 1)
        self.hap0_hap1_classifier = CLSClassifier(self.embed_dim, 1)

        self.fc1 = nn.Linear(feature_count, self.fc1_hidden)
        self.fc2 = nn.Linear(self.read_depth * self.fc1_hidden, self.embed_dim)

        self.converter = nn.Linear(self.embed_dim, self.decoder_embed_dim)
        self.pos_encoder = PositionalEncoding2D(self.fc1_hidden, self.device)
        self.tgt_pos_encoder = PositionalEncoding(self.kmer_dim, batch_first=True, max_len=500).to(self.device)
        logger.info(f"tgt pos encoder: {self.tgt_pos_encoder.pe.shape}, embed dim: {self.decoder_embed_dim}")
        self.encoder = TransformerEncoderStack(
            num_layers=n_encoder_layers, 
            embed_dim=encoder_embed_dim, 
            num_heads=encoder_attention_heads, 
            num_kv_heads=encoder_num_kv_heads, 
            dropout=0.1,
            ff_factor=encoder_ff_factor)
        

        self.decoder0 = TransformerDecoderStack(
                            num_layers=n_decoder_layers, 
                            embed_dim=decoder_embed_dim, 
                            num_heads=decoder_attention_heads, 
                            num_kv_heads=decoder_num_kv_heads, 
                            dropout=0.1, 
                            ff_factor=decoder_ff_factor)
        self.decoder1 = TransformerDecoderStack(
                            num_layers=n_decoder_layers, 
                            embed_dim=decoder_embed_dim, 
                            num_heads=decoder_attention_heads, 
                            num_kv_heads=decoder_num_kv_heads, 
                            dropout=0.1, 
                            ff_factor=decoder_ff_factor)

        self.tgt_input_converter = nn.Linear(self.kmer_dim, self.decoder_embed_dim)
        self.decode_output_converter0 = nn.Linear(self.decoder_embed_dim, self.kmer_dim)
        self.decode_output_converter1 = nn.Linear(self.decoder_embed_dim, self.kmer_dim)

        self.softmax = nn.LogSoftmax(dim=-1)
        self.emb_layernorm = nn.LayerNorm(self.embed_dim)
        self.emb_dropout = nn.Dropout(0.1)

    
    def encode(self, src):
        # Add CLS token to the beginning of the sequence, the dimensions of src are (batch_size, seq_len, read_depth, feature_count)
        src = torch.cat((self.cls_token.unsqueeze(0).repeat(src.shape[0], 1, src.shape[2], 1), src), dim=1)

        src = F.gelu(self.fc1(src)) # Operates on each "feature" (10 feature encoded base)
        src = self.pos_encoder(src)  # For 2D encoding we have to do this before flattening, right?
        src = src.flatten(start_dim=2)
        src = F.gelu(self.fc2(src)) # Operates on an entire alignment column
        src = self.emb_dropout(self.emb_layernorm(src))
        mem = self.encoder(src)

        cls_embed = mem[:, 0, :]
        cls_pred = self.cls_classifier(cls_embed)
        hap0_ref_pred = self.hap0_ref_classifier(cls_embed)
        hap1_ref_pred = self.hap1_ref_classifier(cls_embed)
        hap0_hap1_pred = self.hap0_hap1_classifier(cls_embed)
        return mem, cls_pred, hap0_ref_pred, hap1_ref_pred, hap0_hap1_pred

    def decode(self, mem, tgt, tgt_mask):
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
            #logger.info(f"Forcing tgt_mask shape to be {tgt_mask.shape}, input enc shape is: {mem.shape}")
        
        # Convert 2D causal mask to 3D boolean mask for torchtune self-attention
        # torchtune expects [batch, seq_len, seq_len] boolean mask where True means "can attend"
        # PyTorch's generate_square_subsequent_mask uses -inf for masked positions and 0 for unmasked
        batch_size = tgt0.shape[0]
        causal_mask = (tgt_mask == 0).unsqueeze(0).expand(batch_size, -1, -1)  # [batch, seq, seq]
        
        h0 = self.decoder0(tgt0, mem_proj, mask=causal_mask)
        h1 = self.decoder1(tgt1, mem_proj, mask=causal_mask)

        h0 = self.decode_output_converter0(h0)
        h1 = self.decode_output_converter1(h1)

        h0 = self.softmax(h0)
        h1 = self.softmax(h1)
        return torch.stack((h0, h1), dim=1)

    def forward(self, src, tgt, tgt_mask):
        mem, cls_pred, hap0_ref_pred, hap1_ref_pred, hap0_hap1_pred = self.encode(src)
        result = self.decode(mem, tgt.float(), tgt_mask)
        return result, cls_pred, hap0_ref_pred, hap1_ref_pred, hap0_hap1_pred


if __name__ == "__main__":
    batch_size = 5
    seq_len = 13  # source sequence length
    tgt_seq_len = 10  # target sequence length
    model = NewVarTransformer(
                read_depth=96, 
                feature_count=10, 
                encoder_embed_dim=256, 
                encoder_attention_heads=8, 
                encoder_num_kv_heads=4, 
                encoder_ff_factor=2, 
                decoder_embed_dim=256, 
                decoder_attention_heads=8, 
                decoder_num_kv_heads=4, 
                decoder_ff_factor=2, 
                kmer_dim=260, 
                n_encoder_layers=6, 
                n_decoder_layers=6, cls_output_classes=1, device='cpu')
    src = torch.randn(batch_size, seq_len, 96, 10)
    tgt = torch.randn(batch_size, 2, tgt_seq_len, 260)
    # tgt_mask is a causal mask for self-attention in the decoder
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len)
    result, cls_pred, hap0_ref_pred, hap1_ref_pred, hap0_hap1_pred = model(src, tgt, tgt_mask)
    print(result.shape)
    print(result[0, 0, :, :])
    print(result[0, 1, :, :])
    print(result[0, 0, 5, :])