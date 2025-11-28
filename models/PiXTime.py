import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np

from torch import distributed

import layers.Transformer_EncDec as Transformer_EncDec


class Patch_EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len) #[b_s, n_vars, patch_num, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) #[b_s * n_vars, patch_num, patch_len]
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) #[b_s * n_vars, patch_num, d_model]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1])) #[b_s, n_vars, patch_num, d_model]
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class iTrans_ExEmbedding(nn.Module):
    def __init__(self, 
                 seq_len = 96,
                 n_vars = 7, #变量数，用于生成位置编码
                 d_model = 128, #指定
                 dropout = 0.1,
                 factor = 3,
                 n_heads = 8,
                 d_ff = 128,
                 e_layers = 1,
                 activation = 'gelu'
                 ):
        super().__init__()
        self.seq_len = seq_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(c_in = seq_len, d_model = d_model, dropout = dropout)
        self.pos_embedding = nn.Embedding(n_vars, d_model)
        # Encoder
        self.encoder = Transformer_EncDec.Encoder(
            [
                Transformer_EncDec.EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forward(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None) #[bs, seq_len, n_var] -> [bs, n_var, d_model]
        enc_out = enc_out + self.pos_embedding(torch.arange(enc_out.shape[1]).cuda()).unsqueeze(0)
        enc_out, attns = self.encoder(enc_out)

        return enc_out


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x[:,:,:,:-1]) #去除最后一个glo token，以实现patch-wise的映射
        x = self.linear(x)
        x = self.dropout(x)
        return x


class XDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class XDecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):
    def __init__(self,
                 task_name = 'long_term_forecast',
                 features = 'M',
                 seq_len = 96, #指定
                 pred_len = 96, #指定
                 use_norm = 1,
                 patch_len = 48, #指定
                 n_vars = 7, #所有变量数量
                 d_model = 128, #指定
                 dropout = 0.1,
                 factor = 3,
                 n_heads = 2, #每头的维度一般为64
                 en_d_ff = 512, #一般是d_model的4倍
                 de_d_ff = 512, #同上
                 en_layers = 1, #指定，测试后认为1层比2层性能好，主要是语句太短了
                 de_layers = 2, #指定
                 activation = 'gelu'
                 ):
        super().__init__()
        self.task_name = task_name
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        self.patch_len = patch_len

        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})")
        self.patch_num = seq_len // patch_len

        self.n_vars = 1 if features == 'MS' else n_vars

        # embedding
        self.patch_embedding = Patch_EnEmbedding(self.n_vars, d_model, self.patch_len, dropout)

        # encoder for ex_var
        self.ex_embedding = iTrans_ExEmbedding(seq_len, n_vars, d_model, dropout, factor, n_heads, en_d_ff, en_layers)

        # decoder
        self.cross_decoder = XDecoder(
            [
                XDecoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    d_model,
                    de_d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(de_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.head_nf = d_model * (self.patch_num)
        self.head = FlattenHead(self.head_nf, pred_len, head_dropout=dropout)


    def forecast(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        patch_embed, n_vars = self.patch_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1)) #x_enc: [b_s, 1, length]
        ex_embed = self.ex_embedding(x_enc[:, :, :-1])

        out = self.cross_decoder(patch_embed, ex_embed)
        out = torch.reshape(out, (-1, n_vars, out.shape[-2], out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        out = out.permute(0, 1, 3, 2)

        out = self.head(out)  # z: [bs x nvars x target_window]
        out = out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            out = out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            out = out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return out


    def forecast_multi(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        patch_embed, n_vars = self.patch_embedding(x_enc.permute(0, 2, 1)) #x_enc: [b_s, n_vars, length]
        ex_embed = self.ex_embedding(x_enc)

        out = self.cross_decoder(patch_embed, ex_embed)
        out = torch.reshape(out, (-1, n_vars, out.shape[-2], out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        out = out.permute(0, 1, 3, 2)

        out = self.head(out)  # z: [bs x nvars x target_window]
        out = out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            out = out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            out = out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.features == 'M':
            dec_out = self.forecast_multi(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]