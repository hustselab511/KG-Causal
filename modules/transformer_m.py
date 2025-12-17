import copy
import math
import os

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import tensor_utils
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel


### transformer 组件

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# def attention(query, key, value, mask=None, dropout=None):
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     p_attn = F.softmax(scores, dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn


def attention(query, key, value, mask=None, dropout=None, return_counterfactual=False, k=10):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    # 保存原始注意力分布用于反事实生成
    original_p_attn = p_attn.clone()

    # 应用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 计算输出
    output = torch.matmul(p_attn, value)

    # 生成反事实注意力和输出
    if return_counterfactual:
        batch_size, num_heads, seq_len_q, seq_len_k = original_p_attn.shape
        cf_attn = original_p_attn.clone()

        # 对每个样本和头遮盖Top-K高注意力区域
        for b in range(batch_size):
            for h in range(num_heads):
                attn_dist = cf_attn[b, h]
                topk_values, topk_indices = torch.topk(attn_dist, k=k, dim=-1)
                # 创建掩码并应用
                mask = torch.ones_like(attn_dist)
                mask.scatter_(-1, topk_indices, 0)
                cf_attn[b, h] = attn_dist * mask

        # 应用dropout到反事实注意力
        if dropout is not None:
            cf_attn = dropout(cf_attn)

        cf_output = torch.matmul(cf_attn, value)
        return output, p_attn, cf_output, cf_attn

    return output, p_attn




class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class FBEmbeddings(nn.Module):
    def __init__(self, d_model, vocab,tokenizer_m=None):
        super(FBEmbeddings, self).__init__()
        self.d_model = d_model
        self.tokenizer = tokenizer_m

        self.embed_dim = 768

        max_idx = max(self.tokenizer.idx2embed.keys())
        self.embedding_matrix = torch.zeros((max_idx + 1, self.embed_dim))
        for idx, embed in self.tokenizer.idx2embed.items():
            self.embedding_matrix[idx] = torch.tensor(embed)

        self.embedding = nn.Embedding.from_pretrained(
            self.embedding_matrix,
            freeze=True,
            # freeze=False,
            padding_idx=0  # 假设0是填充标记
        )

        self.pro = nn.Linear(self.embed_dim, d_model)

    def forward(self, x):
        embed = self.embedding(x.long())   # * math.sqrt(self.embed_dim)
        embed = self.pro(embed)
        return embed

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.pos_embed = None

    def forward(self, x):
        self.pos_embed = self.pe[:, :x.size(1)]
        x = x + self.pos_embed
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1,k_n=20):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.counterfactual_attn  = None
        self.dropout = nn.Dropout(p=dropout)
        self.k_n = k_n

    def forward(self, query, key, value, mask=None,return_counterfactual=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        if return_counterfactual:
            x, self.attn, x_cf, self.counterfactual_attn = attention(
                query, key, value, mask=mask, dropout=self.dropout,
                return_counterfactual=True, k=self.k_n
            )

            x_cf = x_cf.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
            x_cf = self.linears[-1](x_cf)
        else:
            x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = self.linears[-1](x)

        return (x, x_cf) if return_counterfactual else x


class SublayerConnection(nn.Module):
    def __init__(self, embed_dim, dropout,cf = False):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(embed_dim)
        if cf :
            self.cf_norm = LayerNorm(embed_dim)  # 反事实特征专用归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        sublayer_output = sublayer(self.norm(x))
        if isinstance(sublayer_output, tuple):
            # 对反事实特征应用独立归一化
            fact_output, cf_output = sublayer_output
            cf_output = self.cf_norm(cf_output)
            return x + self.dropout(fact_output), x + self.dropout(cf_output)
        return x + self.dropout(sublayer_output)


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadedAttention(num_heads, embed_dim, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.sublayer_attn = SublayerConnection(embed_dim, dropout)
        self.sublayer_ff = SublayerConnection(embed_dim, dropout)

    def forward(self, x, mask=None,k=None,v=None):
        if k is None:
            x = self.sublayer_attn(x, lambda x: self.attn(x, x, x, mask))
            x = self.sublayer_ff(x, self.feed_forward)
            return x
        else:
            x = self.sublayer_attn(x, lambda x: self.attn(x, k, v, mask))
            x = self.sublayer_ff(x, self.feed_forward)
            return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                     for _ in range(num_layer)])
        self.norm = LayerNorm(embed_dim)

    def forward(self, h, mask=None,k=None,v=None):
        for layer in self.layers:
            h = layer(h, mask,k,v)
        h = self.norm(h)
        return h


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(num_heads, embed_dim, dropout)
        self.cross_attn = MultiHeadedAttention(num_heads, embed_dim, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.sublayer_cross = SublayerConnection(embed_dim, dropout)
        self.sublayer_self = SublayerConnection(embed_dim, dropout)
        self.sublayer_ff = SublayerConnection(embed_dim, dropout)

    def forward(self, x, h, self_mask=None, cross_mask=None):
        x = self.sublayer_self(x, lambda x: self.self_attn(x, x, x, self_mask))
        x = self.sublayer_cross(x, lambda x: self.cross_attn(x, h, h, cross_mask))
        x = self.sublayer_ff(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                     for _ in range(num_layer)])
        self.norm = LayerNorm(embed_dim)

    def forward(self, x, h, self_mask=None, cross_mask=None):
        for i in range(len(self.layers)):
            x = self.layers[i](x, h, self_mask, cross_mask)
        x = self.norm(x)
        return x


def get_hv_mask(hv):
    v_masks = hv.new_ones(hv.shape[:2], dtype=torch.long)
    v_masks = v_masks.unsqueeze(-2)
    return v_masks

def get_ht_mask(seq=None):
    if seq is not None:
        # crop the last one
        seq = seq[:, :-1]
        seq_mask = (seq.data > 0)
        seq_mask[:, 0] += True
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & tensor_utils.subsequent_mask(seq.size(-1)).to(seq_mask)
    else:
        seq_mask = None
    return seq_mask, seq


def get_txt_mask(seq):
    seq = seq[:, :-1]
    seq_mask = (seq.data > 0)
    seq_mask[:, 0] += True
    seq_mask = seq_mask.unsqueeze(-1)
    return seq_mask

def get_cross_mask(hf_mask, txt_mask):
    h_mask = hf_mask

    cross_mask = txt_mask & h_mask.bool()
    return cross_mask


def get_hf_mask(hf):
    f_masks = hf.new_ones(hf.shape[:2], dtype=torch.long)
    f_masks = f_masks.unsqueeze(-2)
    return f_masks
