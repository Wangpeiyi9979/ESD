"""
@Time: 2021/6/24 15:42
@Author: Wang Peiyi
@Site : 
@File : utils.py
"""
import sys
import util
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import BertModel

class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta
    def forward(self, x):
        return x * torch.sigmoid_(self.beta*x)

def sequence_mask(sequence_length): 
    """
    """
    max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.tensor(range(0, max_len)).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand.to(sequence_length.device)
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand > seq_length_expand

def swish(x):
    return x * torch.sigmoid(x)

def fast_att(Q, V, activation='swish'):
    """
    :param Q: Q_num x d
    :param V: V_num x d
    :return: Q_num x d
    """
    Q_num, _ = Q.size()
    if len(V.size()) == 2:
        V_num, _ = V.size()
        V_expand =  V.unsqueeze(0).expand(Q_num, -1, -1)
    else:
        V_expand = V
        _, V_num, _ = V.size()
    Q_expand = Q.unsqueeze(1).expand(-1, V_num, -1)

    if activation == 'swish':
        att_score = swish((Q_expand * V_expand)).sum(-1).softmax(dim=-1) # Q_num x V_num
    else:
        att_score = (Q_expand * V_expand).tanh().sum(-1).softmax(dim=-1) # Q_num x V_num
    O = torch.matmul(att_score.unsqueeze(1), V_expand).squeeze(1)        # Q_num x d
    return O


class MultiHeadedAttentionWithFNN(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, activation='siwsh'):
        super().__init__()
        self.torch_mul_attentioner = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm0 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            Swish() if activation == 'swish' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
    def forward(self, q, k, v, key_padding_mask=None):
        q = q.permute(1, 0, 2)
        k = q.permute(1, 0, 2)
        v = q.permute(1, 0, 2)
        out_res = q
        out, _ = self.torch_mul_attentioner(q, k, v, key_padding_mask=key_padding_mask)
        out = out + out_res
        out = self.norm0(out)
        out_res = out
        out = self.ffn(out)
        out = self.dropout_layer(out)
        out = out + out_res
        out = self.norm1(out)
        return out.permute(1, 0, 2).contiguous()
