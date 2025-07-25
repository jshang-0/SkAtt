
import torch
import torch.nn as nn
import math
import json
from torch.utils.checkpoint import checkpoint


def attn_selector(attn_type, config, W_q=None, W_k=None, W_v=None):


    if attn_type.startswith("softmax"):
        attn = SoftmaxAttention(config)
    elif attn_type.startswith("exp_kernel"):
        attn = ExpKernelAttention(config)
    elif attn_type.startswith("sketch_exp_kernel"):
        from models.attention_sketch_exp_kernel import SkExpKernelAttention
        attn = SkExpKernelAttention(config)
    elif attn_type.startswith("rsketch_exp_kernel"):
        from models.attention_sketch_exp_kernelr import SkExpKernelAttentionResid
        attn = SkExpKernelAttentionResid(config)

    elif attn_type.startswith("linformer"):
        from models.attention_linformer import LinformerAttention
        attn = LinformerAttention(config)
    elif attn_type.startswith("informer"):
        from models.attention_informer import ProbAttention
        attn = ProbAttention(config)
    elif attn_type.startswith("nystrom"):
        from models.attention_nystrom import NystromAttention
        attn = NystromAttention(config)
    elif attn_type.startswith("performer"):
        from models.attention_performer import PerformerAttention
        attn = PerformerAttention(config)
    elif attn_type.startswith("bigbird"):
        from models.attention_bigbird import BigBirdAttention
        attn = BigBirdAttention(config)

    return attn


class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V, mask):
        # input [batch_size, nb_heads, seq_len, dim_head]
        # print('Q', Q.abs().median()) # check scale
        # print('K', K.abs().median())
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)

        # output [batch_size, nb_heads, seq_len, dim_head]
        return X

class ExpKernelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V, mask):
        # input [batch_size, nb_heads, seq_len, dim_head]
        # print('Q', Q.abs().median()) # check scale
        # print('K', K.abs().median())
        Q = Q * mask[:, None, :, None]
        K = K * mask[:, None, :, None]
        V = V * mask[:, None, :, None]

        Q=nn.functional.softmax(Q,dim=-1)
        K=nn.functional.softmax(K,dim=-2)
        dot = torch.matmul(torch.transpose(K, -2, -1),V)
        attn=torch.matmul(Q,dot)

        attn = self.drop_attn(attn)

        # output [batch_size, nb_heads, seq_len, dim_head]
        return attn


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()


        self.dim = config["transformer_dim"] # input_dim
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.attn_type = config["attn_type"]

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.attn = attn_selector(self.attn_type, config, self.W_q, self.W_k, self.W_v)

        self.grad_checkpointing = (self.attn_type == "softmax")

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask):

        if self.attn_type.startswith("longformer") or self.attn_type.startswith("reformer"):
            with torch.cuda.amp.autocast(enabled = False):
                attn_out = self.attn(X.float(), mask.float())
        else:
            Q = self.split_heads(self.W_q(X))
            K = self.split_heads(self.W_k(X))
            V = self.split_heads(self.W_v(X))
            with torch.cuda.amp.autocast(enabled = False):
                if self.grad_checkpointing:
                    attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float())
                else:
                    attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
            attn_out = self.combine_heads(attn_out)
        out = self.ff(attn_out)

        return out


    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

