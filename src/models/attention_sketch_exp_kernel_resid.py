
import torch
import torch.nn as nn
import math

class SkExpKernelAttentionResid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]
        self.sampleSize_q=config["sampleSize_q"]
        self.sampleSize_k=config["sampleSize_k"]
        self.device = config['device'] if 'device' in config else 'cuda'

    def forward(self, Q, K, V, mask):
        # input [batch_size, nb_heads, seq_len, dim_head]
        # print('Q', Q.abs().median()) # check scale
        # print('K', K.abs().median())
        attn = torch.zeros_like(V).to(self.device)

        seq_len = Q.shape[-2]
        sample_idx_q=torch.randint(low=0, high=seq_len, size=(round(self.sampleSize_q*seq_len),))
        sample_idx_q,_=torch.sort(sample_idx_q)
        sample_idx_q=sample_idx_q.to(self.device)
        Q = torch.index_select(Q,-2,sample_idx_q)

        sample_idx_k = torch.randint(low=0, high=seq_len, size=(round(self.sampleSize_k*seq_len),))
        # sample_idx_k,_ = torch.sort(sample_idx_k)
        sample_idx_k=sample_idx_k.to(self.device)
        K = torch.index_select(K, -2, sample_idx_k)
        V = torch.index_select(V, -2, sample_idx_k)

        Q = nn.functional.softmax(Q, dim=-1)
        K = nn.functional.softmax(Q, dim=-2)

        dot = torch.matmul(torch.transpose(K, -2, -1),V)
        attn_compressed=torch.matmul(Q,dot)
        attn_compressed=(seq_len**2/round(self.sampleSize_q*seq_len)/round(self.sampleSize_k*seq_len))*attn_compressed
        attn_compressed += (seq_len/round(self.sampleSize_k*seq_len))*V

        sample_idx_q_uni=torch.unique(sample_idx_q).to(self.device)
        sample_idx_q_uni_cor=torch.zeros_like(sample_idx_q_uni).to(self.device)
        j_uni_cor=0
        for i in range(len(sample_idx_q)-1):
            if sample_idx_q[i+1]==sample_idx_q[i]:
               attn_compressed[:,:,i+1,:]+=attn_compressed[:,:,i,:]
            else:
               sample_idx_q_uni_cor[j_uni_cor] = i
               j_uni_cor += 1
        sample_idx_q_uni_cor[-1]=len(sample_idx_q)-1

        attn[:,:,sample_idx_q_uni,:]=attn_compressed[:,:,sample_idx_q_uni_cor,:]

        attn = self.drop_attn(attn)
        # print(f"seq_len is {seq_len} and Q sketching size is {len(sample_idx_q)} K sketching size is {len(sample_idx_q)}")

        # output [batch_size, nb_heads, seq_len, dim_head]
        return attn
