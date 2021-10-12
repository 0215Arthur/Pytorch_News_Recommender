

import torch
import torch.nn as nn
import numpy as np
import torchsnooper
import torch.nn.functional as F
import math

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, head_nums,mask=None, dropout=None):
        # 
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        
        # 设置掩码
       # print(query.size(),scores.size())

        if mask is not None:
            #print(scores[0])
            _mask=(mask.unsqueeze(1)*mask.unsqueeze(2)).unsqueeze(1)
            #print(_mask.size())
            _mask=torch.cat([_mask for _ in range(head_nums)],1)
            scores = scores.masked_fill(_mask == 0, -1e9)
            #print(scores[0])

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadSelfAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)
  #  @torchsnooper.snoop()
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, attn = self.attention(query, key, value, head_nums=self.h, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
  
class AdditiveAttention(torch.nn.Module):
    def __init__(self, query_vector_dim, input_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(input_vector_dim, query_vector_dim)

        ## change: uniform_(-1,1)->uniform_(-0.1,0.1)


        self.query_vector = nn.Parameter(torch.empty(query_vector_dim).uniform_(-0.1, 0.1)) 

    def forward(self, input,mask=None):
        '''
        config:
            input: batch_size, n_input_vector, input_vector_dim
        Returns:
            result: batch_size, input_vector_dim
        '''
        # batch_size, n_input_vector, query_vector_dim -> 512*25*800= > 512*25*512
        tmp = torch.tanh(self.linear(input))
        scores=torch.matmul(tmp, self.query_vector)
       # print('Add_Attn: scores',scores.size())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # batch_size, n_input_vector
        weight = F.softmax(scores, dim=1)
        result = torch.bmm(weight.unsqueeze(dim=1), input).squeeze(dim=1)
        return result

 