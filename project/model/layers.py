

import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.linear import Linear
import torchsnooper
import torch.nn.functional as F
import math

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, head_nums,mask=None, dropout=None, topic_query=None, topic_key=None):
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
            if topic_query is not None:
                topic_scores = torch.matmul(topic_query, topic_key.transpose(-2, -1)) / math.sqrt(topic_query.size(-1))
                _mask=mask.unsqueeze(1)*mask.unsqueeze(2)
                # print(_mask.size())
                # for row in topic_scores.detach().cpu().numpy():
                #     for _ in row:
                #         print(_, end='')
                #     print() 
                # print("topic_scores: ",topic_scores)
                topic_scores = topic_scores.masked_fill(_mask == 0, 1).unsqueeze(1)
                topic_scores = F.softmax(topic_scores, dim=-1)
                scores = scores * topic_scores
                # scores /= topic_query.size(-1)
                # print(scores)
                # topic_scores
            #print(scores[0])

        p_attn = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadSelfAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout, topic_embed=100):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        # self.topic_layers = nn.ModuleList([nn.Linear(topic_embed, topic_embed) for _ in range(2)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)
    
    def initialize(self):
        for parameter in self.linear_layers.parameters():
            if len(parameter.size()) >= 2:
                nn.init.xavier_uniform_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)


  #  @torchsnooper.snoop()
    def forward(self, query, key, value, mask=None, topic=None, save_attn=False):
        batch_size = query.size(0)
        topic_query=None
        topic_key=None
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # if topic is not None:
        #     topic_query, topic_key = [l(x) for l, x in zip(self.topic_layers, (topic, topic))]
        # 2) Apply attention on all the projected vectors in batch. 
        x, attn = self.attention(query, key, value, head_nums=self.h, mask=mask, dropout=self.dropout)
        # if save_attn:
        #     torch.save
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn
  
class AdditiveAttention(torch.nn.Module):
    def __init__(self, query_vector_dim, input_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(input_vector_dim, query_vector_dim)

        ## change: uniform_(-1,1)->uniform_(-0.1,0.1)
        self.query_vector = nn.Parameter(torch.empty(query_vector_dim).uniform_(-0.1, 0.1)) 
    
    def initialize(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.linear.bias)
    

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

class CNEAttention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super(CNEAttention, self).__init__()
        self.affine1 = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=True)
        self.affine2 = nn.Linear(in_features=attention_dim, out_features=1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    # Input
    # feature : [batch_size, length, feature_dim]
    # mask    : [batch_size, length]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, mask=None):
        attention = torch.tanh(self.affine1(feature))                                 # [batch_size, length, attention_dim]
        a = self.affine2(attention).squeeze(dim=2)                                    # [batch_size, length]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1) # [batch_size, 1, length]
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)                              # [batch_size, 1, length]
        out = torch.bmm(alpha, feature).squeeze(dim=1)                                # [batch_size, feature_dim]
        return out


class ScaledDotProduct_CandidateAttention(nn.Module):
    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super(ScaledDotProduct_CandidateAttention, self).__init__()
        self.K = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=False)
        self.Q = nn.Linear(in_features=query_dim, out_features=attention_dim, bias=True)
        self.attention_scalar = math.sqrt(float(attention_dim))

    def initialize(self):
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, query, mask=None):
        a = torch.bmm(self.K(feature), self.Q(query).unsqueeze(dim=2)).squeeze(dim=2) / self.attention_scalar # [batch_size, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1)                                          # [batch_size, feature_num]
        else:
            alpha = F.softmax(a, dim=1)                                                                       # [batch_size, feature_num]
        out = torch.bmm(alpha.unsqueeze(dim=1), feature).squeeze(dim=1)                                       # [batch_size, feature_dim]
        return out
