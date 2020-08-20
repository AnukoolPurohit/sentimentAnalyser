import torch
import torch.nn.functional as F
from torch import nn


class WordSentenceAttention(nn.Module):
    def __init__(self, hidden_sz):
        super().__init__()
        
        self.context_weight     = nn.Parameter(torch.Tensor(hidden_sz).uniform_(-0.1,0.1))
        self.context_projection = nn.Linear(hidden_sz, hidden_sz)
        return
    
    def forward(self, context):
        context_proj = torch.tanh(self.context_projection(context))
        αt = context_proj.matmul(self.context_weight)
        attn_score = F.softmax(αt, dim=1).unsqueeze(2)
        sentence = context.transpose(1,2).bmm(attn_score)
        return sentence.squeeze(2)