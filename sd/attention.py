import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, num_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.in_prokj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_prokj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.num_heads = num_heads
        self.d_head = d_embed // num_heads
    
    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # X : (Batch_size, Seq_len, d_embed) aka (batch, width*height, features)
        input_shape = x.shape

        batch_size, seq_len, d_embed = x.shape

        intermediate_shape = (batch_size, seq_len, self.num_heads, self.d_head)

        # (Batch_size, Seq_len, d_embed) -> (Batch_size, Seq_len, d_embed * 3) == 3 tensors of shape (Batch_size, Seq_len, d_embed)
        q, k, v = self.in_prokj(x).chunk(3, dim=-1)

        # Reshape for multi-head attention

        # (Batch_size, Seq_len, d_embed) -> (Batch_size, Seq_len, num_heads, d_head) -> (Batch_size, num_heads, Seq_len, d_head)
        q = q.view(*intermediate_shape).transpose(1, 2)  
        k = k.view(*intermediate_shape).transpose(1, 2)
        v = v.view(*intermediate_shape).transpose(1, 2)

        # compute attention scores
        # (Batch_size, num_heads, Seq_len, Seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # mask future positions (upper triangular part of the weight matrix)
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight = weight.masked_fill(mask, -torch.inf)

        weight = weight / math.sqrt(self.d_head)
        weight = torch.softmax(weight, dim=-1)

        # compute attention output
        # (Batch_size, num_heads, Seq_len, Seq_len) @ (Batch_size, num_heads, Seq_len, (d_embed/num_heads) d_head)
        output = weight @ v  # 

        # transpose back from (Batch_size, , Seq_len, d_head) -> to (Batch_size, Seq_len, num_heads, d_head)
        output = output.transpose(1, 2)

        # reshape back to (Batch_size, Seq_len, d_embed)
        output = output.reshape(input_shape)

        # (Batch_size, Seq_len, d_embed)
        return self.out_prokj(output)
