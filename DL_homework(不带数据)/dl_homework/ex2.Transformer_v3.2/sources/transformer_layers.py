import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs (cross-attention)
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key_MLP = nn.Linear(embed_dim, embed_dim)
        self.query_MLP = nn.Linear(embed_dim, embed_dim)
        self.value_MLP = nn.Linear(embed_dim, embed_dim)
        self.proj_MLP = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop_layer = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE *****
        # 1. Linear projections
        Q = self.query_MLP(query)  # (N, S, E)
        K = self.key_MLP(key)      # (N, T, E)
        V = self.value_MLP(value)  # (N, T, E)
        
        # 2. Reshape for multi-head: (N, S/T, H, E//H) -> (N, H, S/T, E//H)
        N, S, E = Q.shape
        H = self.n_head
        head_dim = self.head_dim
        
        Q = Q.view(N, S, H, head_dim).transpose(1, 2)  # (N, H, S, head_dim)
        K = K.view(N, T, H, head_dim).transpose(1, 2)  # (N, H, T, head_dim)
        V = V.view(N, T, H, head_dim).transpose(1, 2)  # (N, H, T, head_dim)
        
        # 3. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)  # (N, H, S, T)
        
        # 4. Masking
        if attn_mask is not None:
            # attn_mask: (S, T), broadcast to (N, H, S, T)
            scores = scores.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # 5. Softmax
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop_layer(attn)
        
        # 6. Weighted sum
        out = torch.matmul(attn, V)  # (N, H, S, head_dim)
        
        # 7. Concat heads
        out = out.transpose(1, 2).contiguous().view(N, S, E)  # (N, S, E)
        
        # 8. Final linear projection
        output = self.proj_MLP(out)
        self.attn_weights = attn.detach()
        # *****END OF YOUR CODE *****
        return output


class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array (pe) as described in       #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting.                                #
        ############################################################################
        # *****START OF YOUR CODE *****
        position = torch.arange(0, max_len).unsqueeze(1)     # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        # (embed_dim // 2)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # *****END OF YOUR CODE *****

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness). And it won't be updated automatically.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE *****
        # x: (N, S, D)
        pe_slice = self.pe[:, :S, :]  # (1, S, D)
        output = x + pe_slice
        output = self.dropout(output)
        # *****END OF YOUR CODE *****
        return output
