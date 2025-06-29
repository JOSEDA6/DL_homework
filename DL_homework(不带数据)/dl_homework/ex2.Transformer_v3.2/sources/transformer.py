import numpy as np
import copy

import torch
import torch.nn as nn

from .transformer_layers import *


class CaptioningTransformer(nn.Module):
    """
    A CaptioningTransformer produces captions from image features using a
    Transformer decoder.

    The Transformer receives input vectors of size D, has a vocab size of V,
    works on sequences of length T, uses word vectors of dimension W, and
    operates on minibatches of size N.
    """
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_blocks=2, max_length=50):
        """
        Construct a new CaptioningTransformer instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - num_heads: Number of attention heads.
        - num_blocks: Number of transformer blocks.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        self.visual_projection_layer = nn.Linear(input_dim, wordvec_dim)
        self.embedding_layer = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding_layer = PositionalEncoding(wordvec_dim, max_len=max_length)

        decoderblock_layer = TransformerDecoderBlock(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformerdecoder = TransformerDecoder(decoderblock_layer, num_blocks=num_blocks)
        self.apply(self._init_weights)

        self.output_layer = nn.Linear(wordvec_dim, vocab_size)

    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.

        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)

        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        N, T = captions.shape

        # memory is for transformerdecoder
        memory = self.visual_projection_layer(features)
        memory = memory.unsqueeze(1) # [N,D] -> [N, 1, D]

        # Create a placeholder, to be overwritten by your code below.
        scores = torch.empty((N, T, self.vocab_size))
        ############################################################################
        # TODO: Implement the forward function for CaptionTransformer.             #
        # A few hints:                                                             #
        #  1) You first have to embed your caption and add positional              #
        #     encoding. You then have to project the image features into the same  #
        #     dimensions.                                                          #
        #  2) You have to prepare a mask (tgt_mask) for masking out the future     #
        #     timesteps in captions. torch.tril() function might help in preparing #
        #     this mask.                                                           #
        #  3) Finally, apply the decoder features on the text & image embeddings   #
        #     along with the tgt_mask. Project the output to scores per token      #
        ############################################################################
        # CaptioningTransformer.forward
        # *****START OF YOUR CODE *****
        if not isinstance(captions, torch.Tensor):
            captions = torch.tensor(captions, dtype=torch.long, device=features.device)
        N, T = captions.shape
        
        # memory is for transformerdecoder
        memory = self.visual_projection_layer(features)
        memory = memory.unsqueeze(1)  # [N,D] -> [N, 1, D]
        
        # 1. Embed captions and add positional encoding
        embedded = self.embedding_layer(captions)  # (N, T, W)
        embedded = self.positional_encoding_layer(embedded)  # (N, T, W)
        
        # 2. Prepare tgt_mask (T, T)
        device = captions.device
        tgt_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device))
        
        # 3. Pass through the decoder
        out = self.transformerdecoder(embedded, memory, tgt_mask=tgt_mask)
        
        # 4. Project to vocabulary space
        scores = self.output_layer(out)  # (N, T, V)
        # *****END OF YOUR CODE *****

        return scores

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.

        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length

        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            word = None

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                if word is not None: 
                    partial_caption2 = word
                    output_logits2 = self.forward(features, partial_caption2)
                    output_logits2 = output_logits2[:, -1, :]
                    dif = output_logits - output_logits2


                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions


class TransformerDecoderBlock(nn.Module):
    """
    A single block of a Transformer decoder, to be used with TransformerDecoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerDecoderBlock instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.multihead_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer.

        Inputs:
        - tgt: the sequence to the decoder layer, of shape (N, T, W)
        - memory: the sequence from the last layer of the encoder, of shape (N, S, D)
        - tgt_mask: the parts of the target sequence to mask, of shape (T, T)

        Returns:
        - out: the Transformer features, of shape (N, T, W)
        """
        ############################################################################
        # TODO: Perform TransformerDecoderBlock forward pass.                      #
        # Be aware that the original paper states that "We apply dropout to        #
        # the output of each sub-layer, before it is added to the sub-layer input  #
        # and normalized." So don't forget to add dropout, residual connection and # 
        # layer norm for each sub-layer.                                           #
        #  1) Perform self-attention on the target sequence (along with dropout,   #
        #     residual connection and layer norm).                                 #
        #  2) Perform cross attention, which attends to both the target sequence   #
        #     and the sequence from the last encoder layer (along with dropout,    #
        #     residual connection and layer norm).                                 #
        #  3) Perform feedforward MLP layer. The MLP architecture is:              #
        #       linear1 -> activation(Relu) -> dropout -> linear2                  #
        #     (along with dropout, residual connection and layer norm).            #
        ############################################################################
        # *****START OF YOUR CODE *****
        # 1. Self-attention with mask
        x = self.norm1(tgt + self.dropout1(self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)))
        # 2. Cross-attention with memory
        x = self.norm2(x + self.dropout2(self.multihead_attn(x, memory, memory)))
        # 3. Feedforward MLP
        mlp = self.linear2(self.dropout(self.activation(self.linear1(x))))
        out = self.norm3(x + self.dropout3(mlp))
        return out
        # *****END OF YOUR CODE *****
        return tgt

def clones(module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerDecoder(nn.Module):
    def __init__(self, decoderblock_layer, num_blocks):
        super().__init__()
        self.layers = clones(decoderblock_layer, num_blocks)
        self.num_blocks = num_blocks

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)

        return output
