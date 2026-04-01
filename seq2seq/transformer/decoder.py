import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention, FeedForwardNN
from .encoder import PositionalEncoding
from seq2seq.data.fr_en import tokenizer


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float = 0.1,
    ):
        """
        Each decoder layer will take in two embeddings of
        shape (B, T, C):

        1. The `target` embedding, which comes from the decoder
        2. The `source` embedding, which comes from the encoder

        and will output a representation
        of the same shape.

        The decoder layer will have three main components:
            1. A Masked Multi-Head Attention layer (you'll need to
               modify the MultiHeadAttention layer to handle this!)
            2. A Multi-Head Attention layer for cross-attention
               between the target and source embeddings.
            3. A Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding(s)!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        self.masked_mha = MultiHeadAttention(self.num_heads, self.embedding_dim, self.qk_length, self.value_length)
        self.mha = MultiHeadAttention(self.num_heads, self.embedding_dim, self.qk_length, self.value_length)
        self.ffnn = FeedForwardNN(self.embedding_dim, self.ffn_hidden_dim)

        self.weight_q = nn.Linear(self.embedding_dim, self.num_heads * self.qk_length)
        self.weight_k = nn.Linear(self.embedding_dim, self.num_heads * self.qk_length)
        self.weight_v = nn.Linear(self.embedding_dim, self.num_heads * self.value_length)

        self.weight_q_new = nn.Linear(self.embedding_dim, self.num_heads * self.qk_length)
        self.weight_k_enc = nn.Linear(self.embedding_dim, self.num_heads * self.qk_length)
        self.weight_v_enc = nn.Linear(self.embedding_dim, self.num_heads * self.value_length)

        self.ln1 = nn.LayerNorm(normalized_shape = self.embedding_dim)
        self.ln2 = nn.LayerNorm(normalized_shape = self.embedding_dim)
        self.ln3 = nn.LayerNorm(normalized_shape = self.embedding_dim)

        #raise NotImplementedError("Need to implement DecoderLayer layers")


    def forward(
        self,
        x: torch.Tensor,
        enc_x: torch.Tensor | None,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The forward pass of the DecoderLayer.
        """

        Q = self.weight_q(x)
        K = self.weight_k(x)
        V = self.weight_v(x)

        masked_mha_output = self.masked_mha(Q, K, V, tgt_mask)

        ln1_output = self.ln1(masked_mha_output + x)

        if enc_x is not None:

            Q_new = self.weight_q_new(ln1_output)
            K_enc = self.weight_k_enc(enc_x)
            V_enc = self.weight_v_enc(enc_x)

            mha_output = self.mha(Q_new, K_enc, V_enc, src_mask)

            ln2_output = self.ln2(mha_output + ln1_output)

            ffnn_output = self.ffnn(ln2_output)

            ln3_output = self.ln3(ffnn_output + ln2_output)
        
        else:

            ffnn_output = self.ffnn(ln1_output)

            ln3_output = self.ln3(ffnn_output + ln1_output)

        return ln3_output

        #raise NotImplementedError("Need to implement DecoderLayer forward pass.")


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        max_length: int,
        dropout: float = 0.1,
    ):
        """
        Remember that the decoder will take in a sequence
        of tokens AND a source embedding
        and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Decoder layers.
        For this, we need to specify the number of layers
        and the number of heads.

        Additionally, for every Multi-Head Attention layer, we
        need to know how long each query/key is, and how long
        each value is.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length

        self.dropout = dropout
        self.max_length = max_length

        # Define any layers you'll need in the forward pass
        # Hint: You may find `ModuleList`s useful for creating
        # multiple layers in some kind of list comprehension.
        #
        # Recall that the input is just a sequence of tokens,
        # so we'll have to first create some kind of embedding
        # and then use the other layers we've implemented to
        # build out the Transformer decoder.
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.pos_enc = PositionalEncoding(self.embedding_dim, self.dropout, self.max_length)

        self.softmax = nn.Softmax(-1)

        self.output = nn.Linear(self.embedding_dim, self.vocab_size) 

        self.decoder_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.decoder_layers.append(DecoderLayer(self.num_heads, self.embedding_dim, self.ffn_hidden_dim, self.qk_length, self.value_length, self.dropout))

        #raise NotImplementedError("Need to implement Decoder layers")

    def forward(
        self,
        x: torch.Tensor,
        enc_x: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The forward pass of the Decoder.
        """

        x = x.long()

        embed_output = self.embedding(x)
        decoder_output = self.pos_enc(embed_output)
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, enc_x, tgt_mask, src_mask)

        linear_output = self.output(decoder_output)

        #softmax_output = self.softmax(linear_output)

        return linear_output

        #raise NotImplementedError("Need to implement forward pass of Decoder")
