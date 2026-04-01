from typing import Optional

import torch
import torch.nn as nn




class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads: int, embedding_dim: int, qk_length: int, value_length: int
    ):
        """
        The Multi-Head Attention layer will take in Q, K, and V
        matrices and will output an attention matrix of shape <TODO>.

        First, Q, K, and V should be projected to have
        a shape of (B, T, C) where C = num_heads * qk_length
        (OR value_length). You are then expected to split
        the C dimension into num_heads different heads, each
        with shape (B, T, vec_length).

        Next, you will compute the scaled dot-product attention
        between Q, K, and V.

        Finally, you will concatenate the heads and project the
        output to have a shape of (B, T, C).

        Check out the `masked_fill` method in PyTorch to help
        you implement the masking step!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # (hint: number of Linear layers needed != 3)
        #self.project_q = nn.Linear(self.num_heads * self.qk_length, self.num_heads * self.qk_length)
        #self.project_k = nn.Linear(self.num_heads * self.qk_length, self.num_heads * self.qk_length)
        #self.project_v = nn.Linear(self.num_heads * self.value_length, self.num_heads * self.value_length)
        self.project_o = nn.Linear(self.num_heads * self.value_length, self.embedding_dim)
        self.softmax = nn.Softmax(dim=-1)

        #raise NotImplementedError("Need to implement MHA layers")

    def split_heads(self, x: torch.Tensor, vec_length: int) -> torch.Tensor:
        """
        Split the C dimension of the input tensor into num_heads
        different heads, each with shape (B, T, vec_length).
        Hint: check out the `view` and 'permute` methods in PyTorch to help
        you reshape the tensor.

        Args:
            x: torch.Tensor of shape (B, T, C), where C = num_heads * vec_length
            vec_length: int, the length of the query/key/value vectors

        Returns:
            torch.Tensor of shape (B, num_heads, T, vec_length)
        """
        B, T, C = x.size()

        assert C // self.num_heads == vec_length, (
            "Input tensor does not have the correct shape for splitting."
        )

        x_v = x.view(B, T, self.num_heads, vec_length)
        x_p = x_v.permute(0, 2, 1, 3)

        return x_p

        #raise NotImplementedError("Need to implement split_heads")

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the num_heads different heads into a single tensor.
        Hint: check out the `contiguous` method in PyTorch to help
        you reshape the tensor.

        Args:
            x: torch.Tensor of shape (B, num_heads, T, vec_length)

        Returns:
            torch.Tensor of shape (B, T, num_heads * vec_length)
        """
        B, num_heads, T, vec_length = x.size()

        x_p = x.permute(0, 2, 1, 3)

        x_c = x_p.contiguous(memory_format=torch.contiguous_format)

        x_v = x_c.view(B, T, num_heads*vec_length)

        return x_v

        #raise NotImplementedError("Need to implement combine_heads")

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the scaled dot-product attention given Q, K, and V.
        This is where the pad_mask and causal_mask are applied.

        Args:
            Q: torch.Tensor of shape (B, num_heads, T, qk_length)
            K: torch.Tensor of shape (B, num_heads, T, qk_length)
            V: torch.Tensor of shape (B, num_heads, T, value_length)
            mask: Optional boolean torch.Tensor, broadcastable to (B, num_heads, T, T).
        """
        # Transposing a 4-D K
        K_trans = K.permute(0, 1, 3, 2)

        qk_mask = (Q @ K_trans) / (self.qk_length)**0.5

        if mask is not None:
            qk_mask = qk_mask.masked_fill(mask, float('-inf'))
        else:
            pass

        # Softmax along the 'rows'
        sdp_attention = self.softmax(qk_mask) @ V

        return sdp_attention

        #raise NotImplementedError("Need to implement scaled_dot_product_attention")

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        The forward pass of the Multi-Head Attention layer.

        Args:
            Q: torch.Tensor of shape (B, T, C)
            K: torch.Tensor of shape (B, T, C)
            V: torch.Tensor of shape (B, T, C)
            mask: Optional torch.Tensor of shape (B, T, T) or None

        Returns:
            torch.Tensor of shape (B, T, C)
        """

        # Apply the linear layers to project/scale the components of Q, K, and V as needed for each head
        #Q_allheads = self.project_q(Q)
        #K_allheads = self.project_k(K)
        #V_allheads = self.project_v(V)

        # Split the Q, K, and Vs up
        Q_split_heads = self.split_heads(Q, self.qk_length)
        K_split_heads = self.split_heads(K, self.qk_length)
        V_split_heads = self.split_heads(V, self.value_length)

        sdp_attention = self.scaled_dot_product_attention(Q_split_heads, K_split_heads, V_split_heads, mask)

        concat_attentions = self.combine_heads(sdp_attention)

        output = self.project_o(concat_attentions)

        return output

        #raise NotImplementedError("Need to implement forward pass of MHA")



class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        The Feed-Forward Neural Network layer will take in
        an input tensor of shape (B, T, C) and will output
        a tensor of the same shape.

        The FFNN will have two linear layers, with a ReLU
        activation function in between.

        Args:
            hidden_dim: int, the size of the hidden layer
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Define any layers you'll need in the forward pass
        #raise NotImplementedError("Need to implement FeedForwardNN layers")

        self.layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the FeedForwardNN.
        """

        x_1 = self.layer1(x)
        x_1r = self.relu(x_1)
        x_2 = self.layer2(x_1r)

        return x_2
        #raise NotImplementedError("Need to implement forward pass of FeedForwardNN")
