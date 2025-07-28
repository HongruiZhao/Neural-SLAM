import torch
import torch.nn as nn
from einops import rearrange

from .attn import (
    PreNorm,
    Attention,
    FeedForward,
    DenseBlock,
    cache_fn,
)

class TensorViewTransformer(nn.Module):
    def __init__(
        self,
        rank: int,
        tensor_lengths: list[int],
        attn_dim: int = 128,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        weight_tie_layers: bool = False,
        dropout: float = 0.0,
    ):
        """
        Initializes the TensorViewTransformer.

        @param rank: The rank of the input tensors (feature dimension).
        @param tensor_lengths: A list or tuple of three integers representing the
                               spatial dimensions (lengths) of the X, Y, and Z tensors.
        @param attn_dim: The dimension of the attention mechanism.
        @param depth: The total number of transformer layers. Must be even.
        @param heads: The number of attention heads.
        @param dim_head: The dimension of each attention head.
        @param weight_tie_layers: Whether to share weights between attention layers.
        @param dropout: The dropout rate.
        """
        super().__init__()
        assert len(tensor_lengths) == 3, "tensor_lengths must be a list/tuple of 3 integers."
        assert depth % 2 == 0, "Depth must be even for two-stage attention."

        self.tensor_lengths = tensor_lengths
        self.num_views = 3

        # Input projection to map rank to attn_dim
        self.input_projection = DenseBlock(rank, attn_dim, norm=None, activation=None)

        # Learnable positional embedding for the concatenated sequence of all views
        total_length = sum(self.tensor_lengths)
        self.pos_embed = nn.Parameter(torch.randn(1, total_length, attn_dim))

        # --- Transformer Layers ---
        get_attn = lambda: PreNorm(
            attn_dim,
            Attention(
                attn_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            ),
        )
        get_ff = lambda: PreNorm(attn_dim, FeedForward(attn_dim))
        get_attn, get_ff = map(cache_fn, (get_attn, get_ff))

        # if weight_tie_layers = True, all layers share the same weights
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([get_attn(**cache_args), get_ff(**cache_args)])
            )

        # Decoder to produce 3D heatmap
        self.decoder = DenseBlock(attn_dim, rank, norm=None, activation=None)


    def forward(self, views: list[torch.Tensor]):
        """

        @param views: A list of 3 tensors from TensorCP, each with shape (B, rank, length, 1).
                      Order should be [x_tensor, y_tensor, z_tensor].
        @return: A tensor representing the 3D translation heatmap of shape (B, L_x, L_y, L_z).
        """

        # projection and position 
        views = [rearrange(view.squeeze(-1), 'b r l -> b l r')
                    for view in views]
        views = torch.cat(views, dim=1)
        projected_views = self.input_projection(views) # (B, total_len, attn_dim)
        views_with_pos = torch.split(projected_views+self.pos_embed,
                                     self.tensor_lengths, dim=1)

        # self-attention
        num_self_att = len(self.layers) // 2
        processed_views = []
        for view_seq in views_with_pos:
            for attn, ff in self.layers[:num_self_att]:
                view_seq = attn(view_seq) + view_seq
                view_seq = ff(view_seq) + view_seq
            processed_views.append(view_seq)

        # cross-attention
        x = torch.cat(processed_views, dim=1)
        for attn, ff in self.layers[num_self_att:]:
            x = attn(x) + x
            x = ff(x) + x

        # Decode to heatmap values
        # (B, total_len, rank)
        decoded_tokens = self.decoder(x)

        # 3D heatmap
        x_heat, y_heat, z_heat = \
        torch.split(decoded_tokens, self.tensor_lengths, dim=1)
        heatmap_3d = torch.einsum('bxr, byr, bzr ->bxyz', 
                                  x_heat, y_heat, z_heat)


        return heatmap_3d



