# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import torch
from torch import nn
from einops import rearrange

from .attn import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    FeedForward,
    cache_fn,
    DenseBlock,
)

class SimplifiedMVT(nn.Module):
    """
    A simplified version of the Multi-View Transformer (MVT) that only processes
    visual information from a fixed number of camera views (3) to produce
    translation heatmaps.

    This version removes support for proprioceptive and language inputs, and only
    outputs the raw heatmaps for translation prediction, excluding rotation and
    other features for simplicity.
    """
    def __init__(
        self,
        depth: int,
        img_size: int,
        img_feat_dim: int,
        im_channels: int,
        attn_dim: int,
        attn_heads: int,
        attn_dim_head: int,
        activation: str,
        weight_tie_layers: bool,
        attn_dropout: float,
        img_patch_size: int,
        final_dim: int,
        use_fast_attn: bool = False,
    ):
        """
        Initializes the SimplifiedMVT model.

        :param depth: Depth of the attention network.
        :param img_size: Size of the input images (height and width).
        :param img_feat_dim: Dimension of features for each pixel in the input image.
        :param im_channels: Intermediate channel size for convolutional blocks.
        :param attn_dim: Dimension of the attention mechanism.
        :param attn_heads: Number of attention heads.
        :param attn_dim_head: Dimension of each attention head.
        :param activation: Activation function to use.
        :param weight_tie_layers: Whether to share weights between attention layers.
        :param attn_dropout: Dropout rate for the attention mechanism.
        :param img_patch_size: Size of the patches to extract from the image.
        :param final_dim: Final feature dimension before the output decoder.
        :param use_fast_attn: Whether to use xformers for faster attention.
        """
        super().__init__()
        self.num_img = 3
        self.img_size = img_size
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size

        # Input Preprocessing and Patching
        self.input_preprocess = Conv2DBlock(
            img_feat_dim,
            self.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=activation,
        )

        self.patchify = Conv2DBlock(
            self.im_channels,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )

        # Positional Encoding
        spatial_size = img_size // self.img_patch_size
        num_pe_token = spatial_size**2 * self.num_img
        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_pe_token, self.im_channels)
        )

        # Transformer Backbone
        self.fc_bef_attn = DenseBlock(
            self.im_channels, attn_dim, norm=None, activation=None
        )

        # lambda to return a new instance of the module 
        get_attn_attn = lambda: PreNorm(
            attn_dim,
            Attention(
                attn_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
                use_fast=use_fast_attn,
            ),
        )
        get_attn_ff = lambda: PreNorm(attn_dim, FeedForward(attn_dim))
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))

        # if weight_tie_layers = True, all layers share the same weights
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)])
            )

        self.fc_aft_attn = DenseBlock(
            attn_dim, self.im_channels, norm=None, activation=None
        )

        # Upsampling and Output Decoder
        self.up0 = Conv2DUpsampleBlock(
            self.im_channels,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm=None,
            activation=activation,
            out_size=self.img_size,
        )

        self.final = Conv2DBlock(
            self.im_channels + self.im_channels, # Concatenates pre-patch features
            final_dim,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=activation,
        )

        self.trans_decoder = Conv2DBlock(
            final_dim,
            1, # Output one channel for the heatmap
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=None,
        )


    def forward(self, img: torch.Tensor):
        """
        Forward pass for the SimplifiedMVT.

        :param img: A tensor of shape (bs, num_img, img_feat_dim, h, w)
        :return: A dictionary containing the translation heatmaps
                 {'trans': tensor of shape (bs, num_img, h, w)}
        """
        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        assert num_img == self.num_img, f"Expected {self.num_img} images, but got {num_img}"
        assert h == w == self.img_size, "Image dimensions do not match img_size"
        assert len(self.layers) % 2 == 0, "depth must be even for self-cross attention"

        img = img.view(bs * self.num_img, img_feat_dim, h, w)

        # Preprocess and create patches
        d0 = self.input_preprocess(img)
        ins = self.patchify(d0)
        ins = ins.view(bs, self.num_img, self.im_channels, num_pat_img, num_pat_img)
        ins = rearrange(ins, "b v c h w -> b (v h w) c")

        # add positional encoding
        ins += self.pos_encoding

        # Transformer backbone
        x = self.fc_bef_attn(ins)

        # Reshape for within-view self-attention
        num_patches_per_view = num_pat_img * num_pat_img
        x = rearrange(x, 'b (v p) d -> (b v) p d', v=self.num_img, p=num_patches_per_view)

        # 1. Self-attention within each view
        for self_attn, self_ff in self.layers[:len(self.layers) // 2]:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # Reshape for cross-attention
        x = rearrange(x, '(b v) p d -> b (v p) d', v=self.num_img)

        # 2. Cross-attention across views
        for cross_attn, cross_ff in self.layers[len(self.layers) // 2:]:
            x = cross_attn(x) + x
            x = cross_ff(x) + x

        x = self.fc_aft_attn(x)


        # Reshape back to image-like format
        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=self.num_img, h=num_pat_img)

        # Upsample and decode to get heatmaps
        u0 = self.up0(x)
        u0 = torch.cat([u0, d0], dim=1) # Skip connection
        u = self.final(u0)
        trans = self.trans_decoder(u).view(bs, self.num_img, h, w)

        return {"trans": trans}
    

    
