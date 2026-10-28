# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
    'deit_tiny_patch16_224_dual', 'deit_small_patch16_224_dual',
    'deit_base_patch16_224_dual',
    'deit_tiny_patch16_224_concat', 'deit_small_patch16_224_concat',
    'deit_base_patch16_224_concat',
    'deit_tiny_patch16_224_reverse', 'deit_small_patch16_224_reverse',
    'deit_base_patch16_224_reverse',
    'deit_tiny_patch16_224_frequency', 'deit_small_patch16_224_frequency',
    'deit_base_patch16_224_frequency',
]


def _dct_basis(length):
    """Build an orthonormal DCT-II basis for a fixed spatial dimension."""
    positions = torch.arange(length, dtype=torch.float32) + 0.5
    frequencies = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos(math.pi * frequencies * positions / length)
    basis[0].mul_(math.sqrt(1.0 / length))
    basis[1:].mul_(math.sqrt(2.0 / length))
    return basis


class PooledFrequencyTokenizer(nn.Module):
    """Convert non-overlapping signed DCT blocks into a fixed token grid."""

    def __init__(self, img_size, in_chans, embed_dim, pool_size, norm_layer):
        super().__init__()
        if isinstance(img_size, (tuple, list)):
            image_height, image_width = img_size
        else:
            image_height = image_width = img_size
        if pool_size <= 0:
            raise ValueError('freq_pool_size must be a positive integer')
        if pool_size > min(image_height, image_width):
            raise ValueError(
                'freq_pool_size cannot exceed the frequency-map dimensions; '
                f'got pool_size={pool_size} for image size '
                f'{(image_height, image_width)}'
            )
        if image_height % pool_size or image_width % pool_size:
            raise ValueError(
                'The image dimensions must be divisible by freq_pool_size so '
                'that every DCT coefficient belongs to exactly one frequency '
                f'block; got pool_size={pool_size} for image size '
                f'{(image_height, image_width)}'
            )

        self.image_size = (image_height, image_width)
        self.pool_size = pool_size
        self.register_buffer('dct_height', _dct_basis(image_height), persistent=False)
        self.register_buffer('dct_width', _dct_basis(image_width), persistent=False)

        ac_mask = torch.ones(1, 1, image_height, image_width)
        ac_mask[..., 0, 0] = 0.0
        self.register_buffer('ac_mask', ac_mask, persistent=False)

        block_size = (image_height // pool_size, image_width // pool_size)
        # A strided convolution is a shared linear projection of each complete
        # frequency block. Unlike average pooling, it can distinguish every
        # coefficient inside the block without increasing the token count.
        self.tokenizer = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=block_size,
            stride=block_size,
        )
        self.token_norm = norm_layer(embed_dim)
        trunc_normal_(self.tokenizer.weight, std=.02)
        nn.init.zeros_(self.tokenizer.bias)

    @property
    def num_tokens(self):
        return self.pool_size * self.pool_size

    def forward(self, x):
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(
                'The frequency tokenizer was configured for images of size '
                f'{self.image_size}, but received {tuple(x.shape[-2:])}'
            )

        # Keep the fixed spectral transform in fp32 even when training uses AMP.
        autocast_disabled = (
            torch.cuda.amp.autocast(enabled=False)
            if x.device.type == 'cuda' else nullcontext()
        )
        with autocast_disabled:
            x_float = x.float()
            spectrum = torch.matmul(self.dct_height, x_float)
            spectrum = torch.matmul(spectrum, self.dct_width.transpose(0, 1))

            # Keep coefficient signs (spatial structure) while logarithmically
            # compressing the large dynamic range. DC remains excluded so this
            # branch measures AC frequency information only.
            signed_spectrum = (
                torch.sign(spectrum) * torch.log1p(spectrum.abs()) * self.ac_mask
            )
        frequency_grid = self.tokenizer(signed_spectrum.to(dtype=x.dtype))
        tokens = frequency_grid.flatten(2).transpose(1, 2)
        return self.token_norm(tokens)


class SpatialFrequencyVisionTransformer(VisionTransformer):
    """DeiT with block-projected DCT tokens guiding spatial cross-attention."""

    def __init__(self, freq_pool_size=7, **kwargs):
        # timm 1.x supplies registry metadata to model factory functions. It is
        # not part of VisionTransformer's constructor in every supported version.
        kwargs.pop('pretrained_cfg', None)
        kwargs.pop('pretrained_cfg_overlay', None)
        kwargs.pop('cache_dir', None)
        embed_dim = kwargs.get('embed_dim', 768)
        num_heads = kwargs.get('num_heads', 12)
        in_chans = kwargs.get('in_chans', 3)
        img_size = kwargs.get('img_size', 224)
        norm_layer = kwargs.get('norm_layer', nn.LayerNorm)
        attn_drop_rate = kwargs.get('attn_drop_rate', 0.0)
        drop_rate = kwargs.get('drop_rate', 0.0)
        super().__init__(**kwargs)

        self.frequency_tokenizer = PooledFrequencyTokenizer(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            pool_size=freq_pool_size,
            norm_layer=norm_layer,
        )
        self.freq_pos_embed = nn.Parameter(
            torch.zeros(1, self.frequency_tokenizer.num_tokens, embed_dim)
        )
        self.spatial_fusion_norm = norm_layer(embed_dim)
        self.frequency_fusion_norm = norm_layer(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop_rate,
        )
        self.fusion_drop = nn.Dropout(drop_rate)

        trunc_normal_(self.freq_pos_embed, std=.02)
        self.frequency_tokenizer.apply(self._init_weights)
        self.spatial_fusion_norm.apply(self._init_weights)
        self.frequency_fusion_norm.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'freq_pos_embed'}

    def forward_features(self, x, attn_mask=None):
        batch_size = x.shape[0]

        spatial_tokens = self.patch_embed(x)
        spatial_tokens = spatial_tokens + self.pos_embed[:, 1:]

        frequency_tokens = self.frequency_tokenizer(x)
        frequency_tokens = frequency_tokens + self.freq_pos_embed

        # MultiheadAttention in the supported PyTorch versions expects L x B x D.
        queries = self.spatial_fusion_norm(spatial_tokens).transpose(0, 1)
        keys_and_values = self.frequency_fusion_norm(frequency_tokens).transpose(0, 1)
        frequency_context, _ = self.cross_attention(
            queries,
            keys_and_values,
            keys_and_values,
            need_weights=False,
        )
        spatial_tokens = spatial_tokens + self.fusion_drop(
            frequency_context.transpose(0, 1)
        )

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        cls_token = cls_token + self.pos_embed[:, :1]
        tokens = torch.cat((cls_token, spatial_tokens), dim=1)
        tokens = self.pos_drop(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return tokens[:, 0]

    def forward(self, x):
        return self.head(self.forward_features(x))


class ConcatenatedSpatialFrequencyVisionTransformer(SpatialFrequencyVisionTransformer):
    """Keep all spatial and frequency tokens in one shared DeiT sequence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        embed_dim = self.embed_dim

        # Concatenation needs no pre-encoder cross-attention. Removing these
        # modules keeps its parameter count and comparison unambiguous.
        del self.spatial_fusion_norm
        del self.frequency_fusion_norm
        del self.cross_attention
        del self.fusion_drop

        self.spatial_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.frequency_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.spatial_type_embed, std=.02)
        trunc_normal_(self.frequency_type_embed, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed', 'cls_token', 'freq_pos_embed',
            'spatial_type_embed', 'frequency_type_embed',
        }

    def forward_features(self, x, attn_mask=None):
        batch_size = x.shape[0]
        spatial_tokens = (
            self.patch_embed(x) + self.pos_embed[:, 1:] + self.spatial_type_embed
        )
        frequency_tokens = (
            self.frequency_tokenizer(x)
            + self.freq_pos_embed
            + self.frequency_type_embed
        )

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        cls_token = cls_token + self.pos_embed[:, :1]
        tokens = torch.cat((cls_token, spatial_tokens, frequency_tokens), dim=1)
        tokens = self.pos_drop(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return tokens[:, 0]


class ReverseCrossAttentionVisionTransformer(SpatialFrequencyVisionTransformer):
    """Use frequency queries so cross-attention preserves every frequency token."""

    def forward_features(self, x, attn_mask=None):
        batch_size = x.shape[0]
        spatial_tokens = self.patch_embed(x) + self.pos_embed[:, 1:]
        frequency_tokens = self.frequency_tokenizer(x) + self.freq_pos_embed

        queries = self.frequency_fusion_norm(frequency_tokens).transpose(0, 1)
        keys_and_values = self.spatial_fusion_norm(spatial_tokens).transpose(0, 1)
        spatial_context, _ = self.cross_attention(
            queries,
            keys_and_values,
            keys_and_values,
            need_weights=False,
        )
        frequency_tokens = frequency_tokens + self.fusion_drop(
            spatial_context.transpose(0, 1)
        )

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        cls_token = cls_token + self.pos_embed[:, :1]
        tokens = torch.cat((cls_token, frequency_tokens), dim=1)
        tokens = self.pos_drop(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return tokens[:, 0]


class FrequencyVisionTransformer(VisionTransformer):
    """DeiT ablation that classifies using only block-projected DCT tokens."""

    def __init__(self, freq_pool_size=7, **kwargs):
        kwargs.pop('pretrained_cfg', None)
        kwargs.pop('pretrained_cfg_overlay', None)
        kwargs.pop('cache_dir', None)
        embed_dim = kwargs.get('embed_dim', 768)
        in_chans = kwargs.get('in_chans', 3)
        img_size = kwargs.get('img_size', 224)
        norm_layer = kwargs.get('norm_layer', nn.LayerNorm)
        super().__init__(**kwargs)

        self.frequency_tokenizer = PooledFrequencyTokenizer(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            pool_size=freq_pool_size,
            norm_layer=norm_layer,
        )
        # The spatial patch projection and its positional embedding are not part
        # of this ablation. Removing them also keeps the parameter count honest.
        del self.patch_embed
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.frequency_tokenizer.num_tokens + 1, embed_dim
        ))

        trunc_normal_(self.pos_embed, std=.02)
        self.frequency_tokenizer.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, attn_mask=None):
        batch_size = x.shape[0]
        frequency_tokens = self.frequency_tokenizer(x)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, frequency_tokens), dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return tokens[:, 0]

    def forward(self, x):
        return self.head(self.forward_features(x))


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_patch16_224_dual(pretrained=False, **kwargs):
    model = SpatialFrequencyVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def deit_small_patch16_224_dual(pretrained=False, **kwargs):
    model = SpatialFrequencyVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def deit_base_patch16_224_dual(pretrained=False, **kwargs):
    model = SpatialFrequencyVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


_DEIT_VARIANTS = {
    'tiny': (192, 3, 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'small': (384, 6, 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'base': (768, 12, 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'),
}


def _create_frequency_fusion_model(model_cls, variant, pretrained, **kwargs):
    embed_dim, num_heads, url = _DEIT_VARIANTS[variant]
    model = model_cls(
        patch_size=16, embed_dim=embed_dim, depth=12, num_heads=num_heads,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location='cpu', check_hash=True)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


@register_model
def deit_tiny_patch16_224_concat(pretrained=False, **kwargs):
    return _create_frequency_fusion_model(
        ConcatenatedSpatialFrequencyVisionTransformer, 'tiny', pretrained, **kwargs)


@register_model
def deit_small_patch16_224_concat(pretrained=False, **kwargs):
    return _create_frequency_fusion_model(
        ConcatenatedSpatialFrequencyVisionTransformer, 'small', pretrained, **kwargs)


@register_model
def deit_base_patch16_224_concat(pretrained=False, **kwargs):
    return _create_frequency_fusion_model(
        ConcatenatedSpatialFrequencyVisionTransformer, 'base', pretrained, **kwargs)


@register_model
def deit_tiny_patch16_224_reverse(pretrained=False, **kwargs):
    return _create_frequency_fusion_model(
        ReverseCrossAttentionVisionTransformer, 'tiny', pretrained, **kwargs)


@register_model
def deit_small_patch16_224_reverse(pretrained=False, **kwargs):
    return _create_frequency_fusion_model(
        ReverseCrossAttentionVisionTransformer, 'small', pretrained, **kwargs)


@register_model
def deit_base_patch16_224_reverse(pretrained=False, **kwargs):
    return _create_frequency_fusion_model(
        ReverseCrossAttentionVisionTransformer, 'base', pretrained, **kwargs)


def _load_spatial_deit_backbone(model, url):
    """Load reusable DeiT encoder weights into a frequency-only model."""
    checkpoint = torch.hub.load_state_dict_from_url(
        url=url, map_location="cpu", check_hash=True
    )["model"]
    checkpoint = {
        name: value for name, value in checkpoint.items()
        if not name.startswith('patch_embed.') and name != 'pos_embed'
    }
    model.load_state_dict(checkpoint, strict=False)


@register_model
def deit_tiny_patch16_224_frequency(pretrained=False, **kwargs):
    model = FrequencyVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        _load_spatial_deit_backbone(
            model,
            "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        )
    return model


@register_model
def deit_small_patch16_224_frequency(pretrained=False, **kwargs):
    model = FrequencyVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        _load_spatial_deit_backbone(
            model,
            "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        )
    return model


@register_model
def deit_base_patch16_224_frequency(pretrained=False, **kwargs):
    model = FrequencyVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        _load_spatial_deit_backbone(
            model,
            "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        )
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
