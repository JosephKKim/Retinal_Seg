# -*- coding:utf-8 -*-
import math
import time
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile

'''
Module that gives cross attention

x1, x2 as input


'''


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    # input x [B, H, W, C]
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # reshape:  [B, H // wid_size, window_size, W // wind_size, win_size, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # [B, H // wid_size, W // wind_size, win_size, win_size, C] -> 앞에 3개를 합쳐버림
    # print("windows: ", windows.shape)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, window_size2, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.window_size2 = window_size2
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # dimension 크기로 나눠주는거
        self.upscale = 2

        # define a parameter table of relative position bias

        # 필요한 개수 몇개?
        ####### relative for Upsample: swin_u.py #####

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((((self.window_size[0] + self.window_size[1] * self.upscale) - 1) ** 2,
                         num_heads)))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.window_size[0])

        coords_w = torch.arange(window_size[0] * self.upscale)

        coords_1 = torch.stack(torch.meshgrid([coords_h, coords_h]))
        coords_2 = torch.stack(torch.meshgrid([coords_w, coords_w]))
        coords_r1 = coords_1[0].flatten(0)
        coords_r2 = coords_2[0].flatten(0)
        coords_c1 = coords_1[1].flatten(0)
        coords_c2 = coords_2[1].flatten(0)

        rel_row = coords_r1[:, None] - coords_r2[None, :]
        rel_col = coords_c1[:, None] - coords_c2[None, :]
        relative_coords = torch.stack((rel_row, rel_col))
        # [2, 16, 64]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # [16, 64, 2]

        relative_coords[:, :, 0] += self.window_size[0] * self.upscale - 1
        relative_coords[:, :, 1] += self.window_size[0] * self.upscale - 1
        # 1D로 표현해야하니까 한줄 땡겨줘라 이말이야
        relative_coords[:, :, 0] *= (self.window_size[0] + self.window_size[1] * self.upscale) - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        # # print("rpi: ", relative_position_index.shape)

        # relative position bias table

        ####### Original code ######
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # # print("relateive_coords: ", relative_coords)
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.qkv2 = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        # print(self.relative_position_bias_table.shape)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, mask=None):

        # x1, x2 모두 이런 shape로 들어오도록 [512, 16, 256]
        # [512, 16, 256]

        # 어차피 shape는 같으니까
        B_, N, C = x1.shape
        B_2, N2, C2 = x2.shape

        qkv = self.qkv(x1)
        qkv_2 = self.qkv2(x2)
        # TODO:q, kv 따로
        qkv = qkv.reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        # q, k, v= qkv[0], qkv[1], qkv[1]
        # print("q shape: ", q.shape) # [512, 1, 16, 256] [512, # heads, 16, C / # heads]
        qkv_2 = qkv_2.reshape(B_2, N2, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = qkv_2[0], qkv_2[1]
        # q2, k2, v2 = qkv_2[0], qkv_2[1], qkv_2[2]

        q = q * self.scale
        attn = (q @ k2.transpose(-2, -1))

        # q = q * self.scale
        # print("up_k", up_k.shape)
        # print("q", q.shape)

        # attn = (q @ k.transpose(-2, -1))
        # print(k.shape)
        # attn = (q @ up_k.transpose(-2, -1))
        # [512, 1, 16, 64]

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[0], (self.window_size[0] * self.upscale) ** 2, -1)
        # # print("rpb shaep before reshape", relative_position_bias.shape)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # print("rpb shape: ", relative_position_bias.shape)

        attn = attn + relative_position_bias.unsqueeze(0)
        # [512, 1, 16, 64]

        # TODO mask problem

        # print("attention: ", attn.shape) [512, 1, 16, 16]
        if mask is not None:
            # print("mask is not none?")
            # mask [64, 16, 16]
            nW = mask.shape[0]
            # print(mask)
            # print(mask.unsqueeze(1).unsqueeze(0).shape)
            # 지금 현재 문제 생기는 곳: 여기?
            # print(attn.view(B_ // nW, nW, self.num_heads, N, N).shape)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        # print(attn.shape)

        # x = (attn @ up_v).transpose(1, 2).reshape(B_, N, C)
        # print("up_v: ", up_v.shape)
        # print("x: ", x.shape)
        x = (attn @ v2).transpose(1, 2).reshape(B_, N, C)
        # print("x: ", x.shape)
        # print("q x k v", x.shape) [512, 16, 256]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class LocalContextExtractor(nn.Module):

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class ContextAwareTransformer(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=8, window_size2=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_size2 = window_size2
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), window_size2=to_2tuple(self.window_size2),
            num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            print("calculate_mask?")
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        # print("w_sclices: ", w_slices)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # print(attn_mask)
        return attn_mask

    def forward(self, x, x2, x_size, x2_size):
        # print(x.shape, x2.shape)
        H, W = x_size
        B, L, C = x.shape

        H2, W2 = x2_size
        B2, L2, C2 = x2.shape

        # assert L == H * W, "input feature has wrong size"
        # print("inside transformer: ", x.shape)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # local context features
        lcf = x.permute(0, 3, 1, 2)

        x2 = x2.view(B2, H2, W2, C2)
        # [8, 256, 32, 32]
        # cyclic shift

        # skip here
        if self.shift_size > 0:
            print("shifted!")
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_x2 = x2
        # partition windows
        # x를 window로 나누어줌 [512, 4, 4, 256]
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        x_windows2 = window_partition(shifted_x2, self.window_size2)  # nW*B, window_size, window_size, C

        x_windows2 = x_windows2.view(-1, self.window_size2 * self.window_size2, C)  # nW*B, window_size*window_size, C
        # [512, 16, 256]
        # [512, 16, 256]
        # print("x_windows: ", x_windows.shape)

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        # print(self.input_resolution, x_size)
        # if self.input_resolution == x_size:
        # attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        ## 여기에 이제 2개씩 들어가도록 해야함

        attn_windows = self.attn(x_windows, x_windows2, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # else:
        #     # print("calculate mask")
        #     attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # print("attn_windows before: ", attn_windows.shape)
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # print("attn_windows: ", attn_windows.shape)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        # print("shifted ", shifted_x.shape)
        # [8, 32, 32, 256]
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # local context
        # lc = self.lce(lcf)
        # lc = lc.view(B, C, H * W).permute(0, 2, 1)
        # x = lc + x
        # print("out from CAT: ", x.shape)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, window_size2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            ContextAwareTransformer(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    window_size2=window_size2,
                                    shift_size=0,  # if (i % 2 == 0) else window_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x2, x_size, x_size2):
        # input [B, 12 * 12, 256]

        for blk in self.blocks:
            if self.use_checkpoint:
                # ?
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x2, x_size, x_size2)  # B L C
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class ContextAwareTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, window_size2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         window_size2=window_size2,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.dilated_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=2, bias=True, dilation=2)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.dilated_conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, kernel_size=3, padding=2, dilation=2),  # 크기 유지한채로 dilation conv. 적용
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, kernel_size=3, padding=2, dilation=2)
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x2, x_size, x2_size):
        res = self.residual_group(x, x2, x_size, x2_size)  # B L C
        res = self.patch_unembed(res, x_size)  # B c H W
        # print("after unembed: ", res.shape)
        res = self.dilated_conv(res)
        res = self.patch_embed(res) + x
        # print("after embed: ", res.shape)
        return res

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # print("before patch embed: ", x.shape)
        x = x.flatten(2).transpose(1, 2)  # B C H W ==> B Ph*Pw C
        if self.norm is not None:
            # print(self.norm)
            x = self.norm(x)
        # print("after patch embed: ", x.shape) [8, 1024, 256]
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class SpatialAttentionModule(nn.Module):

    def __init__(self, dim):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map


class HDRTransformer(nn.Module):

    def __init__(self, img_size=32, patch_size=1, in_chans=6,
                 embed_dim=256, depths=[3], num_heads=[1],
                 window_size=4, window_size2=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(HDRTransformer, self).__init__()

        ################################### 2. HDR Reconstruction Network ###################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.up_channel = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1, stride=1)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Context-aware Transformer Blocks (CTBs)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ContextAwareTransformerBlock(dim=embed_dim,
                                                 input_resolution=(patches_resolution[0],
                                                                   patches_resolution[1]),
                                                 depth=depths[i_layer],
                                                 num_heads=num_heads[i_layer],
                                                 window_size=window_size,
                                                 window_size2=window_size2,
                                                 mlp_ratio=self.mlp_ratio,
                                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                                 norm_layer=norm_layer,
                                                 downsample=None,
                                                 use_checkpoint=use_checkpoint,
                                                 img_size=img_size,
                                                 patch_size=patch_size,
                                                 resi_connection=resi_connection
                                                 )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, x2):
        # x [B, embed_dim, h, w]
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # B L C

        x2 = self.up_channel(x2)
        x2_size = (x2.shape[2], x2.shape[3])
        x2 = self.patch_embed1(x2)

        # [8, 1024 (32 x 32), 256]
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x2 = self.pos_drop(x2)
        # print(self.layers)
        # layer가 CATBlock 이므로
        for layer in self.layers:
            x = layer(x, x2, x_size, x2_size)
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x1, x2):

        # CTBs for HDR reconstruction
        x = self.conv_after_body(self.forward_features(x1, x2) + x1)

        return x

# model = HDRTransformer(img_size=12, embed_dim=128)


# inp = torch.zeros([256, 128, 12, 12])
# out = model(inp)
# print(out.shape)


# inp = torch.zeros([1, 128, 148, 148])
# out = model(inp)
# print(out.shape)



