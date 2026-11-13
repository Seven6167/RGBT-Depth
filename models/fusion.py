"""Continuous K-stage fusion network and shared density prediction head.

The registered module hierarchy and all forward expressions are preserved.
"""

from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from .msfm import MSSM, MSFM
from .state_space import MB
from .state_space import MFB


class AB(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0)
        )
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):  # 输入 BHWC 输出 BHWC
        x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        x = self.conv(x)
        skip = x
        x1, x2 = torch.split(x, C // 2, dim=1)
        avg_out = self.mlp1(self.avg_pool(x1))
        max_out = self.mlp2(self.max_pool(x2))
        y1 = self.sigmoid1(avg_out)
        y2 = self.sigmoid2(max_out)
        z = torch.cat((x1 * y1, x2 * y2), dim=1)
        perm = torch.randperm(C)
        z = z[:, perm, :, :]
        z = z + skip
        z = z.permute(0, 2, 3, 1).contiguous()
        return z


class FB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.select1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.select2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):  # 输入 BHWC 输出 BHWC
        B, H, W, C = x.shape
        skip = x
        x = x.permute(0, 3, 1, 2).contiguous()
        y = torch.fft.rfft2(x) + 1e-8
        a = torch.abs(y)
        p = torch.angle(y)
        a = self.select1(a)
        p = self.select2(p)
        real = a * torch.cos(p)
        imag = a * torch.sin(p)
        out = torch.complex(real, imag) + 1e-8
        out = torch.fft.irfft2(out, s=(H, W), norm='backward') + 1e-8
        out = torch.abs(out) + 1e-8

        out = out.permute(0, 2, 3, 1).contiguous()
        out = out + skip

        return out


class Block(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            MSS_en: bool = False,
            FSS_en: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln = norm_layer(hidden_dim)
        self.mamba = MB(d_model=hidden_dim, d_state=d_state, expand=mlp_ratio, dropout=attn_drop_rate,
                        **kwargs)
        self.drop_path = DropPath(drop_path)
        self.attention = AB(hidden_dim)
        self.frequency = FB(hidden_dim)
        self.MSS_en = MSS_en
        self.FSS_en = FSS_en
        if self.MSS_en:
            self.MSSM = MSSM(hidden_dim)
        if self.FSS_en:
            self.MSFM = MSFM(hidden_dim)
    def forward(self, input, x_size):
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln(input)
        x = x.permute(0,3,1,2)
        if self.MSS_en:
            x = self.MSSM(x)
        if self.FSS_en:
            x = self.MSFM(x)        
        x = x.permute(0,2,3,1)
        # x = self.drop_path(self.mamba(x)) + self.attention(x) + self.frequency(x)

        x = x.view(B, -1, C).contiguous()
        return x


class Blocks(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=2.,
                 drop_path=0.,
                 MSS_en = False,
                 FSS_en = False
                 ):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(Block(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                mlp_ratio=mlp_ratio,
                d_state=16,
                MSS_en = MSS_en,
                FSS_en = FSS_en

            ))

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        return x


class BlocksGroup(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=2.,
                 drop_path=0.,
                 MSS_en = False,
                 FSS_en = False
                 ):
        super().__init__()

        self.dim = dim

        self.residual_group = Blocks(
            dim=dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
                MSS_en = MSS_en,
                FSS_en = FSS_en
        )

    def forward(self, x, x_size):
        return self.residual_group(x, x_size) + x


class Fusion(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 embed_dim,
                 depths,
                 mlp_ratio,
                 drop_rate,
                 norm_layer,
                 patch_norm,
                 MSSM_en = True,
                 MSFM_en = True,
                 NLMF_en = False,
                 **kwargs):
        super(Fusion, self).__init__()
        num_in_ch = in_chans
        num_out_ch = out_chans

        # ------------------------- shallow feature extraction ------------------------- #
        self.conv_y_1 = nn.Conv2d(num_in_ch, int(embed_dim / 2), 3, 1, 1)
        self.conv_ir_1 = nn.Conv2d(num_in_ch, int(embed_dim / 2), 3, 1, 1)
        self.conv_y_2 = nn.Conv2d(int(embed_dim / 2), embed_dim, 3, 1, 1)
        self.conv_ir_2 = nn.Conv2d(int(embed_dim / 2), embed_dim, 3, 1, 1)
        # ------------------------- deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.pro1_en = MSSM_en
        self.pro2_en = MSFM_en
        self.pro3_en = NLMF_en

        self.layers_y = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BlocksGroup(
                dim=embed_dim,
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                MSS_en = self.pro1_en,
                # FSS_en = self.pro2_en,
            )
            self.layers_y.append(layer)
        self.norm_y = norm_layer(self.num_features)

        self.layers_fuse = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BlocksGroup(
                dim=embed_dim,
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                MSS_en = self.pro1_en,
                FSS_en = self.pro2_en,
            )
            self.layers_fuse.append(layer)
        self.norm_fuse = norm_layer(self.num_features)

        self.layers_ir = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BlocksGroup(
                dim=embed_dim,
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                # MSS_en = self.pro1_en,
                FSS_en = self.pro2_en,
            )
            self.layers_ir.append(layer)
        self.norm_ir = norm_layer(self.num_features)

        # ------------------------- fusion module ------------------------- #

        self.layers_fusion = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MFB(hidden_dim=embed_dim, drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                        d_state=16,
                        mlp_ratio=2.,NSS_en =self.pro3_en)
            
            self.layers_fusion.append(layer)

        # ------------------------- restoration module ------------------------- #

        self.conv_last_1 = nn.Sequential(
            nn.Conv2d(embed_dim, int(embed_dim / 2), 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.conv_last_2 = nn.Conv2d(int(embed_dim / 2), num_out_ch, 3, 1, 1)

        self.y_conv_last_1 = nn.Sequential(
            nn.Conv2d(embed_dim, int(embed_dim / 2), 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.y_conv_last_2 = nn.Conv2d(int(embed_dim / 2), num_out_ch, 3, 1, 1)

        self.ir_conv_last_1 = nn.Sequential(
            nn.Conv2d(embed_dim, int(embed_dim / 2), 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.ir_conv_last_2 = nn.Conv2d(int(embed_dim / 2), num_out_ch, 3, 1, 1)

    def forward_features_y(self, x):  # B,C,H,W
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # B,L,C
        x = self.pos_drop(x)
        z = []
        for layer in self.layers_y:
            x = layer(x, x_size)
            z.append(x)
        out = []
        for i in z:
            i = self.norm_y(i)
            i = self.patch_unembed(i, x_size)
            out.append(i)
        x = self.norm_y(x)  # B,L,C
        x = self.patch_unembed(x, x_size)  # B,C,H,W
        return x, out

    def forward_features_fuse(self, x, out_y, out_ir, route_mask=None):  # B,C,H,W
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # B,L,C
        x = self.pos_drop(x)
        i = 0
        for layer in self.layers_fuse:
            x = layer(x, x_size)
            x = self.norm_fuse(x)  # B,L,C
            x = self.patch_unembed(x, x_size)  # B,C,H,W
            x_fused = self.layers_fusion[i](out_y[i], out_ir[i], x)
            if route_mask is not None:
                gate = route_mask[:, i].view(-1, 1, 1, 1).to(dtype=x.dtype, device=x.device)
                x = gate * x_fused + (1.0 - gate) * x
            else:
                x = x_fused
            i = i + 1
            x = self.patch_embed(x)  # B,L,C
            x = self.pos_drop(x)

        x = self.norm_fuse(x)  # B,L,C
        x = self.patch_unembed(x, x_size)  # B,C,H,W
        return x

    def forward_features_ir(self, x):  # B,C,H,W
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # B,L,C
        x = self.pos_drop(x)
        z = []
        for layer in self.layers_ir:
            x = layer(x, x_size)
            z.append(x)
        out = []
        for i in z:
            i = self.norm_ir(i)
            i = self.patch_unembed(i, x_size)
            out.append(i)
        x = self.norm_ir(x)  # B,L,C
        x = self.patch_unembed(x, x_size)  # B,C,H,W
        return x, out

    def forward(self, y, ir, route_mask=None):
        y1 = self.conv_y_1(y)
        y2 = self.conv_y_2(y1)
        ir1 = self.conv_ir_1(ir)
        ir2 = self.conv_ir_2(ir1)
        fuse1 = torch.concat([y1, ir1], dim=1)
        y3, out_y = self.forward_features_y(y2)
        y3 = y3 + y2
        ir3, out_ir = self.forward_features_ir(ir2)
        ir3 = ir3 + ir2
        fuse2 = self.forward_features_fuse(fuse1, out_y, out_ir, route_mask=route_mask)
        
        y_out = self.y_conv_last_2(self.y_conv_last_1(y3))
        ir_out = self.ir_conv_last_2(self.ir_conv_last_1(ir3))

        fuse_out = self.conv_last_2(self.conv_last_1(fuse2))
        
        return fuse_out, y_out, ir_out


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim, norm_layer=None):
        super().__init__()
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x
