"""Two-stream ResNet feature extractors and frozen FusionPro topology."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .fusion import Fusion as Fusionp


class RouterResidualCNN(nn.Module):
    """Shared residual CNN router that outputs per-layer routing logits."""
    def __init__(self, in_ch=128, hidden_ch=128, num_layers=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
        )
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_ch, num_layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.act(x + self.res(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _straight_through_binary(prob, threshold=0.5):
    hard = (prob >= threshold).to(prob.dtype)
    return hard.detach() - prob.detach() + prob


def _strict_prefix_mask(route_prob, hard=True, threshold=0.5, tau=0.3):
    """
    Convert per-layer route scores to strict prefix mask.
    Example for 5 layers: [1,1,1,0,0].
    """
    bsz, n_layers = route_prob.shape
    device = route_prob.device
    dtype = route_prob.dtype
    layer_idx = torch.arange(n_layers, device=device, dtype=dtype).view(1, -1)

    if hard:
        hard_vec = (route_prob >= threshold).to(dtype)
        k = hard_vec.sum(dim=1, keepdim=True)  # [B,1], integer-like in float
        return (layer_idx < k).to(dtype)

    # Soft prefix: monotonic decreasing gate approximating a prefix.
    k_soft = route_prob.sum(dim=1, keepdim=True).clamp(0.0, float(n_layers))
    return torch.sigmoid((k_soft - layer_idx - 0.5) / tau)


def _resnet_encoder(arch, pretrained):
    arch = arch.lower()
    if arch == 'resnet18':
        try:
            w = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=w)
        except AttributeError:
            net = models.resnet18(pretrained=bool(pretrained))
        out_ch = 512
    elif arch == 'resnet50':
        try:
            w = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=w)
        except AttributeError:
            net = models.resnet50(pretrained=bool(pretrained))
        out_ch = 512
    else:
        raise ValueError(f'Unsupported backbone: {arch}, choose from resnet18/resnet50')
    return net, out_ch


class FusionPro(nn.Module):
    """
    Two-stream ResNet18 backbone (RGB/IR) + Fusion internal architecture + density output.
    Features are taken at stride 8 (layer2 output), then fused by `Fusion`.
    """
    def __init__(
        self,
        pretrained_backbone=True,
        fusion_mode='fuse',
        embed_dim=128,
        router_select='max',
        router_hard=True,
        router_threshold=0.5,
        luyou_en = True,
        res18backbone = True,
        MSSM_en = True,
        MSFM_en = True,
    ):
        super().__init__()
        if fusion_mode not in ('fuse', 'rgb', 'ir', 'avg'):
            raise ValueError("fusion_mode must be one of: fuse/rgb/ir/avg")
        if router_select not in ('max', 'min'):
            raise ValueError("router_select must be one of: max/min")
        self.fusion_mode = fusion_mode
        self.router_select = router_select
        self.router_hard = bool(router_hard)
        self.router_threshold = float(router_threshold)

        self.luyou_en = luyou_en
        self.res18backbone = res18backbone
        self.MSSM_en = MSSM_en
        self.MSFM_en = MSFM_en
        self.router_layers = len([1, 1, 1, 1])
        if self.res18backbone:

            rgb_res, _ = _resnet_encoder('resnet18', pretrained_backbone)
            ir_res, _ = _resnet_encoder('resnet18', pretrained_backbone)
                    # if resnet18 in_chans = 128
            self.fusion = Fusionp(
                in_chans=128,
                out_chans=1,
                embed_dim=embed_dim,
                depths=[1,1,1,1],
                mlp_ratio=2.0,
                drop_rate=0.0,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                MSSM_en = self.MSSM_en,
                MSFM_en = self.MSFM_en,
            )
            self.shared_router = RouterResidualCNN(
            in_ch=128, hidden_ch=512, num_layers=self.router_layers
            )
        else:
            rgb_res, _ = _resnet_encoder('resnet50', pretrained_backbone)
            ir_res, _ = _resnet_encoder('resnet50', pretrained_backbone)
                    # if resnet18 in_chans = 128
            self.fusion = Fusionp(
                in_chans=512,
                out_chans=1,
                embed_dim=embed_dim,
                depths=[1,1,1,1],
                mlp_ratio=2.0,
                drop_rate=0.0,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                MSSM_en = self.MSSM_en,
                MSFM_en = self.MSFM_en,
            )
            self.shared_router = RouterResidualCNN(
            in_ch=512, hidden_ch=512, num_layers=self.router_layers
            )            
        # ResNet18 up to layer2 -> stride 8, channels 128
        self.rgb_enc = nn.Sequential(
            rgb_res.conv1, rgb_res.bn1, rgb_res.relu, rgb_res.maxpool,
            rgb_res.layer1, rgb_res.layer2,
        )
        self.ir_enc = nn.Sequential(
            ir_res.conv1, ir_res.bn1, ir_res.relu, ir_res.maxpool,
            ir_res.layer1, ir_res.layer2,
        )

        

        self.out_act = nn.ReLU(inplace=True)


    def _build_route_mask(self, fr, fi):
        rgb_logits = self.shared_router(fr)
        ir_logits = self.shared_router(fi)
        rgb_prob = torch.sigmoid(rgb_logits)
        ir_prob = torch.sigmoid(ir_logits)
        if self.router_select == 'min':
            route_prob = torch.minimum(rgb_prob, ir_prob)
        else:
            route_prob = torch.maximum(rgb_prob, ir_prob)
        return _strict_prefix_mask(
            route_prob,
            hard=self.router_hard,
            threshold=self.router_threshold,
        )

    def forward(self, x):
        rgb = x[:, :3]
        ir = x[:, 3:6]
        fr = self.rgb_enc(rgb)  # B,128,H/8,W/8
        fi = self.ir_enc(ir)    # B,128,H/8,W/8

        if self.luyou_en:
            route_mask = self._build_route_mask(fr, fi)
            fuse_out, y_out, ir_out = self.fusion(fr, fi, route_mask=route_mask)
        else:
            fuse_out, y_out, ir_out = self.fusion(fr, fi, route_mask=None)

        if self.fusion_mode == 'rgb':
            out = y_out
        elif self.fusion_mode == 'ir':
            out = ir_out
        elif self.fusion_mode == 'avg':
            out = 0.5 * (y_out + ir_out)
        else:
            out = fuse_out
        _, _, hin, win = x.shape
        return F.interpolate(out, size=(hin // 8, win // 8), mode='bilinear', align_corners=False)
