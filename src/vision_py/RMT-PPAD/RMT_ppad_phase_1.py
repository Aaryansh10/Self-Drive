

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ConvNormAct(nn.Module):
    
    def __init__(self, in_c: int, out_c: int, k: int = 1, s: int = 1, p: int | None = None,
                 act: bool = True, groups: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p,
                               groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):

    def __init__(self, c: int):
        super().__init__()
        self.conv3x3 = ConvNormAct(c, c, k=3, s=1, act=False)
        self.conv1x1 = ConvNormAct(c, c, k=1, s=1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv3x3(x) + self.conv1x1(x))


class CSPRepLayer(nn.Module):

    def __init__(self, in_c: int, out_c: int, num_blocks: int = 3, expansion: float = 0.5):
        super().__init__()
        hidden_c = int(out_c * expansion)
        self.conv1 = ConvNormAct(in_c, hidden_c, k=1, s=1)
        self.conv2 = ConvNormAct(in_c, hidden_c, k=1, s=1)
        self.bottlenecks = nn.Sequential(*[RepVggBlock(hidden_c) for _ in range(num_blocks)])
        self.conv3 = ConvNormAct(hidden_c, out_c, k=1, s=1) if hidden_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.bottlenecks(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(x1 + x2)

class ResNet18Backbone(nn.Module):
    """Standard torchvision ResNet-18 truncated to a multi-scale feature extractor.

    Returns S3 (stride 8, layer2), S4 (stride 16, layer3), S5 (stride 32, layer4).
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)  # stride 4
        self.layer1 = net.layer1  # stride 4,  64 ch
        self.layer2 = net.layer2  # stride 8,  128 ch  -> S3
        self.layer3 = net.layer3  # stride 16, 256 ch  -> S4
        self.layer4 = net.layer4  # stride 32, 512 ch  -> S5

        self.out_channels = [128, 256, 512]  # S3, S4, S5

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        s3 = self.layer2(x)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        return [s3, s4, s5]

class AIFI(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int = 8, num_layers: int = 1,
                 dim_feedforward: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._pos_cache: dict[tuple[int, int], torch.Tensor] = {}

    @staticmethod
    def _build_2d_sincos_position_embedding(w: int, h: int, embed_dim: int,
                                             temperature: float = 10000.0,
                                             device=None, dtype=None) -> torch.Tensor:
        assert embed_dim % 4 == 0, "AIFI embed_dim must be divisible by 4 for 2D sincos pos-embed"
        grid_w = torch.arange(w, device=device, dtype=dtype)
        grid_h = torch.arange(h, device=device, dtype=dtype)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, device=device, dtype=dtype) / pos_dim
        omega = 1.0 / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        pos_embed = torch.cat(
            [out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1
        )[None, :, :]  
        return pos_embed

    def _get_pos_embed(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w)
        if key not in self._pos_cache or self._pos_cache[key].device != device:
            pos = self._build_2d_sincos_position_embedding(
                w, h, self.embed_dim, device=device, dtype=dtype
            )
            self._pos_cache[key] = pos
        return self._pos_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert c == self.embed_dim, f"AIFI expected {self.embed_dim} channels, got {c}"

        src = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        pos_embed = self._get_pos_embed(h, w, x.device, x.dtype)
        src = self.encoder(src + pos_embed)

        out = src.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        return out


class CCFM(nn.Module):
    

    def __init__(self, hidden_dim: int = 256, num_csp_blocks: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.lateral_f5 = ConvNormAct(hidden_dim, hidden_dim, k=1, s=1)
        self.fuse_s4_td = CSPRepLayer(hidden_dim * 2, hidden_dim, num_blocks=num_csp_blocks)

        self.lateral_inter_s4 = ConvNormAct(hidden_dim, hidden_dim, k=1, s=1)
        self.fuse_s3_out = CSPRepLayer(hidden_dim * 2, hidden_dim, num_blocks=num_csp_blocks)

        self.downsample_s3 = ConvNormAct(hidden_dim, hidden_dim, k=3, s=2)  # 80->40
        self.fuse_s4_bu = CSPRepLayer(hidden_dim * 2, hidden_dim, num_blocks=num_csp_blocks)

        self.downsample_s4 = ConvNormAct(hidden_dim, hidden_dim, k=3, s=2)  # 40->20
        self.fuse_s5_out = CSPRepLayer(hidden_dim * 2, hidden_dim, num_blocks=num_csp_blocks)

    def forward(self, s3: torch.Tensor, s4: torch.Tensor, f5: torch.Tensor) -> List[torch.Tensor]:
        f5_lat = self.lateral_f5(f5)                          # (B, C, 20, 20)
        f5_up = self.upsample(f5_lat)                          # (B, C, 40, 40)
        inter_s4 = self.fuse_s4_td(torch.cat([f5_up, s4], dim=1))   # (B, C, 40, 40)  [saved for skip]

        inter_s4_lat = self.lateral_inter_s4(inter_s4)         # (B, C, 40, 40)
        inter_s4_up = self.upsample(inter_s4_lat)               # (B, C, 80, 80)
        out_s3 = self.fuse_s3_out(torch.cat([inter_s4_up, s3], dim=1))  # (B, C, 80, 80) FINAL

        s3_down = self.downsample_s3(out_s3)                    # (B, C, 40, 40)
        out_s4 = self.fuse_s4_bu(torch.cat([s3_down, inter_s4], dim=1))  # skip-connect inter_s4

        s4_down = self.downsample_s4(out_s4)                    # (B, C, 20, 20)
        out_s5 = self.fuse_s5_out(torch.cat([s4_down, f5], dim=1))   # skip-connect original f5

        return [out_s3, out_s4, out_s5]


class RMTPPADFeatureExtractor(nn.Module):
    
    def __init__(self, hidden_dim: int = 256, pretrained_backbone: bool = False,
                 aifi_heads: int = 8, aifi_layers: int = 1):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)
        c3, c4, c5 = self.backbone.out_channels

     
        self.proj_s3 = ConvNormAct(c3, hidden_dim, k=1, s=1)
        self.proj_s4 = ConvNormAct(c4, hidden_dim, k=1, s=1)
        self.proj_s5 = ConvNormAct(c5, hidden_dim, k=1, s=1)

        self.aifi = AIFI(embed_dim=hidden_dim, num_heads=aifi_heads, num_layers=aifi_layers)
        self.ccfm = CCFM(hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        s3, s4, s5 = self.backbone(x)

        s3 = self.proj_s3(s3)
        s4 = self.proj_s4(s4)
        s5 = self.proj_s5(s5)

        f5 = self.aifi(s5)               
        p3, p4, p5 = self.ccfm(s3, s4, f5)  

        return [p3, p4, p5]


if __name__ == "__main__":
    torch.manual_seed(0)
    HIDDEN_DIM = 256
    B = 2

    print("=" * 70)
    print("TEST 1: AIFI in isolation on a dummy S5 tensor (20x20)")
    print("=" * 70)
    dummy_s5 = torch.randn(B, HIDDEN_DIM, 20, 20)
    aifi = AIFI(embed_dim=HIDDEN_DIM, num_heads=8, num_layers=1)
    f5 = aifi(dummy_s5)
    print(f"  input : {tuple(dummy_s5.shape)}")
    print(f"  output: {tuple(f5.shape)}")
    assert f5.shape == dummy_s5.shape
    print("  PASS: AIFI preserves shape while injecting global context.\n")

    print("=" * 70)
    print("TEST 2: CCFM in isolation on dummy multi-scale tensors")
    print("=" * 70)
    dummy_s3 = torch.randn(B, HIDDEN_DIM, 80, 80)
    dummy_s4 = torch.randn(B, HIDDEN_DIM, 40, 40)
    dummy_f5 = torch.randn(B, HIDDEN_DIM, 20, 20)

    ccfm = CCFM(hidden_dim=HIDDEN_DIM)
    out_s3, out_s4, out_s5 = ccfm(dummy_s3, dummy_s4, dummy_f5)

    print(f"  out_s3: {tuple(out_s3.shape)}  (expected [{B}, {HIDDEN_DIM}, 80, 80])")
    print(f"  out_s4: {tuple(out_s4.shape)}  (expected [{B}, {HIDDEN_DIM}, 40, 40])")
    print(f"  out_s5: {tuple(out_s5.shape)}  (expected [{B}, {HIDDEN_DIM}, 20, 20])")

    assert out_s3.shape == (B, HIDDEN_DIM, 80, 80)
    assert out_s4.shape == (B, HIDDEN_DIM, 40, 40)
    assert out_s5.shape == (B, HIDDEN_DIM, 20, 20)
    print("  PASS: CCFM produces correctly shaped multi-scale outputs.\n")

    print("=" * 70)
    print("TEST 3: Full pipeline -- ResNet18 backbone -> proj -> AIFI -> CCFM")
    print("=" * 70)
    dummy_image = torch.randn(B, 3, 640, 640)
    model = RMTPPADFeatureExtractor(hidden_dim=HIDDEN_DIM, pretrained_backbone=False)
    model.eval()
    with torch.no_grad():
        p3, p4, p5 = model(dummy_image)

    print(f"  input image: {tuple(dummy_image.shape)}")
    print(f"  P3 (Scale 3): {tuple(p3.shape)}  (expected [{B}, {HIDDEN_DIM}, 80, 80])")
    print(f"  P4 (Scale 4): {tuple(p4.shape)}  (expected [{B}, {HIDDEN_DIM}, 40, 40])")
    print(f"  P5 (Scale 5): {tuple(p5.shape)}  (expected [{B}, {HIDDEN_DIM}, 20, 20])")

    assert p3.shape == (B, HIDDEN_DIM, 80, 80)
    assert p4.shape == (B, HIDDEN_DIM, 40, 40)
    assert p5.shape == (B, HIDDEN_DIM, 20, 20)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {num_params / 1e6:.2f}M")
    print("  PASS: End-to-end shared feature extraction foundation verified.")
    print("=" * 70)