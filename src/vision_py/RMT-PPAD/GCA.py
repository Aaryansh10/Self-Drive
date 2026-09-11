from __future__ import annotations

import torch
import torch.nn as nn



class DepthwiseSeparableConv(nn.Module):

    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=k, stride=1, padding=k // 2,
            groups=channels, bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class Adapter(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.reduce = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(channels)
        self.reduce_act = nn.SiLU(inplace=True)
        self.dwsep = DepthwiseSeparableConv(channels, k=3)

    def forward(self, f_shared: torch.Tensor) -> torch.Tensor:
        x = self.reduce_act(self.reduce_bn(self.reduce(f_shared)))
        f_task = self.dwsep(x)
        return f_task


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_task: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(f_task))
        max_out = self.mlp(self.max_pool(f_task))
        c_gate = self.sigmoid(avg_out + max_out)  # (B, C, 1, 1)
        return c_gate


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_task: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(f_task, dim=1, keepdim=True)   # (B, 1, H, W)
        max_map, _ = torch.max(f_task, dim=1, keepdim=True)  # (B, 1, H, W)
        pooled = torch.cat([avg_map, max_map], dim=1)         # (B, 2, H, W)
        s_gate = self.sigmoid(self.conv(pooled))               # (B, 1, H, W)
        return s_gate


class GCA(nn.Module):

    def __init__(self, channels: int, reduction: int = 16, sa_kernel_size: int = 7,
                 gate_min: float = 0.05, gate_max: float = 0.95):
        super().__init__()
        self.channels = channels
        self.gate_min = gate_min
        self.gate_max = gate_max

        self.adapter = Adapter(channels)
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=sa_kernel_size)

        self.fg_gate_logit = nn.Parameter(torch.zeros(1))

    @property
    def fg_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.fg_gate_logit)

    def forward(self, f_shared: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_shared.shape
        assert c == self.channels, (
            f"GCA configured for {self.channels} channels, got input with {c}"
        )

        # 1. Adapter
        f_task = self.adapter(f_shared)

        # 2. Channel Attention
        c_gate = self.channel_attention(f_task)  # (B, C, 1, 1)

        # 3. Spatial Attention
        s_gate = self.spatial_attention(f_task)  # (B, 1, H, W)

        # 4. Fusion Gate
        fg = self.fg_gate  # scalar tensor in (0, 1)
        gate = fg * c_gate + (1.0 - fg) * s_gate  # broadcasts to (B, C, H, W)
        gate = torch.clamp(gate, min=self.gate_min, max=self.gate_max)

        # 5. Residual Interpolation
        out = f_shared + gate * (f_task - f_shared)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)

    HIDDEN_DIM = 256
    B = 2
    SCALE_SHAPES = {
        "S3 (80x80)": (B, HIDDEN_DIM, 80, 80),
        "S4 (40x40)": (B, HIDDEN_DIM, 40, 40),
        "S5 (20x20)": (B, HIDDEN_DIM, 20, 20),
    }
    TASKS = ["detection", "segmentation"]

    print("=" * 78)
    print("TEST: instantiate 6 GCA modules (3 scales x 2 tasks) on dummy CCFM outputs")
    print("=" * 78)

    gca_bank = {}
    for task in TASKS:
        for scale_name in SCALE_SHAPES:
            key = f"{task}_{scale_name.split()[0]}"
            gca_bank[key] = GCA(channels=HIDDEN_DIM)

    print(f"  Instantiated {len(gca_bank)} GCA modules: {list(gca_bank.keys())}\n")

    all_passed = True
    for task in TASKS:
        for scale_name, shape in SCALE_SHAPES.items():
            key = f"{task}_{scale_name.split()[0]}"
            gca = gca_bank[key]
            f_shared = torch.randn(*shape)

            out = gca(f_shared)

            shape_ok = out.shape == f_shared.shape

            with torch.no_grad():
                f_task = gca.adapter(f_shared)
                c_gate = gca.channel_attention(f_task)
                s_gate = gca.spatial_attention(f_task)
                fg = gca.fg_gate
                gate = torch.clamp(fg * c_gate + (1.0 - fg) * s_gate, 0.05, 0.95)
            gate_ok = bool((gate.min() >= 0.05 - 1e-6) and (gate.max() <= 0.95 + 1e-6))

            status = "PASS" if (shape_ok and gate_ok) else "FAIL"
            all_passed &= (shape_ok and gate_ok)
            print(
                f"  [{status}] {key:26s} in {tuple(f_shared.shape)} -> out {tuple(out.shape)} "
                f"| gate range [{gate.min().item():.4f}, {gate.max().item():.4f}] "
                f"| FG_gate={fg.item():.4f}"
            )

    print()
    num_params_one = sum(p.numel() for p in GCA(channels=HIDDEN_DIM).parameters())
    print(f"  Params per GCA instance (C={HIDDEN_DIM}): {num_params_one / 1e3:.1f}K")
    print(f"  Params across 6 instances:                {6 * num_params_one / 1e6:.2f}M")

    assert all_passed
    print("\n  PASS: all 6 GCA modules preserve shape and respect the [0.05, 0.95] gate clamp.")
    print("=" * 78)