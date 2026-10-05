import torch
import torch.nn as nn


CLASSES = ["octagon_sign", "barrel", "traffic_cone", "tire"]
NUM_CLASSES = len(CLASSES)


class MLP(nn.Module):
 
    def __init__(self, di, dh, do, nl=3):
        super().__init__()
        self.nl = nl
        h = [dh] * (nl - 1)
        self.ls = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip([di] + h, h + [do])
        )

    def forward(self, x):
        for i, l in enumerate(self.ls):
            x = torch.relu(l(x)) if i < self.nl - 1 else l(x)
        return x


class RTDETRHead(nn.Module):
 
    def __init__(self, d_model=256, nhead=8, num_layers=6, num_queries=300,
                 num_classes=NUM_CLASSES, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.nq = num_queries
        self.dm = d_model

        # learnable query embeddings (the "300 fixed queries")
        self.qe = nn.Embedding(self.nq, d_model)

        dl = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True,
        )
        self.dec = nn.TransformerDecoder(dl, num_layers=num_layers)

        # project flattened backbone/GCA feature map to d_model tokens
        self.ip = nn.LazyConv2d(d_model, kernel_size=1)
        self.pe = None  # positional embedding hook, wired in Phase 4 (AIFI already applies PE upstream)

        self.cls_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, nl=3)

    def forward(self, f):
        # f: (B, C, H, W) fused feature map from a GCA module
        b = f.shape[0]
        t = self.ip(f).flatten(2).permute(0, 2, 1)  # (B, HW, d_model) -> memory/tokens

        q = self.qe.weight.unsqueeze(0).expand(b, -1, -1)  # (B, 300, d_model)
        o = self.dec(tgt=q, memory=t)  # (B, 300, d_model)

        logits = self.cls_embed(o)               # (B, 300, NUM_CLASSES)
        boxes = self.bbox_embed(o).sigmoid()      # (B, 300, 4) normalized cx,cy,w,h

        return {"pred_logits": logits, "pred_boxes": boxes}


class ConvBNAct(nn.Module):
    def __init__(self, ci, co, k=3, s=1, p=1):
        super().__init__()
        self.c = nn.Conv2d(ci, co, k, s, p, bias=False)
        self.n = nn.BatchNorm2d(co)
        self.a = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.a(self.n(self.c(x)))


class SegmentationHead(nn.Module):
 
    CHANNEL_NAMES = ["white_lane", "yellow_lane", "stop_line", "pothole"]

    def __init__(self, out_scale=4, mid_ch=128):
        super().__init__()
        # out_scale: how many 2x upsample blocks to apply (e.g. 4 -> 16x)
        self.blocks = nn.ModuleList()
        ci = None  # inferred lazily on first forward via LazyConv2d below
        self.stem = nn.LazyConv2d(mid_ch, kernel_size=1)

        ups = []
        c = mid_ch
        for i in range(out_scale):
            co = max(c // 2, 16)
            ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvBNAct(c, co),
            ))
            c = co
        self.ups = nn.ModuleList(ups)

        self.out = nn.Conv2d(c, 4, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, f):
        x = self.stem(f)
        for u in self.ups:
            x = u(x)
        x = self.out(x)
        return self.act(x)  # (B, 4, H_up, W_up) in [0, 1]