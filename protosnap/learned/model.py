"""Networks for the wedge detector: a plain U-Net and a U-Net with a ResNet-34 encoder."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_REG = 8   # offsets to apex, head_a, head_b, tail


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Heads(nn.Module):
    def __init__(self, cin: int, mid: int = 64):
        super().__init__()
        def head(cout):
            return nn.Sequential(nn.Conv2d(cin, mid, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(mid, cout, 1))
        self.apex, self.aux, self.reg = head(1), head(2), head(N_REG)
        for h in (self.apex, self.aux):
            h[-1].bias.data.fill_(-math.log((1 - 0.1) / 0.1))   # CenterNet prior: start at p = 0.1

    def forward(self, x):
        return {"apex": self.apex(x), "aux": self.aux(x), "reg": self.reg(x)}


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32, depth: int = 5):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth)]
        self.downs = nn.ModuleList([_block(in_ch if i == 0 else chs[i - 1], c) for i, c in enumerate(chs)])
        self.ups = nn.ModuleList([_block(chs[i] + chs[i - 1], chs[i - 1]) for i in range(depth - 1, 0, -1)])
        self.heads = Heads(chs[0])

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        return self.heads(x)


class ResUNet(nn.Module):
    """ResNet-34 encoder (ImageNet weights when available) with a U-Net decoder back to full resolution."""

    def __init__(self, in_ch: int = 1, pretrained: bool = True):
        super().__init__()
        import torchvision
        weights = None
        if pretrained:
            try:
                weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
                net = torchvision.models.resnet34(weights=weights)
            except Exception as e:   # no network: fall back to random init
                print(f"[ResUNet] could not load pretrained weights ({e}); using random init")
                net = torchvision.models.resnet34(weights=None)
                weights = None
        else:
            net = torchvision.models.resnet34(weights=None)
        self.pretrained = weights is not None
        w = net.conv1.weight.data                       # (64, 3, 7, 7)
        conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        conv1.weight.data = w.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * (3.0 / in_ch) if self.pretrained else conv1.weight.data
        self.stem = nn.Sequential(conv1, net.bn1, net.relu)     # 1/2, 64
        self.pool = net.maxpool
        self.l1, self.l2, self.l3, self.l4 = net.layer1, net.layer2, net.layer3, net.layer4   # 1/4 64, 1/8 128, 1/16 256, 1/32 512
        self.full = _block(in_ch, 32)                            # 1/1 skip
        self.u3 = _block(512 + 256, 256)
        self.u2 = _block(256 + 128, 128)
        self.u1 = _block(128 + 64, 64)
        self.u0 = _block(64 + 64, 64)
        self.uf = _block(64 + 32, 48)
        self.heads = Heads(48)

    def forward(self, x):
        f = self.full(x)
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)
        y = self.u3(up(s4, s3))
        y = self.u2(up(y, s2))
        y = self.u1(up(y, s1))
        y = self.u0(up(y, s0))
        y = self.uf(up(y, f))
        return self.heads(y)


def build_model(encoder: str, in_ch: int, base: int = 32) -> nn.Module:
    if encoder == "unet":
        return UNet(in_ch, base)
    if encoder == "resnet34":
        return ResUNet(in_ch, pretrained=True)
    raise ValueError(encoder)


# --------------------------------------------------------------------------- losses

def focal_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss. `valid` (B,1,H,W) zeroes ignored regions."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    t = target.float()
    pos = (t > 0.99).float()
    neg = 1.0 - pos
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = -torch.log(1 - p) * p ** alpha * (1 - t) ** beta * neg
    v = valid.float()
    n = pos.sum().clamp(min=1.0)
    return ((pos_loss + neg_loss) * v).sum() / n


def detector_loss(out: dict, batch: dict, w_aux: float = 0.5, w_reg: float = 2.0) -> tuple[torch.Tensor, dict]:
    la = focal_loss(out["apex"], batch["apex_hm"], batch["valid"])
    lx = focal_loss(out["aux"], batch["aux_hm"], batch["valid"])
    m = batch["reg_mask"]
    lr = (F.l1_loss(out["reg"].float(), batch["reg"], reduction="none") * m).sum() / (m.sum().clamp(min=1.0) * N_REG)
    total = la + w_aux * lx + w_reg * lr
    return total, {"apex": la.item(), "aux": lx.item(), "reg": lr.item(), "total": total.item()}
