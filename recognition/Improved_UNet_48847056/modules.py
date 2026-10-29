"""
modules.py - Model components for HipMRI prostate segmentation.

Currently contains:
    UNet2D - baseline 2D U-Net (Ronneberger et al., MICCAI 2015) used as the
             slice-wise benchmark against the 3D model.

Implementation notes vs. the original paper:
    * 'same' padding (padding=1) instead of valid convolutions, so the output
      has the same spatial size as the input and no cropping is needed.
    * BatchNorm after each conv (not in the original paper) for more stable
      training from scratch.
    * Base width is configurable; the paper uses 64. 32 is used by default to
      keep VRAM low (approx. 7.8M parameters instead of approx. 31M).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two (3x3 conv -> BatchNorm -> ReLU) layers.

    This is the basic building block used at every level of the U-Net, on
    both the encoder and decoder side.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Encoder step: 2x2 max-pool (halves H and W), then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Decoder step: 2x2 transposed conv (doubles H and W), concatenate the
    matching encoder feature map (skip connection), then DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Transposed conv halves the channel count while upsampling.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concatenation with the skip: (in/2 from up) + (in/2 from skip) = in.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # If the input size was not divisible by 16, pooling floors the size,
        # so the upsampled map can be 1 pixel smaller than the skip. Pad to match.
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """Baseline 2D U-Net for multi-class slice segmentation.

    Args:
        in_channels:   number of input image channels (1 for MRI).
        num_classes:   number of output classes (6 for HipMRI: background,
                       body, bone, bladder, rectum, prostate).
        base_channels: feature width at the first level; doubles at each of
                       the 4 downsampling steps (32 -> 64 -> 128 -> 256 -> 512).

    Input:  (B, in_channels, H, W)
    Output: (B, num_classes, H, W) raw logits (apply softmax / argmax outside,
            e.g. in the loss function or predict.py).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # Encoder (contracting path)
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)  # bottleneck

        # Decoder (expanding path)
        self.up1 = Up(c * 16, c * 8)
        self.up2 = Up(c * 8, c * 4)
        self.up3 = Up(c * 4, c * 2)
        self.up4 = Up(c * 2, c)

        # 1x1 conv maps features to per-class logits
        self.out_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (used for the resource profiling table)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test: run `python modules.py` to check the model builds, produces
    # correctly shaped output, and can backpropagate. No data needed.
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 6

    model = UNet2D(in_channels=1, num_classes=num_classes, base_channels=32).to(device)
    print(f"UNet2D trainable parameters: {count_parameters(model):,}")

    # 1. Standard HipMRI slice size (256 x 128), batch of 2.
    x = torch.randn(2, 1, 256, 128, device=device)
    y = model(x)
    assert y.shape == (2, num_classes, 256, 128), f"Unexpected output shape {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "Output contains NaN/Inf"
    print(f"[OK] forward pass: input {tuple(x.shape)} -> output {tuple(y.shape)}")

    # 2. Odd size not divisible by 16 (checks the padding in Up).
    x_odd = torch.randn(1, 1, 250, 131, device=device)
    y_odd = model(x_odd)
    assert y_odd.shape == (1, num_classes, 250, 131), f"Unexpected output shape {tuple(y_odd.shape)}"
    print(f"[OK] odd-size forward pass: input {tuple(x_odd.shape)} -> output {tuple(y_odd.shape)}")

    # 3. Backward pass with cross-entropy against random integer labels,
    #    to confirm gradients reach the first layer.
    target = torch.randint(0, num_classes, (2, 256, 128), device=device)
    loss = F.cross_entropy(model(x), target)
    loss.backward()
    first_grad = model.inc.block[0].weight.grad
    assert first_grad is not None and torch.isfinite(first_grad).all(), "Gradient did not reach first layer"
    print(f"[OK] backward pass: loss = {loss.item():.4f} (approx. ln(6) = 1.79 expected at init)")

    print("All smoke tests passed.")