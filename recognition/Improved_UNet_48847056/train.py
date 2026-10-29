"""
train.py - train, validate, test and save a 2D or 3D U-Net on HipMRI 3D volumes.

Both models use the same patient-level split (dataset.get_files_3d), and validation/
testing is always done on whole volumes for a more accurate comparison

Usage (from this folder):
    python train.py --dims 2 --epochs 30
    python train.py --root PATH/TO/HipMRI_Study_open --dims 2 --epochs 30 --batch 16
    python train.py --root PATH/TO/HipMRI_Study_open --dims 3 --epochs 60 --batch 2
Add --early-stop for a quick smoke test on a handful of volumes.
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")  # save plots without a display
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import NUM_CLASSES, get_data_3d
from modules import UNet, DiceCELoss, predict_volume

# Label index -> name 
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
PROSTATE = 5 # target
RANGPUR_ROOT = "/home/groups/comp3710/HipMRI_Study_open"  # Rangpur data path


def build(dims, base):
    return UNet(in_ch=1, num_classes=NUM_CLASSES, base=base, depth=4, dims=dims)


@torch.no_grad()
def evaluate(model, vols, labels, dims, device, amp):
    """Score whole volumes. Returns a dict with

    dice/iou: per-class, pooled over all voxels of all volumes
    vol_prostate_dice: prostate Dice for each volume separately
    slice_prostate_dice: list per volume of per-slice prostate Dice (NaN where the
        slice has no prostate in prediction or truth) - used for base/apex analysis
    sec_per_volume: mean inference time per volume

    this works for both models. predict_volume runs 2d model slice by slice and a 3d model on the complete volume, 
    so both are scored on identical voxels
    """
    model.eval()

    inter = np.zeros(NUM_CLASSES)
    psum = np.zeros(NUM_CLASSES)
    tsum = np.zeros(NUM_CLASSES)
    vol_dice, slice_dice, times = [], [], []

    for v, y in zip(vols, labels):
        x = torch.from_numpy(v).to(device)
        yt = torch.from_numpy(y.astype(np.int64)).to(device)

        # sync before and after GPU calls so the timer measures the actual computation
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device.type, enabled=amp):
            pred = predict_volume(model, x, dims).argmax(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        # accumulate overlap counts for each class (correct predictions)
        for c in range(NUM_CLASSES):
            p, t = pred == c, yt == c
            inter[c] += (p & t).sum().item()
            psum[c] += p.sum().item()
            tsum[c] += t.sum().item()

        # Prostate Dice for this patient
        p, t = pred == PROSTATE, yt == PROSTATE
        i, s = (p & t).sum().item(), p.sum().item() + t.sum().item()
        vol_dice.append(2 * i / s if s > 0 else float("nan"))

        # Prostate Dice 
        i_z = (p & t).sum((0, 1)).float()  # per slice (last axis)
        s_z = (p.sum((0, 1)) + t.sum((0, 1))).float()
        slice_dice.append(torch.where(s_z > 0, 2 * i_z / s_z, torch.nan).cpu().tolist())

    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2 * inter / (psum + tsum)
        iou = inter / (psum + tsum - inter)
    return {"dice": dice.tolist(), "iou": iou.tolist(), "vol_prostate_dice": vol_dice,
            "slice_prostate_dice": slice_dice, "sec_per_volume": float(np.mean(times))}