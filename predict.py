"""Classify tea leaf photos as healthy or spoiled.

    python predict.py path/to/leaf.jpg
    python predict.py path/to/folder  [--csv results.csv]

Works best on one leaf per photo, roughly centred, on a plain background
(the same way the training photos were taken).
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from common import ROOT, build_model, photo_to_input, to_tensor

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load(model_path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), ckpt["size"], ckpt["classes"], device


@torch.no_grad()
def classify(model, size, device, path):
    x = to_tensor(photo_to_input(Image.open(path), size))[None].to(device)
    # Test-time augmentation: average over flips (leaves have no fixed orientation).
    views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)]
    return torch.stack([F.softmax(model(v).float(), 1) for v in views]).mean(0)[0].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("--model", default=str(ROOT / "models" / "tea_leaf_classifier.pt"))
    ap.add_argument("--csv", help="also write results to this CSV file")
    args = ap.parse_args()

    files = []
    for p in map(Path, args.inputs):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"not found: {p}")
    if not files:
        sys.exit("no images found")

    model, size, classes, device = load(args.model)
    rows = []
    for f in files:
        prob = classify(model, size, device, f)
        k = int(prob.argmax())
        rows.append({"file": str(f), "prediction": classes[k], "confidence": f"{prob[k]:.4f}",
                     "p_healthy": f"{prob[0]:.4f}", "p_spoiled": f"{prob[1]:.4f}"})
        print(f"{classes[k]:8s} {prob[k]:6.1%}  {f}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
