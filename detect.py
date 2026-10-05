"""Detect healthy / spoiled tea leaves and save annotated images.

HOW TO USE
  1. Paste the path of an image (or a folder of images) into IMAGE_PATH below.
  2. Run:   .venv\\Scripts\\python detect.py
  The annotated result opens on screen and is saved to
      test/healthy/<name>.jpg   or   test/spoiled/<name>.jpg

You can also pass paths on the command line instead of editing the file:
      .venv\\Scripts\\python detect.py "C:\\photos\\leaf1.jpg" "C:\\photos\\more_leaves"

What the boxes mean
  * Big box (green = HEALTHY, red = SPOILED): the leaf, with the model's verdict
    and confidence. The verdict comes from the trained classifier.
  * Orange boxes (spoiled leaves only): discoloured tissue (brown, yellow or
    dead patches) found by colour analysis inside the leaf. They show where the
    damage is. They are not a second opinion on the verdict.
"""

# ============ PASTE YOUR IMAGE OR FOLDER PATH HERE ============
# Paste between the quotes. Leave it empty ("") to be asked for a path when you run.
IMAGE_PATH = r"D:\PROJECT\tea sickness dataset\healthy\UNADJUSTEDNONRAW_thumb_20b.jpg"
# ==============================================================

SHOW_RESULT = True  # open the annotated image when a single image is processed

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from common import ROOT, photo_to_input, to_tensor
from predict import IMG_EXT, load
from prepare_data import leaf_mask

OUT_DIR = ROOT / "test"
WORK = 640  # analysis resolution (long side); boxes are scaled back to the photo
COLORS = {"healthy": (30, 200, 60), "spoiled": (230, 30, 30)}
AREA_COLOR = (255, 150, 0)


def find_leaf(small: np.ndarray):
    """Leaf mask at analysis resolution; the whole frame if segmentation fails
    (e.g. busy backgrounds the paper-based segmenter was not built for)."""
    m = leaf_mask(small, whole_leaf=False)
    return m if 0.01 < m.mean() < 0.95 else np.ones(small.shape[:2], bool)


def damaged_areas(small: np.ndarray, leaf: np.ndarray, max_boxes=6):
    """Boxes around brown/yellow/dead tissue inside the leaf, plus its share of the leaf.

    Healthy tea leaf tissue is clearly green (Lab a* around -15 to -25);
    lesions and dried tissue have a* near zero or positive, or are very dark.
    """
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness, a = lab[..., 0] * 100 / 255, lab[..., 1] - 128
    inner = ndimage.binary_erosion(leaf, iterations=3)  # ignore the cut-out edge
    lesion = ndimage.binary_opening(inner & ((a > -4) | (lightness < 18)), iterations=1)
    frac = float(lesion.sum() / max(leaf.sum(), 1))
    # Merge spots that sit close together into one box.
    merged = ndimage.binary_dilation(lesion, iterations=6) & leaf
    lab_cc, n = ndimage.label(merged)
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(lab_cc), start=1):
        area = (lesion[sl] & (lab_cc[sl] == i)).sum()
        if area >= 0.004 * leaf.sum():
            boxes.append((area, (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)))
    boxes.sort(key=lambda b: -b[0])
    return [b for _, b in boxes[:max_boxes]], frac


def font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_label(d, x, y, text, fill, fnt, img_w):
    l, t, r, b = d.textbbox((0, 0), text, font=fnt)
    tw, th = r - l, b - t
    pad = max(3, th // 4)
    x = min(max(0, x), max(0, img_w - tw - 2 * pad))
    y = max(0, y - th - 2 * pad)  # sit on top of the box edge
    d.rectangle((x, y, x + tw + 2 * pad, y + th + 2 * pad), fill=fill)
    d.text((x + pad - l, y + pad - t), text, fill="white", font=fnt)


def annotate(img, verdict, conf, leaf_box, areas, damaged_frac):
    img = img.convert("RGB").copy()
    w, h = img.size
    d = ImageDraw.Draw(img)
    lw = max(3, round(min(w, h) / 150))
    for b in areas:
        d.rectangle(b, outline=AREA_COLOR, width=max(2, lw - 1))
    color = COLORS[verdict]
    d.rectangle(leaf_box, outline=color, width=lw)
    text = f"{verdict.upper()} {conf:.0%}"
    draw_label(d, leaf_box[0], leaf_box[1], text, color, font(max(16, round(min(w, h) / 22))), w)
    if verdict == "spoiled":
        if areas:
            note = f"orange = damaged area (~{damaged_frac:.0%} of leaf)"
        elif damaged_frac >= 0.3:
            note = f"damage covers most of the leaf (~{damaged_frac:.0%})"
        else:
            note = "no clear discoloured patches"
        draw_label(d, 0, h, note, (60, 60, 60), font(max(12, round(min(w, h) / 40))), w)
    return img


def detect(model, size, classes, device, path: Path):
    img = Image.open(path).convert("RGB")
    x = to_tensor(photo_to_input(img, size))[None].to(device)
    with torch.no_grad():
        views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)]
        prob = torch.stack([F.softmax(model(v).float(), 1) for v in views]).mean(0)[0].cpu()
    k = int(prob.argmax())
    verdict, conf = classes[k], float(prob[k])

    small_img = img.copy()
    small_img.thumbnail((WORK, WORK), Image.BICUBIC)
    small = np.ascontiguousarray(np.asarray(small_img))
    scale = img.size[0] / small.shape[1]
    leaf = find_leaf(small)
    ys, xs = np.nonzero(leaf)
    to_photo = lambda b: tuple(int(round(v * scale)) for v in b)
    leaf_box = to_photo((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    areas, frac = [], 0.0
    if verdict == "spoiled":
        boxes, frac = damaged_areas(small, leaf)
        box_area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        # A damage box as big as the leaf box adds nothing; the note reports the share.
        areas = [to_photo(b) for b in boxes if box_area(b) < 0.8 * box_area(
            (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))]
    return annotate(img, verdict, conf, leaf_box, areas, frac), verdict, conf, prob, frac


def collect(inputs, out_dir):
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT
                            and out_dir.resolve() not in f.resolve().parents)
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"Path not found: {p}\nCheck IMAGE_PATH at the top of detect.py.")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="image files or folders (default: IMAGE_PATH)")
    ap.add_argument("--model", default=str(ROOT / "models" / "tea_leaf_classifier.pt"))
    ap.add_argument("--out", default=str(OUT_DIR), help="output folder (default: test/)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    path = IMAGE_PATH
    if not args.inputs and not path.strip():
        path = input("Paste the image or folder path and press Enter: ")
    # Explorer's "Copy as path" adds quotes; drop them.
    files = collect(args.inputs or [path.strip().strip('"').strip("'")], out)
    if not files:
        sys.exit("No images found.")
    for c in ("healthy", "spoiled"):
        (out / c).mkdir(parents=True, exist_ok=True)

    model, size, classes, device = load(args.model)
    rows, last = [], None
    for f in files:
        annotated, verdict, conf, prob, frac = detect(model, size, classes, device, f)
        # Prefix with the source folder name so files from different folders don't collide.
        dest = out / verdict / f"{f.parent.name}__{f.stem}.jpg".replace(" ", "_")
        annotated.save(dest, quality=92)
        rows.append({"image": str(f), "prediction": verdict, "confidence": f"{conf:.4f}",
                     "p_healthy": f"{prob[0]:.4f}", "p_spoiled": f"{prob[1]:.4f}",
                     "damaged_fraction": f"{frac:.3f}" if verdict == "spoiled" else "",
                     "annotated": str(dest)})
        print(f"{verdict.upper():8s} {conf:6.1%}  {f}  ->  {dest}")
        last = annotated

    n_sp = sum(r["prediction"] == "spoiled" for r in rows)
    where = rows[0]["annotated"] if len(rows) == 1 else f"{out / 'healthy'} and {out / 'spoiled'}"
    print(f"\n{len(rows)} image(s): {len(rows) - n_sp} healthy, {n_sp} spoiled. Saved to {where}")
    if len(rows) > 1:
        with open(out / "results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Summary table: {out / 'results.csv'}")
    if SHOW_RESULT and not args.no_show and len(files) == 1:
        last.show()


if __name__ == "__main__":
    main()
