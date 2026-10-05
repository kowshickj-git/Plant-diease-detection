"""Segment each leaf from its background and cache (image, mask) pairs.

The raw dataset is confounded in two ways:
  * healthy leaves were shot on blue paper, diseased leaves on white paper;
  * all 1024x1024 photos are healthy, every diseased photo is 768x1024.
A classifier trained on the raw photos can therefore score ~100% by looking at
the paper colour or the aspect ratio. To force it to look at the leaf, we cut
the leaf out here and training pastes it onto random square backgrounds.

Output: cache/<subclass>/<name>.png  RGBA crop around the leaf, alpha = mask
        cache/index.csv              one row per image
"""
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
WORK_SIZE = 640  # long side used for segmentation and caching
# Original subclasses -> binary label (0 = healthy, 1 = spoiled)
SUBCLASSES = {"healthy": 0, "algal leaf": 1, "brown blight": 1}
# Not a tea-leaf sample: a photo of grass and soil with a small weed.
EXCLUDE = {"healthy/UNADJUSTEDNONRAW_thumb_23c.jpg"}


def leaf_mask(rgb: np.ndarray, whole_leaf: bool) -> np.ndarray:
    """Return a bool mask of the leaf in a photo of a leaf on plain paper.

    whole_leaf: healthy leaves are in one piece with no holes, so keep only the
    largest region and fill every hole (holes there are glare GrabCut mistook for
    paper). Diseased leaves can be torn in two and have real holes, so keep big
    secondary pieces and fill only specks smaller than 0.3% of the leaf.
    """
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV 8-bit Lab: L in [0,255], a/b offset by 128. Down-weight lightness so
    # shadows on the paper stay "paper"; chroma is what separates leaf from paper.
    feat = np.dstack([lab[..., 0] * 0.35, lab[..., 1], lab[..., 2]])

    ring = max(4, int(0.04 * min(h, w)))
    border = np.zeros((h, w), bool)
    border[:ring], border[-ring:], border[:, :ring], border[:, -ring:] = True, True, True, True
    # Background colour model: k-means on border pixels copes with lighting gradients.
    # A leaf touching the edge is a minority of the ring, so drop the smallest cluster
    # if it looks like leaf (far from the dominant one).
    bpix = feat[border].reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, lbl, centers = cv2.kmeans(bpix, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lbl.ravel(), minlength=3)
    dom = centers[counts.argmax()]
    bg_centers = [c for c, n in zip(centers, counts)
                  if n > 0.08 * len(bpix) and np.linalg.norm(c - dom) < 25]

    dist = np.min([np.linalg.norm(feat - c, axis=2) for c in bg_centers], axis=0)

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[dist > 14] = cv2.GC_PR_FGD
    gc[dist > 28] = cv2.GC_FGD
    gc[border & (dist < 10)] = cv2.GC_BGD
    if (gc == cv2.GC_FGD).sum() < 50:  # nothing clearly leaf-coloured; fall back to seeds
        gc[dist > 20] = cv2.GC_FGD
    bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgm, fgm, 6,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        pass
    fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)

    fg = ndimage.binary_opening(fg, iterations=2)
    lab_cc, n = ndimage.label(fg)
    if n == 0:
        return np.ones((h, w), bool)
    sizes = ndimage.sum(fg, lab_cc, range(1, n + 1))
    # Largest piece, plus any big secondary piece when the leaf may be torn.
    keep = np.flatnonzero(sizes >= (1.0 if whole_leaf else 0.15) * sizes.max()) + 1
    fg = np.isin(lab_cc, keep)
    fg = ndimage.binary_closing(fg, iterations=2)
    fg = _strip_blue_rim(rgb, fg)
    filled = ndimage.binary_fill_holes(fg)
    if whole_leaf:
        fg = filled
    else:
        holes, nh = ndimage.label(filled & ~fg)
        if nh:
            hsz = ndimage.sum(np.ones_like(fg), holes, range(1, nh + 1))
            small = np.flatnonzero(hsz < 0.003 * fg.sum()) + 1
            fg = fg | np.isin(holes, small)
    # Shrink 1px so no rim of the original paper colour survives compositing.
    return ndimage.binary_erosion(fg, iterations=1)


def _strip_blue_rim(rgb: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Remove blue-paper patches GrabCut left on the leaf outline.

    Any leftover blue paper would be a "healthy" cue. Only small bluish regions
    that touch the outline are removed; a leaf whose whole surface looks bluish
    (reflected sky/paper) is left alone.
    """
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    bluish = fg & (b > g + 12) & (b > r + 30)
    lab, n = ndimage.label(bluish)
    if n == 0:
        return fg
    rim = fg & ~ndimage.binary_erosion(fg, iterations=2)
    area = fg.sum()
    for i in range(1, n + 1):
        comp = lab == i
        if (comp & rim).any() and comp.sum() < 0.08 * area:
            fg = fg & ~ndimage.binary_dilation(comp, iterations=2)
    return ndimage.binary_opening(fg, iterations=1)


def group_near_duplicates(rows, threshold=0.93):
    """Give photos of the same physical leaf the same group id.

    Several leaves were photographed 2-10 times. If copies land in both the
    training and validation folds, cross-validation measures memorisation.
    Leaves are embedded with DINOv2 on a neutral background and clustered with
    complete linkage (no chaining): cosine similarity >= threshold -> same group.
    """
    from common import composite, to_tensor  # sets HF cache env vars first
    import timm
    import torch
    from scipy.cluster.hierarchy import fcluster, linkage

    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0, img_size=336).cuda().eval()
    grey = np.full((336, 336, 3), 128, np.float32)
    rng = np.random.default_rng(0)
    feats = []
    with torch.no_grad():
        for r in rows:
            im = composite(Image.open(ROOT / r["path"]), grey, rng, augment=False, leaf_frac=0.9)
            x = torch.stack([to_tensor(im), to_tensor(im.transpose(Image.FLIP_LEFT_RIGHT))]).cuda()
            f = torch.nn.functional.normalize(model(x), dim=1).mean(0)
            feats.append(torch.nn.functional.normalize(f, dim=0).cpu().numpy())
    feats = np.stack(feats).astype(np.float64)
    groups = fcluster(linkage(feats, method="complete", metric="cosine"),
                      t=1 - threshold, criterion="distance")
    # Groups never mix binary labels in practice; key by label anyway so a
    # stratified split stays well defined.
    for r, gid in zip(rows, groups):
        r["group"] = f"{r['label']}-{gid}"
    return rows


def main():
    rows = []
    for sub, label in SUBCLASSES.items():
        out_dir = CACHE / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted((ROOT / sub).glob("*.jpg")):
            rel = f"{sub}/{f.name}"
            if rel in EXCLUDE:
                continue
            im = Image.open(f).convert("RGB")
            im.thumbnail((WORK_SIZE, WORK_SIZE), Image.BICUBIC)
            rgb = np.ascontiguousarray(np.asarray(im))
            m = leaf_mask(rgb, whole_leaf=(label == 0))
            ys, xs = np.nonzero(m)
            # Crop to the leaf with a small margin: the original frame (and its
            # class-dependent aspect ratio) is thrown away.
            pad = 6
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, rgb.shape[0])
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, rgb.shape[1])
            rgba = np.dstack([rgb, m.astype(np.uint8) * 255])[y0:y1, x0:x1]
            out = out_dir / (f.stem + ".png")
            Image.fromarray(rgba, "RGBA").save(out)
            rows.append({"path": str(out.relative_to(ROOT)), "source": rel,
                         "subclass": sub, "label": label,
                         "mask_frac": round(float(m.mean()), 4)})
    rows = group_near_duplicates(rows)
    with open(CACHE / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fr = np.array([r["mask_frac"] for r in rows])
    print(f"cached {len(rows)} images (excluded {len(EXCLUDE)}); leaf area fraction "
          f"min={fr.min():.3f} median={np.median(fr):.3f} max={fr.max():.3f}")
    from collections import Counter
    sizes = Counter(r["group"] for r in rows)
    print(f"{len(sizes)} distinct-leaf groups; largest group sizes: "
          f"{sorted(sizes.values(), reverse=True)[:10]}")
    for lab in (0, 1):
        g = {r["group"] for r in rows if r["label"] == lab}
        print(f"  label {lab}: {sum(r['label'] == lab for r in rows)} photos, {len(g)} groups")


if __name__ == "__main__":
    main()
