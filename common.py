"""Shared pieces for training and inference: model, input pipeline, compositing."""
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/hf-cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import timm
import torch
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
CLASSES = ["healthy", "spoiled"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# Paper colours measured from the photo borders (see README): blue paper under
# the healthy leaves, white paper under the diseased ones.
BLUE_PAPER = np.array([150, 200, 212], np.float32)
WHITE_PAPER = np.array([226, 218, 222], np.float32)


def build_model(arch: str, pretrained: bool = True, drop_path: float = 0.1):
    return timm.create_model(arch, pretrained=pretrained, num_classes=len(CLASSES),
                             drop_path_rate=drop_path)


def to_tensor(img: Image.Image) -> torch.Tensor:
    a = (np.asarray(img.convert("RGB"), np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


# ---------------------------------------------------------------- inference input
def pad_to_square(img: Image.Image) -> Image.Image:
    """Pad a photo to a square with its own border colour (never stretch it).

    Stretching would make leaf shape depend on the photo's aspect ratio, which
    in this dataset is correlated with the class.
    """
    img = img.convert("RGB")
    w, h = img.size
    if w == h:
        return img
    a = np.asarray(img)
    ring = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3),
                           a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    fill = tuple(int(v) for v in np.median(ring, 0))
    s = max(w, h)
    out = Image.new("RGB", (s, s), fill)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def photo_to_input(img: Image.Image, size: int) -> Image.Image:
    return pad_to_square(img).resize((size, size), Image.BICUBIC)


# ---------------------------------------------------------------- compositing
def _smooth_noise(rng, size, cells):
    g = rng.random((cells, cells)).astype(np.float32)
    im = Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    return np.asarray(im, np.float32)[..., None] / 255.0


def random_background(rng: np.random.Generator, size: int) -> np.ndarray:
    """A random backdrop. Blue and white paper are drawn equally often for both
    classes so background colour carries no information about the label."""
    kind = rng.choice(["white", "blue", "solid", "gradient", "texture"],
                      p=[0.25, 0.25, 0.2, 0.15, 0.15])
    if kind in ("white", "blue"):
        base = (WHITE_PAPER if kind == "white" else BLUE_PAPER) + rng.normal(0, 12, 3)
        c = np.broadcast_to(base, (size, size, 3)).astype(np.float32)
    elif kind == "solid":
        c = np.broadcast_to(rng.uniform(0, 255, 3), (size, size, 3)).astype(np.float32)
    elif kind == "gradient":
        c0, c1 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
        t = np.linspace(0, 1, size, dtype=np.float32)
        t = t[:, None] if rng.random() < 0.5 else t[None, :]
        c = c0 + (c1 - c0) * t[..., None]
        c = np.broadcast_to(c, (size, size, 3)).astype(np.float32)
    else:
        n = _smooth_noise(rng, size, int(rng.integers(3, 24)))
        c0, c1 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
        c = c0 + (c1 - c0) * n
    # Uneven lighting, like a phone photo of paper.
    shade = 1.0 + (_smooth_noise(rng, size, 3) - 0.5) * rng.uniform(0, 0.35)
    return np.clip(c * shade + rng.normal(0, 3, (size, size, 1)), 0, 255).astype(np.float32)


def color_jitter_leaf(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric jitter on the leaf only (float32 HxWx3 in 0..255).

    Hue is kept almost fixed: green vs brown/yellow is the actual signal.
    Per-channel gain simulates white-balance differences between shoots, which
    matter here because the two classes were photographed in different light.
    """
    x = rgb / 255.0
    x = x * rng.uniform(0.9, 1.1, 3)  # white balance
    x = x * rng.uniform(0.7, 1.3)  # brightness
    m = x.mean()
    x = (x - m) * rng.uniform(0.75, 1.25) + m  # contrast
    g = x.mean(-1, keepdims=True)
    x = (x - g) * rng.uniform(0.75, 1.25) + g  # saturation
    x = np.clip(x, 0, 1) ** rng.uniform(0.8, 1.25)  # gamma
    return np.clip(x * 255, 0, 255)


def composite(leaf_rgba: Image.Image, bg: np.ndarray, rng: np.random.Generator,
              augment: bool, leaf_frac: float = 0.72) -> Image.Image:
    """Paste a leaf cut-out onto a square background.

    augment=True: random rotation/flip/scale/position, colour jitter, drop
    shadow. augment=False: upright, centred, fixed scale (for evaluation).
    """
    size = bg.shape[0]
    leaf = leaf_rgba
    if augment:
        if rng.random() < 0.5:
            leaf = leaf.transpose(Image.FLIP_LEFT_RIGHT)
        leaf = leaf.rotate(float(rng.uniform(0, 360)), resample=Image.BICUBIC, expand=True)
        leaf_frac = float(rng.uniform(0.45, 0.95))
    bbox = leaf.getchannel("A").getbbox()
    if bbox:
        leaf = leaf.crop(bbox)
    w, h = leaf.size
    s = leaf_frac * size / max(w, h)
    if augment:
        ar = float(np.exp(rng.uniform(-0.12, 0.12)))  # slight aspect jitter
        nw, nh = max(1, int(w * s * ar)), max(1, int(h * s / ar))
    else:
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
    nw, nh = min(nw, size), min(nh, size)
    leaf = leaf.resize((nw, nh), Image.BICUBIC)
    if augment:
        x0 = int(rng.integers(0, size - nw + 1))
        y0 = int(rng.integers(0, size - nh + 1))
    else:
        x0, y0 = (size - nw) // 2, (size - nh) // 2

    arr = np.asarray(leaf, np.float32)
    rgb, alpha = arr[..., :3], arr[..., 3:] / 255.0
    if augment:
        rgb = color_jitter_leaf(rgb, rng)
        # Feather the mask edge a little so the cut-out line is not a cue.
        a_img = Image.fromarray((alpha[..., 0] * 255).astype(np.uint8))
        alpha = np.asarray(a_img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.2)))),
                           np.float32)[..., None] / 255.0

    out = bg.copy()
    if augment and rng.random() < 0.5:  # drop shadow
        dx, dy = rng.integers(-12, 13, 2)
        sh = np.zeros((size, size), np.float32)
        sx, sy = x0 + dx, y0 + dy
        ys0, xs0 = max(sy, 0), max(sx, 0)
        ys1, xs1 = min(sy + nh, size), min(sx + nw, size)
        if ys1 > ys0 and xs1 > xs0:
            sh[ys0:ys1, xs0:xs1] = alpha[ys0 - sy:ys1 - sy, xs0 - sx:xs1 - sx, 0]
            sh = np.asarray(Image.fromarray((sh * 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(float(rng.uniform(2, 8)))), np.float32) / 255.0
            out *= 1 - sh[..., None] * rng.uniform(0.2, 0.5)
    region = out[y0:y0 + nh, x0:x0 + nw]
    out[y0:y0 + nh, x0:x0 + nw] = region * (1 - alpha) + rgb * alpha
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    if augment:
        if rng.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.5))))
        if rng.random() < 0.3:  # JPEG artefacts
            import io
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=int(rng.integers(40, 95)))
            img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
