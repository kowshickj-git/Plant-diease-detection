"""Train a healthy-vs-spoiled tea leaf classifier.

    python train.py cv    --arch convnext_tiny.in12k_ft_in1k_384   # 5-fold grouped CV
    python train.py final --arch convnext_tiny.in12k_ft_in1k_384   # train on all data

Run prepare_data.py first. Every training image is a leaf cut-out pasted onto a
random background, so the model cannot use the paper colour (blue = healthy,
white = spoiled in this dataset) or the photo's aspect ratio.

CV evaluates each held-out image three ways:
  raw      - the original photo (still has the class-coloured paper)
  swapped  - the leaf on the *other* class's paper colour (healthy on white,
             spoiled on blue); a background-cheating model collapses here
  neutral  - the leaf on plain grey
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common import (BLUE_PAPER, CLASSES, ROOT, WHITE_PAPER, _smooth_noise, build_model,
                    composite, photo_to_input, random_background, seed_everything, to_tensor)

EVAL_MODES = ["raw", "swapped", "neutral"]


def load_rows():
    rows = list(csv.DictReader(open(ROOT / "cache" / "index.csv")))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


class TrainSet(Dataset):
    def __init__(self, rows, size):
        self.rows, self.size = rows, size
        self._rng = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._rng is None:  # one generator per worker, advancing across epochs
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        r = self.rows[i]
        img = composite(Image.open(ROOT / r["path"]), random_background(self._rng, self.size),
                        self._rng, augment=True)
        return to_tensor(img), r["label"]


def paper(color, size, rng):
    shade = 1.0 + (_smooth_noise(rng, size, 3) - 0.5) * 0.2
    return np.clip(np.broadcast_to(color, (size, size, 3)) * shade, 0, 255).astype(np.float32)


class EvalSet(Dataset):
    def __init__(self, rows, size, mode):
        self.rows, self.size, self.mode = rows, size, mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rng = np.random.default_rng(i)
        if self.mode == "raw":
            img = photo_to_input(Image.open(ROOT / r["source"]), self.size)
        else:
            if self.mode == "swapped":
                bg = paper(WHITE_PAPER if r["label"] == 0 else BLUE_PAPER, self.size, rng)
            else:
                bg = np.full((self.size, self.size, 3), 128, np.float32)
            img = composite(Image.open(ROOT / r["path"]), bg, rng, augment=False, leaf_frac=0.8)
        return to_tensor(img), r["label"]


def balanced_sampler(rows):
    """Balance the two classes, and within a class weight each photo by
    1 / (photos of the same leaf) so a leaf shot 11 times is not seen 11x as often."""
    from collections import Counter
    gsize = Counter(r["group"] for r in rows)
    w = np.array([1.0 / gsize[r["group"]] for r in rows])
    for lab in (0, 1):
        idx = [i for i, r in enumerate(rows) if r["label"] == lab]
        w[idx] /= w[idx].sum()
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(rows), replacement=True)


def loader(ds, bs, sampler=None, workers=6):
    return DataLoader(ds, batch_size=bs, sampler=sampler, shuffle=False, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0, drop_last=sampler is not None)


@torch.no_grad()
def predict(model, dl, tta=True):
    model.eval()
    probs, labels = [], []
    for x, y in dl:
        x = x.cuda(non_blocking=True)
        views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)] if tta else [x]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.stack([F.softmax(model(v).float(), 1) for v in views]).mean(0)
        probs.append(p[:, 1].cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def metrics(p_spoiled, y, thr=0.5):
    pred = (p_spoiled >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "healthy_recall": float(cm[0, 0] / max(cm[0].sum(), 1)),
        "spoiled_recall": float(cm[1, 1] / max(cm[1].sum(), 1)),
        "auc": float(roc_auc_score(y, p_spoiled)) if len(set(y)) == 2 else float("nan"),
        "log_loss": float(-np.mean(np.log(np.clip(np.where(y == 1, p_spoiled, 1 - p_spoiled), 1e-6, 1)))),
        "confusion_matrix[true][pred]": cm.tolist(),
        "n": int(len(y)),
    }


def train_one(rows, args, log_prefix=""):
    model = build_model(args.arch, pretrained=True, drop_path=args.drop_path).cuda()
    head = [p for n, p in model.named_parameters() if n.startswith(("head", "classifier", "fc."))]
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.lr * 10}], weight_decay=args.wd)
    dl = loader(TrainSet(rows, args.size), args.bs, sampler=balanced_sampler(rows), workers=args.workers)
    total = args.epochs * len(dl)
    warm = max(1, int(0.06 * total))
    base = [g["lr"] for g in opt.param_groups]
    step = 0
    for ep in range(args.epochs):
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for x, y in dl:
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * max(f, 0.01)
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out.float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            step += 1
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(y)
        if ep == 0 or (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            print(f"{log_prefix}epoch {ep + 1:3d}/{args.epochs}  loss {loss_sum / seen:.4f}  "
                  f"train-acc {correct / seen:.3f}  {time.time() - t0:.1f}s", flush=True)
    return model


def background_rule_baseline(rows, size=384):
    """The shortcut, as a 'model': call a photo healthy if its border is blue."""
    out = {}
    for mode in ("raw", "swapped"):
        ds = EvalSet(rows, size, mode)
        p = []
        for i in range(len(ds)):
            x, _ = ds[i]
            a = x.numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            ring = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3)])
            med = np.median(ring, 0)
            p.append(0.0 if med[2] - med[0] > 0.1 else 1.0)
        y = np.array([r["label"] for r in rows])
        out[mode] = metrics(np.array(p), y)
    return out


def run_cv(args):
    rows = load_rows()
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    out_dir = ROOT / "runs" / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = {m: np.full(len(rows), np.nan) for m in EVAL_MODES}
    fold_of = np.full(len(rows), -1)
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    t_start = time.time()
    fold_metrics = []
    for k, (tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        if args.max_folds and k >= args.max_folds:
            break
        seed_everything(args.seed + k)
        assert not set(groups[tr]) & set(groups[va]), "leaf group leaked across folds"
        print(f"\n=== fold {k + 1}/{args.folds}: train {len(tr)} (healthy {int((y[tr] == 0).sum())}), "
              f"val {len(va)} (healthy {int((y[va] == 0).sum())}) ===", flush=True)
        model = train_one([rows[i] for i in tr], args, log_prefix=f"[f{k + 1}] ")
        fm = {}
        for m in EVAL_MODES:
            p, yy = predict(model, loader(EvalSet([rows[i] for i in va], args.size, m), 32, workers=2))
            oof[m][va] = p
            fm[m] = metrics(p, yy)
        fold_of[va] = k
        fold_metrics.append(fm)
        print("   " + "  |  ".join(f"{m}: acc {fm[m]['accuracy']:.3f} bal {fm[m]['balanced_accuracy']:.3f}"
                                   for m in EVAL_MODES), flush=True)
        del model
        torch.cuda.empty_cache()
    done = fold_of >= 0  # all rows unless --max-folds cut the run short
    summary = {
        "arch": args.arch, "size": args.size, "epochs": args.epochs, "folds": args.folds,
        "n_images": len(rows), "n_leaf_groups": int(len(set(groups))),
        "minutes": round((time.time() - t_start) / 60, 1),
        "out_of_fold": {m: metrics(oof[m][done], y[done]) for m in EVAL_MODES},
        "per_fold": fold_metrics,
        "background_rule_baseline": background_rule_baseline(rows, args.size),
    }
    (out_dir / "cv_metrics.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "oof_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "subclass", "label", "fold"] + [f"p_spoiled_{m}" for m in EVAL_MODES])
        for i, r in enumerate(rows):
            w.writerow([r["source"], r["subclass"], r["label"], fold_of[i]] + [f"{oof[m][i]:.4f}" for m in EVAL_MODES])
    print(f"\n=== out-of-fold results ({int(done.sum())} images, {len(set(groups[done]))} distinct leaves) ===")
    for m in EVAL_MODES:
        s = summary["out_of_fold"][m]
        print(f"{m:8s} acc {s['accuracy']:.4f}  bal-acc {s['balanced_accuracy']:.4f}  auc {s['auc']:.4f}  logloss {s['log_loss']:.4f}  "
              f"healthy-recall {s['healthy_recall']:.3f}  spoiled-recall {s['spoiled_recall']:.3f}  "
              f"cm {s['confusion_matrix[true][pred]']}")
    for m, s in summary["background_rule_baseline"].items():
        print(f"baseline 'blue border = healthy' on {m:8s}: acc {s['accuracy']:.4f}  bal-acc {s['balanced_accuracy']:.4f}")
    print(f"saved {out_dir / 'cv_metrics.json'}  ({summary['minutes']} min)")


def run_final(args):
    rows = load_rows()
    seed_everything(args.seed)
    print(f"training final model on all {len(rows)} images", flush=True)
    model = train_one(rows, args, log_prefix="[final] ")
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    path = out / "tea_leaf_classifier.pt"
    torch.save({"arch": args.arch, "size": args.size, "classes": CLASSES,
                "state_dict": model.state_dict(), "train_args": vars(args)}, path)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "final"])
    ap.add_argument("--arch", default="convnext_tiny.in12k_ft_in1k_384")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=0, help="stop after this many folds (quick test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    run_cv(args) if args.mode == "cv" else run_final(args)


if __name__ == "__main__":
    main()
