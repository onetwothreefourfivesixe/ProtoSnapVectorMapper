"""Train the wedge detector and score it with the Stage 1 harness.

    python -m protosnap.learned.train --name unet_base [--encoder unet|resnet34] [--extra-channels]
                                      [--epochs 40] [--batch 16] [--lr 1e-3] [--repeats 4]

One epoch is `repeats` augmented passes over the training split. Every `--eval-every` epochs
the model is scored on the validation split through protosnap.metrics; the checkpoint with
the best objective (F1 + exact-match - corner error / 100, the same objective the classical
tuner uses) is kept. After training the decode settings are tuned on validation and the best
checkpoint is scored on every split.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import find_pairs, load_skeleton
from ..metrics import evaluate
from ..split import SPLIT_NAMES, load_splits, pairs_in_split, training_pairs
from .data import GlyphDataset
from .decode import DecodeConfig
from .model import build_model, detector_loss
from .predictor import LearnedPredictor


def objective(a) -> float:
    err = a.corner_error if a.corner_error == a.corner_error else 50.0
    return a.f1 + a.exact_rate - err / 100.0


def summarize(a) -> dict:
    return {"precision": a.precision, "recall": a.recall, "f1": a.f1, "corner_error": a.corner_error,
            "point_error": a.point_error, "exact": a.exact_rate, "objective": objective(a)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--encoder", default="unet", choices=["unet", "resnet34"])
    ap.add_argument("--extra-channels", action="store_true", help="add distance transform and skeleton input channels")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-promoted", action="store_true",
                    help="also train on skeletons promoted from reviewed model output (train-split codepoints only); val and test are unaffected")
    ap.add_argument("--root", default=".")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--report", default="reports/stage3")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = find_pairs(args.root)
    skeletons = {p.key: load_skeleton(p) for p in pairs}
    splits = load_splits(args.splits)
    by_split = {n: pairs_in_split(pairs, splits, n) for n in SPLIT_NAMES}       # original human annotations only
    train_pairs = by_split["train"]
    if args.include_promoted:
        train_pairs = training_pairs(find_pairs(args.root, include_promoted=True), splits)
        extra = [p for p in train_pairs if p.promoted]
        for p in extra:
            skeletons[p.key] = load_skeleton(p)
        from collections import Counter
        print(f"[{args.name}] training on {len(train_pairs)} glyphs: {len(train_pairs) - len(extra)} original + {len(extra)} promoted "
              f"{dict(Counter(p.font for p in extra))}")

    ds = GlyphDataset(train_pairs, skeletons, train=True, extra_channels=args.extra_channels,
                      repeats=args.repeats, seed=args.seed)
    model = build_model(args.encoder, 3 if args.extra_channels else 1, args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (len(ds) + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    ckpt_dir = Path(args.out) / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(args.report) / args.name
    rep_dir.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(args), "params": n_params, "device": device, "train_glyphs": len(train_pairs),
           "promoted_glyphs": sum(p.promoted for p in train_pairs), "epochs": []}
    best = -1e9
    print(f"[{args.name}] {args.encoder} params={n_params/1e6:.1f}M device={device} train={len(train_pairs)} glyphs "
          f"x{args.repeats}, {steps_per_epoch} steps/epoch")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ds.epoch = epoch
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                            pin_memory=device == "cuda")
        model.train()
        sums, n = {}, 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                out = model(batch["image"])
            loss, parts = detector_loss(out, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        row = {"epoch": epoch, "lr": sched.get_last_lr()[0], "train": {k: v / max(n, 1) for k, v in sums.items()},
               "elapsed": time.time() - t0}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            pred = LearnedPredictor(model, args.extra_channels, device, name=args.name)
            a = evaluate(pred, by_split["val"], "val", skeletons=skeletons).overall
            row["val"] = summarize(a)
            if row["val"]["objective"] > best:
                best = row["val"]["objective"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val": row["val"],
                            "decode": asdict(DecodeConfig())}, ckpt_dir / "best.pt")
            print(f"[{args.name}] epoch {epoch:3d} loss {row['train']['total']:.4f} (apex {row['train']['apex']:.3f} aux {row['train']['aux']:.3f} "
                  f"reg {row['train']['reg']:.4f}) | val F1 {a.f1:.3f} err {a.corner_error:.2f} exact {a.exact_rate:.3f} obj {row['val']['objective']:.3f} "
                  f"| {row['elapsed']:.0f}s", flush=True)
            model.train()
        log["epochs"].append(row)
        with open(rep_dir / "log.json", "w") as f:
            json.dump(log, f, indent=1)

    # ---- tune decode on val with the best checkpoint, then score every split
    pred = LearnedPredictor.from_checkpoint(ckpt_dir / "best.pt", device)
    best_cfg, best_obj = pred.cfg, -1e9
    for thr in (0.15, 0.2, 0.3, 0.4, 0.5):
        for snap in (0.0, 3.0, 5.0, 8.0):
            cand = pred.with_decode(threshold=thr, snap_radius=snap)
            o = objective(evaluate(cand, by_split["val"], "val", skeletons=skeletons).overall)
            if o > best_obj + 1e-9:
                best_obj, best_cfg = o, cand.cfg
    ck = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    ck["decode"] = asdict(best_cfg)
    torch.save(ck, ckpt_dir / "best.pt")
    pred = pred.with_decode(**asdict(best_cfg))
    log["decode"] = asdict(best_cfg)
    log["best_epoch"] = ck["epoch"]
    log["final"] = {}
    for name in SPLIT_NAMES:
        r = evaluate(pred, by_split[name], name, skeletons=skeletons)
        r.to_json(rep_dir / f"{name}.json")
        log["final"][name] = {"all": summarize(r.overall), **{f: summarize(a) for f, a in r.per_font.items()}}
        a = r.overall
        print(f"[{args.name}] FINAL {name:5s} P={a.precision:.3f} R={a.recall:.3f} F1={a.f1:.3f} err={a.corner_error:.2f} exact={a.exact_rate:.3f}")
    log["train_seconds"] = time.time() - t0
    with open(rep_dir / "log.json", "w") as f:
        json.dump(log, f, indent=1)
    plot_curves(log, rep_dir / "curves.png")
    print(f"[{args.name}] best epoch {ck['epoch']} decode {asdict(best_cfg)} -> {ckpt_dir / 'best.pt'}")
    return 0


def plot_curves(log: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in log["epochs"]]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k in ("total", "apex", "aux", "reg"):
        ax[0].plot(ep, [r["train"][k] for r in log["epochs"]], label=k)
    ax[0].set_yscale("log"); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ev = [r for r in log["epochs"] if "val" in r]
    ax[1].plot([r["epoch"] for r in ev], [r["val"]["f1"] for r in ev], label="F1")
    ax[1].plot([r["epoch"] for r in ev], [r["val"]["exact"] for r in ev], label="exact")
    ax[1].set_ylim(0, 1); ax[1].set_title("validation F1 / exact-match"); ax[1].set_xlabel("epoch"); ax[1].legend()
    ax[2].plot([r["epoch"] for r in ev], [r["val"]["corner_error"] for r in ev])
    ax[2].set_title("validation corner error (px)"); ax[2].set_xlabel("epoch"); ax[2].set_ylim(0, 12)
    fig.suptitle(log["args"]["name"]); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
