"""2x2 ablation (sampler x backbone depth); checkpoints go to ablation/, not the main weights."""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from evaluate import EVAL_TRANSFORM, expected_calibration_error, metrics_at, predict
from memmap_stream import MemmapDataset
from model import get_medical_model

OUT = "ablation"
VARIANTS = {
    "none_partial":     {"sampler": "none",     "backbone": "partial"},
    "balanced_partial": {"sampler": "balanced", "backbone": "partial"},
    "none_full":        {"sampler": "none",     "backbone": "full"},
    "balanced_full":    {"sampler": "balanced", "backbone": "full"},
}
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(backbone, device):
    model = get_medical_model(num_classes=1, freeze_base=(backbone != "full"))  # partial = layer4 + head
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model.to(device), n_train


def run_variant(name, cfg, epochs, seed, device):
    seed_everything(seed)
    train_ds = MemmapDataset("train", transform=TRAIN_TRANSFORM)
    val_ds = MemmapDataset("val", transform=EVAL_TRANSFORM)
    test_ds = MemmapDataset("test", transform=EVAL_TRANSFORM)

    g = torch.Generator().manual_seed(seed)
    if cfg["sampler"] == "balanced":
        labels = train_ds.labels.ravel().astype(int)
        counts = np.bincount(labels)
        weights = (1.0 / counts)[labels]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=0,
                                                   pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, generator=g,
                                                   num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, num_workers=0)

    model, n_train = build_model(cfg["backbone"], device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    ckpt = os.path.join(OUT, f"{name}.pth")
    best_val, history = float("inf"), []
    print(f"\n=== {name}  ({cfg})  trainable params: {n_train:,} ===", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.float().view(-1).to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x).view(-1), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        model.eval()
        val_loss, probs, ys = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.float().view(-1).to(device)
                out = model(x).view(-1)
                val_loss += criterion(out, y).item()
                probs.append(torch.sigmoid(out).cpu().numpy())
                ys.append(y.cpu().numpy())
        val_loss /= len(val_loader)
        p, yv = np.concatenate(probs), np.concatenate(ys)
        val_auc = roc_auc_score(yv, p)
        scheduler.step(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auroc": float(val_auc)})
        print(f"  epoch {epoch:2d}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | val AUROC {val_auc:.4f}"
              f" | {time.time() - t0:.0f}s{'  *' if improved else ''}", flush=True)

    # same protocol as evaluate.py
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    p_val, y_val = predict(model, val_ds, device)
    p_test, y_test = predict(model, test_ds, device)
    fpr, tpr, thr = roc_curve(y_val, p_val)
    best_thr = float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))
    ece, _ = expected_calibration_error(y_test, p_test)

    result = {
        "variant": name, **cfg, "epochs": epochs, "seed": seed, "trainable_params": n_train,
        "best_val_loss": best_val,
        "test_auroc": float(roc_auc_score(y_test, p_test)),
        "test_ece": ece,
        "predicted_positive_rate_at_0.5": float((p_test >= 0.5).mean()),
        "test_at_0.5": metrics_at(y_test, p_test, 0.5),
        "selected_threshold_from_val": best_thr,
        "test_at_selected": metrics_at(y_test, p_test, best_thr),
        "history": history,
    }
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def summarise():
    rows = []
    for name in VARIANTS:
        path = os.path.join(OUT, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                rows.append(json.load(f))
    if not rows:
        print("No results yet.")
        return
    header = ("| Sampler | Backbone | Trainable params | Test AUROC | Sens @0.5 | Spec @0.5 "
              "| Pred. positive @0.5 | Sens @val-thr | Spec @val-thr | ECE |")
    lines = [header, "|" + "---|" * 10]
    for r in rows:
        a, s = r["test_at_0.5"], r["test_at_selected"]
        lines.append(
            f"| {r['sampler']} | {r['backbone']} | {r['trainable_params'] / 1e6:.1f}M | {r['test_auroc']:.3f} "
            f"| {a['sensitivity']:.3f} | {a['specificity']:.3f} | {r['predicted_positive_rate_at_0.5']:.3f} "
            f"| {s['sensitivity']:.3f} | {s['specificity']:.3f} | {r['test_ece']:.3f} |")
    table = "\n".join(lines)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(table + "\n\nTest prevalence: 0.625 (a constant 'pneumonia' predictor scores sens 1.0 / spec 0.0).\n")
    print("\n" + table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=list(VARIANTS))
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if not args.summarise:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        names = [args.only] if args.only else list(VARIANTS)
        for name in names:
            run_variant(name, VARIANTS[name], args.epochs, args.seed, device)
    summarise()


if __name__ == "__main__":
    main()
