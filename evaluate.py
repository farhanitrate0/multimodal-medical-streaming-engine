"""Test-set evaluation: threshold picked on val, metrics + plots saved to results/."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from torchvision import transforms

from memmap_stream import MemmapDataset
from model import get_medical_model

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict(model, dataset, device, batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs, labels = [], []
    for images, y in loader:
        logits = model(images.to(device)).view(-1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.view(-1).numpy())
    return np.concatenate(probs), np.concatenate(labels).astype(int)


def expected_calibration_error(y, p, n_bins=10):
    """Calibration error over equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece, rows = 0.0, []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf, acc = p[mask].mean(), y[mask].mean()
        ece += mask.mean() * abs(acc - conf)
        rows.append((float(conf), float(acc), int(mask.sum())))
    return float(ece), rows


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float((tp + tn) / len(y)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "ppv": float(tp / (tp + fp)) if tp + fp else 0.0,
        "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_auroc_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        scores.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def plot_all(y, p, thr_metrics, ece_rows, ece, auroc, auprc, out):
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ResNet50 (AUROC = {auroc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.scatter(1 - thr_metrics["specificity"], thr_metrics["sensitivity"], c="red", zorder=3,
                label=f"Operating point (t = {thr_metrics['threshold']:.2f})")
    plt.xlabel("False positive rate (1 - specificity)")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title("ROC: PneumoniaMNIST test set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "roc_curve.png"), dpi=150)
    plt.close()

    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec, lw=2, label=f"AUPRC = {auprc:.3f}")
    plt.axhline(y.mean(), color="k", ls="--", lw=1, label=f"Prevalence = {y.mean():.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall: test set")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "pr_curve.png"), dpi=150)
    plt.close()

    conf = [r[0] for r in ece_rows]
    acc = [r[1] for r in ece_rows]
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    plt.plot(conf, acc, "o-", lw=2, label=f"Model (ECE = {ece:.3f})")
    plt.xlabel("Mean predicted P(pneumonia)")
    plt.ylabel("Observed fraction pneumonia")
    plt.title("Reliability diagram: test set")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "reliability.png"), dpi=150)
    plt.close()

    cm = thr_metrics["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        plt.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > mat.max() / 2 else "black", fontsize=14)
    plt.xticks([0, 1], ["Normal", "Pneumonia"])
    plt.yticks([0, 1], ["Normal", "Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.title(f"Confusion matrix (t = {thr_metrics['threshold']:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "confusion_matrix.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="fine_tuned_medical_resnet.pth")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_medical_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    val_ds = MemmapDataset("val", transform=EVAL_TRANSFORM)
    test_ds = MemmapDataset("test", transform=EVAL_TRANSFORM)

    p_val, y_val = predict(model, val_ds, device)
    p_test, y_test = predict(model, test_ds, device)

    # threshold from val only so test stays untouched
    fpr, tpr, thr = roc_curve(y_val, p_val)
    best_thr = float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))

    auroc = float(roc_auc_score(y_test, p_test))
    auprc = float(average_precision_score(y_test, p_test))
    ci = bootstrap_auroc_ci(y_test, p_test)
    ece, ece_rows = expected_calibration_error(y_test, p_test)

    results = {
        "weights": args.weights,
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "test_prevalence": float(y_test.mean()),
        "val_auroc": float(roc_auc_score(y_val, p_val)),
        "test_auroc": auroc,
        "test_auroc_95ci": ci,
        "test_auprc": auprc,
        "test_ece": ece,
        "selected_threshold_from_val": best_thr,
        "test_at_0.5": metrics_at(y_test, p_test, 0.5),
        "test_at_selected": metrics_at(y_test, p_test, best_thr),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_all(y_test, p_test, results["test_at_selected"], ece_rows, ece, auroc, auprc, args.out)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
