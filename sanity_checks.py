"""Grad-CAM sanity checks: weight randomization, patch deletion, example panel."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from evaluate import EVAL_TRANSFORM
from gradcam import GradCAMEngine
from model import get_medical_model

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
CASCADE = ["fc", "layer4", "layer3", "layer2", "layer1", "conv1"]


def load_model(weights, device):
    m = get_medical_model(pretrained=False).to(device)
    m.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return m.eval()


def reinit(module):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv2d, nn.Linear)):
            sub.reset_parameters()
        elif isinstance(sub, nn.BatchNorm2d) or isinstance(sub, nn.BatchNorm1d):
            sub.reset_parameters()
            sub.reset_running_stats()


def threshold_from_results(out):
    try:
        with open(os.path.join(out, "metrics.json")) as f:
            return json.load(f)["selected_threshold_from_val"]
    except (FileNotFoundError, KeyError):
        return 0.5


def sample_indices(labels, n, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    k = n // 2
    return np.concatenate([rng.choice(neg, k, replace=False), rng.choice(pos, n - k, replace=False)])


def randomization_test(weights, images, device, base_maps):
    torch.manual_seed(0)
    model = load_model(weights, device)
    results = []
    for name in CASCADE:
        reinit(getattr(model, name))
        cam = GradCAMEngine(model, model.layer4[-1])
        rhos = []
        for x, ref in zip(images, base_maps):
            m, _ = cam.generate(x.unsqueeze(0).to(device), target="positive")
            if m.std() == 0 or ref.std() == 0:
                rhos.append(0.0)
                continue
            rhos.append(spearmanr(ref.ravel(), m.ravel()).correlation)
        results.append({"randomized_through": name,
                        "spearman_mean": float(np.mean(rhos)),
                        "spearman_std": float(np.std(rhos))})
        print(f"  randomized through {name:7s}: Spearman rho = {np.mean(rhos):.3f} +/- {np.std(rhos):.3f}")
    return results


@torch.no_grad()
def predicted_class_prob(model, batch, sign):
    p = torch.sigmoid(model(batch).view(-1))
    return p if sign > 0 else 1 - p


PATCH = 16  # 14x14 grid of patches
GRID = 224 // PATCH


def deletion_test(model, images, maps, logits, device, seed=0):
    """Patches, not pixels: random pixel noise breaks the model no matter where it lands."""
    rng = np.random.default_rng(seed)
    cam_curves, rnd_curves = [], []
    for x, cam, logit in zip(images, maps, logits):
        sign = 1.0 if logit >= 0 else -1.0
        sal = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        patch_sal = sal.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3)).ravel()
        order_cam = np.argsort(-patch_sal)
        order_rnd = rng.permutation(patch_sal.size)
        # blur fill keeps it looking like an x-ray
        blurred = torch.from_numpy(cv2.GaussianBlur(x.permute(1, 2, 0).numpy(), (0, 0), 10)).permute(2, 0, 1)
        batch_cam, batch_rnd = [], []
        for f in FRACTIONS:
            k = int(round(f * patch_sal.size))
            for order, batch in ((order_cam, batch_cam), (order_rnd, batch_rnd)):
                grid = torch.ones(patch_sal.size)
                grid[torch.from_numpy(order[:k])] = 0
                mask = grid.view(GRID, GRID).repeat_interleave(PATCH, 0).repeat_interleave(PATCH, 1)[None]
                batch.append(x * mask + blurred * (1 - mask))
        cam_curves.append(predicted_class_prob(model, torch.stack(batch_cam).to(device), sign).cpu().numpy())
        rnd_curves.append(predicted_class_prob(model, torch.stack(batch_rnd).to(device), sign).cpu().numpy())
    return np.array(cam_curves), np.array(rnd_curves)


def denorm(x):
    img = x.permute(1, 2, 0).numpy() * STD + MEAN
    return np.clip(img, 0, 1)


def main():
    from memmap_stream import MemmapDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="fine_tuned_medical_resnet.pth")
    ap.add_argument("--n", type=int, default=100, help="number of test images (class-balanced)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = MemmapDataset("test", transform=EVAL_TRANSFORM)
    labels = test_ds.labels.ravel().astype(int)
    idx = sample_indices(labels, args.n)
    images = [test_ds[i][0] for i in idx]
    y = labels[idx]

    model = load_model(args.weights, device)
    engine = GradCAMEngine(model, model.layer4[-1])
    pred_maps, pos_maps, logits = [], [], []
    for x in images:
        t = x.unsqueeze(0).to(device)
        m, z = engine.generate(t)  # predicted class
        pred_maps.append(m)
        logits.append(z)
        pos_maps.append(engine.generate(t, target="positive")[0])
    logits = np.array(logits)

    print("1) Cascading parameter randomization")
    rand = randomization_test(args.weights, images, device, pos_maps)

    print("2) Deletion test")
    cam_c, rnd_c = deletion_test(model, images, pred_maps, logits, device)
    cam_auc = float(np.trapezoid(cam_c.mean(0), FRACTIONS))
    rnd_auc = float(np.trapezoid(rnd_c.mean(0), FRACTIONS))
    print(f"  deletion AUC  Grad-CAM = {cam_auc:.3f}   random = {rnd_auc:.3f}  (lower = more faithful)")

    summary = {
        "n_images": int(len(idx)),
        "randomization": rand,
        "deletion": {
            "fractions": FRACTIONS,
            "gradcam_mean_curve": cam_c.mean(0).tolist(),
            "random_mean_curve": rnd_c.mean(0).tolist(),
            "gradcam_auc": cam_auc,
            "random_auc": rnd_auc,
        },
    }
    with open(os.path.join(args.out, "sanity_checks.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # plots
    plt.figure(figsize=(6, 4))
    xs = ["original"] + CASCADE
    ys = [1.0] + [r["spearman_mean"] for r in rand]
    es = [0.0] + [r["spearman_std"] for r in rand]
    plt.errorbar(range(len(xs)), ys, yerr=es, marker="o", capsize=3)
    plt.xticks(range(len(xs)), xs, rotation=30)
    plt.axhline(0, color="k", lw=0.8, ls="--")
    plt.ylabel("Spearman rank correlation\nwith original Grad-CAM")
    plt.xlabel("Cascading randomization (top to bottom)")
    plt.title("Model-parameter randomization test")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "randomization.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(FRACTIONS, cam_c.mean(0), "o-", label=f"Grad-CAM order (AUC {cam_auc:.3f})")
    plt.plot(FRACTIONS, rnd_c.mean(0), "s--", label=f"Random order (AUC {rnd_auc:.3f})")
    plt.xlabel("Fraction of pixels occluded")
    plt.ylabel("Predicted-class probability")
    plt.title("Deletion test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "deletion_curve.png"), dpi=150)
    plt.close()

    thr = threshold_from_results(args.out)
    probs = 1 / (1 + np.exp(-logits))
    pred = (probs >= thr).astype(int)
    cases = {"True positive": (1, 1), "True negative": (0, 0),
             "False positive": (0, 1), "False negative": (1, 0)}
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for col, (title, (gt, pr)) in enumerate(cases.items()):
        hits = np.where((y == gt) & (pred == pr))[0]
        for row in range(2):
            ax = axes[row, col]
            ax.axis("off")
            if row >= len(hits):
                ax.set_title(f"{title}\n(none in sample)" if row == 0 else "")
                continue
            i = hits[row]
            img = denorm(images[i])
            heat = cv2.resize(pred_maps[i], (224, 224), interpolation=cv2.INTER_CUBIC)
            ax.imshow(img, cmap="gray")
            ax.imshow(heat, cmap="jet", alpha=0.4)
            ax.set_title(f"{title}\nP(pneumonia) = {probs[i]:.2f}", fontsize=10)
    fig.suptitle(f"Grad-CAM of the predicted class (threshold {thr:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "gradcam_examples.png"), dpi=150)
    plt.close()

    print(f"Saved results to {args.out}/")


if __name__ == "__main__":
    main()
