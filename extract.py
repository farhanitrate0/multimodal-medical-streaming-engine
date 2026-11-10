"""Export a labelled demo set (normals, pneumonias, hardest mistakes) to samples/."""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from evaluate import EVAL_TRANSFORM
from memmap_stream import MemmapDataset
from model import get_medical_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="fine_tuned_medical_resnet.pth")
    ap.add_argument("--metrics", default=os.path.join("results", "metrics.json"))
    ap.add_argument("--out", default="samples")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--hard", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        with open(args.metrics) as f:
            thr = json.load(f)["selected_threshold_from_val"]
    except (FileNotFoundError, KeyError):
        thr = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_medical_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    raw = MemmapDataset("test")  # raw PIL
    labels = raw.labels.ravel().astype(int)

    probs = []
    with torch.no_grad():
        for i in range(len(raw)):
            x = EVAL_TRANSFORM(raw[i][0]).unsqueeze(0).to(device)
            probs.append(torch.sigmoid(model(x)).item())
    probs = np.array(probs)
    pred = (probs >= thr).astype(int)

    # most confident hits + worst misses
    tn = [i for i in np.argsort(probs) if labels[i] == 0 and pred[i] == 0][: args.per_class]
    tp = [i for i in np.argsort(-probs) if labels[i] == 1 and pred[i] == 1][: args.per_class]
    fp = [i for i in np.argsort(-probs) if labels[i] == 0 and pred[i] == 1][: args.hard]
    fn = [i for i in np.argsort(probs) if labels[i] == 1 and pred[i] == 0][: args.hard]

    manifest = []
    for group, prefix, outcome in ((tn, "normal", "true negative"), (tp, "pneumonia", "true positive"),
                                   (fp, "hardcase_fp", "false positive"), (fn, "hardcase_fn", "false negative")):
        for i in group:
            name = f"{prefix}_{i}.png"
            Image.fromarray(raw.imgs[i]).save(os.path.join(args.out, name))
            manifest.append({"file": name, "test_index": int(i),
                             "ground_truth": "pneumonia" if labels[i] else "normal",
                             "p_pneumonia": round(float(probs[i]), 4), "outcome": outcome})

    with open(os.path.join(args.out, "samples.json"), "w") as f:
        json.dump({"threshold": thr, "samples": manifest}, f, indent=2)
    for m in manifest:
        print(f"{m['file']:24s} truth={m['ground_truth']:9s} p={m['p_pneumonia']:.3f}  {m['outcome']}")


if __name__ == "__main__":
    main()
