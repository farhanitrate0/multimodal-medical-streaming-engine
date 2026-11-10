"""Raw logit check on samples/, no server involved (run extract.py first)."""
import json
import os

import torch
from PIL import Image

from evaluate import EVAL_TRANSFORM
from model import get_medical_model


def run_diagnostic(samples_dir="samples", weights="fine_tuned_medical_resnet.pth"):
    manifest_path = os.path.join(samples_dir, "samples.json")
    if not os.path.exists(manifest_path):
        print("samples/ not found. Run `python extract.py` first.")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    thr = manifest["threshold"]

    device = torch.device("cpu")
    model = get_medical_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    print(f"Decision threshold: {thr:.3f}\n")
    print(f"{'file':24s} {'truth':10s} {'logit':>8s} {'P(pneu)':>8s}  prediction")
    for s in manifest["samples"]:
        img = Image.open(os.path.join(samples_dir, s["file"])).convert("RGB")
        with torch.no_grad():
            logit = model(EVAL_TRANSFORM(img).unsqueeze(0)).item()
        p = torch.sigmoid(torch.tensor(logit)).item()
        pred = "pneumonia" if p >= thr else "normal"
        flag = "" if pred == s["ground_truth"] else "   <-- misclassified"
        print(f"{s['file']:24s} {s['ground_truth']:10s} {logit:8.3f} {p:8.3f}  {pred}{flag}")


if __name__ == "__main__":
    run_diagnostic()
