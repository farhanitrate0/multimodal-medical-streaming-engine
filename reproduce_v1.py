"""Re-run the original v1 recipe (28px data, 3 epochs, lr 1e-4) and test it at 28px and 224px."""
import json
import os

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from medmnist import INFO
from sklearn.metrics import roc_auc_score

from ablation import seed_everything
from evaluate import EVAL_TRANSFORM, metrics_at, predict
from model import get_medical_model

OUT = "ablation"


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DataClass = getattr(medmnist, INFO["pneumoniamnist"]["python_class"])

    # v1 bug: no size=224, so 28px images
    train_ds = DataClass(split="train", transform=EVAL_TRANSFORM, download=True)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

    model = get_medical_model(num_classes=1).to(device)  # layer4 + head, as in v1
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    for epoch in range(3):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)  # (B,1) vs (B,1), same as v1
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch {epoch + 1}/3 | train loss {running / len(loader):.4f}", flush=True)

    model.eval()  # v1 kept the last epoch, no validation
    out = {"recipe": "v1: 28px data upscaled, 3 epochs, lr 1e-4, partial backbone, natural sampling"}
    for res in (28, 224):
        test_ds = DataClass(split="test", transform=EVAL_TRANSFORM, download=True, size=res)
        p, y = predict(model, test_ds, device)
        m = metrics_at(y, p, 0.5)
        out[f"test_{res}px"] = {"auroc": float(roc_auc_score(y, p)),
                                "predicted_positive_rate": float((p >= 0.5).mean()),
                                "mean_p": float(p.mean()), **m}
        print(f"tested at {res}px: AUROC {out[f'test_{res}px']['auroc']:.3f} | "
              f"sens {m['sensitivity']:.3f} | spec {m['specificity']:.3f} | "
              f"predicted positive {np.mean(p >= 0.5):.3f}", flush=True)

    with open(os.path.join(OUT, "v1_reproduction.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
