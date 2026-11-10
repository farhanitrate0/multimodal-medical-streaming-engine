import threading

import numpy as np
import torch


class GradCAMEngine:
    """Grad-CAM for a single-logit head; explains whichever class was predicted."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._lock = threading.Lock()  # hooks share state

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_batch(self, batch, target="predicted"):
        """Cams and logits for a whole batch in one pass."""
        with self._lock, torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(batch).view(-1)

            if target == "positive":
                sign = torch.ones_like(logits)
            elif target == "negative":
                sign = -torch.ones_like(logits)
            else:
                sign = torch.where(logits >= 0, 1.0, -1.0).detach()

            (sign * logits).sum().backward()  # samples are independent in eval mode; -z explains "normal"

            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cams = torch.relu((weights * self.activations).sum(dim=1))
            peak = cams.amax(dim=(1, 2), keepdim=True)
            cams = torch.where(peak > 0, cams / peak.clamp_min(1e-12), cams)
            return cams.detach().cpu().numpy(), logits.detach().cpu().numpy()

    def generate(self, input_tensor, target="predicted"):
        """Returns (heatmap 0-1, raw logit); target: predicted / positive / negative."""
        cams, logits = self.generate_batch(input_tensor, target)
        return cams[0], float(logits[0])

    def generate_heatmap(self, input_tensor, class_idx=0):
        """Old API."""
        heatmap, _ = self.generate(input_tensor)
        return heatmap
