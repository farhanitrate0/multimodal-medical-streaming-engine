---
title: Multimodal Medical Streaming Engine
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
short_description: Explainable pneumonia screening with Grad-CAM
---

# Multimodal Medical Streaming Engine: live demo

ResNet50 fine-tuned on PneumoniaMNIST (224x224) with Grad-CAM explanations of the predicted class.
Test AUROC 0.987, sensitivity 99.0%, specificity 87.2% at a validation-selected threshold of 0.75.

Research prototype, not for clinical use. Full project, evaluation, ablations and the real-time dashboard:
https://github.com/farhanitrate0/multimodal-medical-streaming-engine
