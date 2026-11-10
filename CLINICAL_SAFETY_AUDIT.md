# Clinical Safety and Audit Documentation

## System Intended Use and Architecture
The Multimodal Medical Streaming Engine is an experimental clinical decision support tool. It is not built for primary diagnosis. 

* Radiological Diagnosis Mode: This uses a finetuned ResNet50 network to check chest radiographs for signs of pneumonia consolidation.
* Spatial Telemetry Mode: This uses spatial tracking to monitor patient movements like face mesh and hand positions. The landmarks are returned for downstream kinematic analysis and are not used for diagnosis.
* Scope Restriction: All outputs are based on probabilities and require human in the loop oversight. They absolutely do not replace a formal review by a radiologist.

## Explainability and Clever Hans Mitigation
Medical artificial intelligence needs strict transparency. We have to make sure the model looks at real anatomical pathology and does not just cheat by finding background artifacts.

* Gradient Mapping Integration: The engine uses Gradient weighted Class Activation Mapping on the final convolutional block. This highlights the image regions that most influenced the prediction, at the coarse 7x7 resolution of that block.
* Artifact Review: Example heatmaps across true and false positives and negatives are reviewed for attention outside the lung fields, such as image borders, screen bezels or hospital watermarks. This is a manual audit on a sample, not an automated per-image check, so every served heatmap should still be inspected.
* Sanity Checks: `sanity_checks.py` runs a cascading model-parameter randomization test (Adebayo et al., 2018) and a patch-deletion faithfulness test. On 100 balanced test images, heatmap rank correlation with the original falls to about 0 once the weights are randomized. Occluding Grad-CAM-ranked regions lowers predicted-class confidence faster than occluding random regions (deletion AUC 0.447 vs 0.512). These tests are evidence that the heatmaps depend on the learned weights and point at regions that matter to the decision. They do not prove the heatmaps are clinically correct.
* Observed Shortcut: at least one false positive in the test sample is driven by an image corner rather than the lung fields (see `results/gradcam_examples.png`).

## Data Privacy and Memory Architecture
The streaming engine is designed to keep patient images out of persistent storage. It has not been assessed against any regulatory standard (for example HIPAA or GDPR).

* In Memory Processing: Telemetry frames and image scans stream through WebSockets as base64 payloads and decode directly into RAM.
* No Persistent Storage: The engine does not write diagnostic frames or spatial coordinates to disk, which removes one common source of residual data. Transport encryption, access control and logging policy depend on how it is deployed.
* Hosted Demo: The public Gradio demo stores uploads in temporary files and must not be used with real patient data.

## Known Failure Modes and Limitations
* Out of Distribution Degradation: The diagnostic mode was trained exclusively on chest radiographs. If you feed it random pictures, it will produce erratic and unstable heatmaps.
* Optical Artifacts: Pointing a webcam at a physical monitor creates glare and moire patterns. These lighting issues can artificially shift the neural network focus.
* Kinematic Occlusion: Bad lighting or physical objects blocking the camera will cause the spatial mesh to temporarily lose tracking.