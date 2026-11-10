"""Gradio demo for Hugging Face Spaces: pneumonia screening with Grad-CAM + spatial snapshot."""
import csv
import json
import os
import tempfile

try:
    import spaces  # zerogpu; must come before torch

    def gpu(seconds):
        return spaces.GPU(duration=seconds)  # short slots stretch the visitor quota
except ImportError:
    def gpu(seconds):
        return lambda fn: fn

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from gradcam import GradCAMEngine
from model import get_medical_model

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = "fine_tuned_medical_resnet.pth"
MODEL_REPO = os.environ.get("MODEL_REPO", "Farhanitrate0/pneumonia-resnet50")
GITHUB = "https://github.com/farhanitrate0/multimodal-medical-streaming-engine"


def weights_path():
    local = os.path.join(HERE, WEIGHTS)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(MODEL_REPO, WEIGHTS)


def load_metrics():
    try:
        with open(os.path.join(HERE, "results", "metrics.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


METRICS = load_metrics()
THRESHOLD = float(METRICS.get("selected_threshold_from_val", 0.5))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_medical_model(pretrained=False)
model.load_state_dict(torch.load(weights_path(), map_location="cpu", weights_only=True))
model.eval().to(device)
cam_engine = GradCAMEngine(model, model.layer4[-1])

# same as the val transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EMPTY = "<p class='placeholder'>Awaiting input. Upload a radiograph or pick an example.</p>"


def verdict_html(prob, logit):
    pos = prob >= THRESHOLD
    tone = ("#fde4e4", "#b42332", "#ff4b4b") if pos else ("#dff3e6", "#177233", "#177233")
    title = "Pneumonia suspected" if pos else "No pneumonia detected"
    side = "above" if pos else "below"
    return f"""
    <div class="verdict" style="background:{tone[0]};color:{tone[1]}">
      <strong>{title}</strong>
      P(pneumonia) {prob:.3f} is {side} the decision threshold of {THRESHOLD:.2f}.
    </div>
    <div class="prob">
      <div class="track"><div class="fill" style="width:{100 * prob:.1f}%;background:{tone[2]}"></div>
        <div class="thr" style="left:{100 * THRESHOLD:.1f}%"></div></div>
      <div class="scale"><span>0.0 normal</span><span>pneumonia 1.0</span></div>
    </div>
    <div class="stats">
      <div><span>P(pneumonia)</span><b>{prob:.3f}</b></div>
      <div><span>Logit</span><b>{logit:.2f}</b></div>
      <div><span>Threshold</span><b>{THRESHOLD:.2f}</b></div>
    </div>"""


def overlay_on(pil, heatmap):
    rgb = np.array(pil)
    h, w = rgb.shape[:2]
    heat = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * np.clip(heat, 0, 1)), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(heat_rgb, 0.4, rgb, 0.6, 0)


MAX_FILES, CHUNK = 100, 16


def gpu_seconds(batch):
    return min(60, 10 + len(batch) // 4)


@gpu(gpu_seconds)
def gradcam_gpu(batch):
    """The only gpu call: cams + logits, 16 images per pass."""
    cams, logits = [], []
    for i in range(0, len(batch), CHUNK):
        c, z = cam_engine.generate_batch(batch[i:i + CHUNK].to(device))
        cams.extend(c)
        logits.extend(z.tolist())
    return cams, logits


def analyze(image):
    if image is None:  # no gpu slot for empty input
        return None, EMPTY
    pil = image.convert("RGB")
    cams, logits = gradcam_gpu(transform(pil).unsqueeze(0))
    prob = float(torch.sigmoid(torch.tensor(logits[0])))
    return overlay_on(pil, cams[0]), verdict_html(prob, logits[0])


def analyze_batch(paths):
    if not paths:
        return [], None, "Upload several radiographs, or load the example set.", None
    pils, names = [], []
    for p in paths[:MAX_FILES]:
        try:
            pils.append(Image.open(p).convert("RGB"))
            names.append(os.path.basename(p))
        except OSError:
            continue
    if not pils:
        return [], None, "None of the files could be read as images.", None
    cams, logits = gradcam_gpu(torch.stack([transform(p) for p in pils]))
    gallery, rows = [], []
    for pil, name, cam, logit in zip(pils, names, cams, logits):
        prob = float(torch.sigmoid(torch.tensor(logit)))
        label = "pneumonia suspected" if prob >= THRESHOLD else "normal"
        gallery.append((overlay_on(pil, cam), f"{name} · {label} · p = {prob:.3f}"))
        rows.append([name, label, round(prob, 4), round(logit, 3)])

    n_pos = sum(r[1] != "normal" for r in rows)
    summary = (f"**{len(rows)} images analyzed** · <span style='color:#b42332'>**{n_pos} pneumonia suspected**</span> · "
               f"<span style='color:#177233'>**{len(rows) - n_pos} normal**</span> · threshold {THRESHOLD:.2f}")
    out_dir = tempfile.mkdtemp()  # per request, so users don't overwrite each other
    csv_path = os.path.join(out_dir, "batch_results.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows([["file", "prediction", "p_pneumonia", "logit"], *rows])
    return gallery, rows, summary, csv_path


# spatial (cpu, no gpu quota)
try:
    from tracking import SpatialTracker, draw
    tracker = SpatialTracker(video=False)
except Exception:
    tracker = None


def track(frame):
    if frame is None:
        return None, "Take a snapshot or upload a photo."
    rgb = frame[..., :3]  # drop alpha if present
    face, hands = tracker.process(rgb)
    out = draw(rgb.copy(), face, hands)  # rgb in, rgb out
    if not face and not hands:
        return out, ("**No face or hands found.** Face the camera from about an arm's length, "
                     "with even lighting, and keep your hand fully in frame.")
    return out, f"**Face:** {'detected' if face else 'none'} · **Hands:** {len(hands)} · **Landmarks:** {len(face) + 21 * len(hands)}"


def pct(v):
    return "-" if v is None else f"{100 * v:.1f}%"


sel = METRICS.get("test_at_selected", {})
HEADER = f"""
<div class="hero">
  <h1>Multimodal Medical Streaming Engine</h1>
  <p class="caption">Explainable pneumonia screening with Grad-CAM · research prototype, not for clinical use ·
     <a href="{GITHUB}" target="_blank">GitHub</a></p>
  <div class="metrics">
    <div><span>Model</span><b>ResNet50</b><small>fine-tuned on PneumoniaMNIST</small></div>
    <div><span>Test AUROC</span><b>{METRICS.get('test_auroc', 0):.3f}</b><small>n = {METRICS.get('n_test', '-')}</small></div>
    <div><span>Sensitivity / Specificity</span><b>{pct(sel.get('sensitivity'))} / {pct(sel.get('specificity'))}</b><small>at the decision threshold</small></div>
    <div><span>Decision threshold</span><b>{THRESHOLD:.2f}</b><small>selected on validation</small></div>
  </div>
</div>"""

FOOTER = f"""<p class="footer">Experimental decision-support research software. Outputs are probabilistic and must be
reviewed by a qualified clinician. The full real-time dashboard (live camera tracking, WebSocket streaming) runs locally
with <code>docker compose up</code>; see <a href="{GITHUB}" target="_blank">the repository</a>.</p>"""

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto; font-family: 'Source Sans 3', sans-serif; }
.hero h1 { font-size: 2.3rem; font-weight: 700; color: #1f2a3c; margin: 8px 0 0; }
.hero .caption { color: #6b7280; margin: 4px 0 18px; }
.hero .caption a { color: #ff4b4b; }
.metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 20px; padding: 18px 0; border-top: 1px solid #d9dde4; border-bottom: 1px solid #d9dde4; }
.metrics span { display: block; font-size: .85rem; color: #262730; }
.metrics b { display: block; font-size: 1.7rem; font-weight: 400; color: #1f2a3c; line-height: 1.3; }
.metrics small { color: #6b7280; font-size: .78rem; }
.placeholder { font-style: italic; color: #6b7280; }
.verdict { padding: 14px 16px; border-radius: 8px; font-size: .95rem; }
.verdict strong { display: block; font-size: 1.05rem; }
.prob { margin: 16px 0 6px; }
.prob .track { position: relative; height: 8px; border-radius: 4px; background: #e6e9ef; }
.prob .fill { position: absolute; inset: 0 auto 0 0; border-radius: 4px; }
.prob .thr { position: absolute; top: -5px; bottom: -5px; width: 2px; background: #1f2a3c; }
.prob .scale { display: flex; justify-content: space-between; font-size: .8rem; color: #6b7280; margin-top: 6px; }
.stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 10px; }
.stats span { display: block; font-size: .8rem; color: #6b7280; }
.stats b { font-family: ui-monospace, Consolas, monospace; font-size: 1.05rem; color: #1f2a3c; }
.footer { color: #6b7280; font-size: .82rem; border-top: 1px solid #d9dde4; padding-top: 14px; margin-top: 8px; }
.footer a { color: #ff4b4b; }
@media (max-width: 800px) { .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
"""

# force light mode
FORCE_LIGHT = """() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); }
}"""

theme = gr.themes.Default(
    primary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Source Sans 3"), "sans-serif"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#f0f2f6",
    block_background_fill="#ffffff",
    block_border_color="#dfe3ea",
    button_primary_background_fill="#ff4b4b",
    button_primary_background_fill_hover="#e63e3e",
    button_primary_text_color="#ffffff",
)

SAMPLES = [
    ("normal_106.png", "Normal (correct)"),
    ("normal_86.png", "Normal (correct)"),
    ("pneumonia_230.png", "Pneumonia (correct)"),
    ("pneumonia_336.png", "Pneumonia (correct)"),
    ("hardcase_fp_103.png", "Hard case: normal called pneumonia"),
    ("hardcase_fn_88.png", "Hard case: missed pneumonia"),
]

with gr.Blocks(title="Multimodal Medical Streaming Engine", theme=theme, css=CSS, js=FORCE_LIGHT) as demo:
    gr.HTML(HEADER)
    with gr.Tab("Radiological diagnosis"):
        with gr.Row(equal_height=False):
            with gr.Column():
                xray = gr.Image(type="pil", label="Chest radiograph", sources=["upload", "clipboard"], height=380)
                gr.Markdown("Analysis runs as soon as an image is uploaded or an example is picked.")
                gr.Examples(
                    examples=[[os.path.join(HERE, "samples", f)] for f, _ in SAMPLES],
                    example_labels=[label for _, label in SAMPLES],
                    inputs=[xray],
                    label="Test-set examples (hard cases are deliberate model errors)",
                )
            with gr.Column():
                cam_out = gr.Image(label="Grad-CAM attention for the predicted class", interactive=False, height=380)
                verdict = gr.HTML(EMPTY)
        xray.change(analyze, inputs=xray, outputs=[cam_out, verdict])  # one gpu call per image

    with gr.Tab("Batch screening"):
        gr.Markdown(f"Upload up to {MAX_FILES} radiographs. They run through Grad-CAM {CHUNK} at a time in one "
                    "forward/backward pass; click any result to enlarge it.")
        with gr.Row():
            files = gr.File(file_count="multiple", file_types=["image"], type="filepath", label="Radiographs", scale=3)
            with gr.Column(scale=1):
                load_examples = gr.Button("Load the 10 example images")
                run_batch = gr.Button("Run batch", variant="primary")
        batch_summary = gr.Markdown()
        gallery = gr.Gallery(label="Grad-CAM per image", columns=5, height="auto", object_fit="contain")
        with gr.Row():
            table = gr.Dataframe(headers=["file", "prediction", "p_pneumonia", "logit"], label="Results", scale=3)
            csv_out = gr.File(label="Download CSV", scale=1)
        load_examples.click(lambda: [os.path.join(HERE, "samples", f) for f in sorted(os.listdir(os.path.join(HERE, "samples")))
                                     if f.endswith(".png")], outputs=files)
        run_batch.click(analyze_batch, inputs=files, outputs=[gallery, table, batch_summary, csv_out])

    if tracker is not None:
        with gr.Tab("Spatial kinematics (snapshot)"):
            gr.Markdown("Face mesh and hand landmarks from a single frame. Start the camera, press the capture button, "
                        "and tracking runs automatically (or upload a photo). Small or distant faces are found with a "
                        "full-range detector and re-meshed on a zoomed crop. Live 5 FPS tracking is available in the "
                        "full dashboard (`docker compose up`).")
            with gr.Row():
                snap = gr.Image(type="numpy", sources=["webcam", "upload"], label="Snapshot", height=360)
                tracked = gr.Image(label="Landmarks", interactive=False, height=360)
            summary = gr.Markdown()
            snap.change(track, inputs=snap, outputs=[tracked, summary])

    gr.HTML(FOOTER)

if __name__ == "__main__":
    demo.launch()
