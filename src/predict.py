"""Predict a single X-ray and produce a Grad-CAM heatmap showing which regions drove the decision.

    python src/predict.py path/to/xray.jpg
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config as C
from dataset import eval_transform, load_image, standardize_xray
from model import load_checkpoint


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = self.gradients = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "activations", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0]))

    def forward(self, x):
        """ResNet forward pass that tracks gradients only from layer4 onwards. Grad-CAM needs nothing
        earlier, so this keeps far fewer activations in memory than backpropagating through the whole network."""
        m = self.model
        with torch.no_grad():
            x = m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x)))))))
        x = x.requires_grad_()
        return m.fc(torch.flatten(m.avgpool(m.layer4(x)), 1))

    def __call__(self, x, class_idx=None, logits=None):
        self.model.zero_grad()
        if logits is None:
            logits = self.forward(x)
        if class_idx is None:
            class_idx = int(logits[0].argmax())
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def overlay(img: Image.Image, cam: np.ndarray, alpha=0.4) -> Image.Image:
    import matplotlib.cm as cm
    base = np.asarray(img.resize((C.IMG_SIZE, C.IMG_SIZE))).astype(float) / 255
    heat = cm.jet(cam)[..., :3]
    return Image.fromarray((255 * ((1 - alpha) * base + alpha * heat)).astype(np.uint8))


def describe_cam(cam: np.ndarray, gray: np.ndarray, threshold=0.5) -> dict:
    """Summarise where the Grad-CAM evidence is: position, size, number of regions, and whether it lies
    mostly on dark background (a sign the prediction is not driven by the bone)."""
    from scipy import ndimage
    hot = cam >= threshold
    ys, xs = np.nonzero(hot)
    cy, cx = ys.mean() / cam.shape[0], xs.mean() / cam.shape[1]
    vert = ["upper", "middle", "lower"][min(int(cy * 3), 2)]
    horiz = ["left", "centre", "right"][min(int(cx * 3), 2)]
    position = "centre" if (vert, horiz) == ("middle", "centre") else f"{vert} {horiz}".replace("middle ", "")
    return {"position": position,
            "area_pct": 100 * float(hot.mean()),
            "regions": int(ndimage.label(hot)[1]),
            "on_background": float((gray[hot] < 25).mean()) > 0.5}


def load_confidence_bands(metrics_file=C.OUTPUT_DIR / "test_metrics.json"):
    try:
        d = json.loads(Path(metrics_file).read_text())
        return d.get("confidence_bands"), d
    except (OSError, ValueError):
        return None, None


def explain(probs: dict, cam_info: dict, bands=None, test=None) -> str:
    """Plain-language explanation built only from quantities the model and its evaluation actually produced."""
    label = max(probs, key=probs.get)
    p = probs[label]
    level = "high" if p >= 0.9 else "moderate" if p >= 0.7 else "low"
    other = [c for c in probs if c != label][0]
    lines = [f"### Result: {label.upper()}  ({100 * p:.1f}% model score)",
             "",
             "**Why the model gave this result**",
             f"- **Score:** {100 * p:.1f}% {label} vs {100 * probs[other]:.1f}% {other}. The class above 50% is "
             f"reported, so this is a **{level}-confidence** {label} prediction"
             + (" - close to the 50% boundary, so it could easily go the other way." if level == "low" else "."),
             f"- **Where it looked (Grad-CAM):** the image features that pushed the model towards *{label}* are "
             f"concentrated in the **{cam_info['position']}** of the X-ray, covering about "
             f"{cam_info['area_pct']:.0f}% of the image in {cam_info['regions']} "
             f"region{'s' if cam_info['regions'] != 1 else ''} (red/yellow in the heatmap). "
             "Check that this is where the lesion is."]
    if cam_info["on_background"]:
        lines.append("- **Warning:** most of the highlighted area is dark background, not bone. The prediction "
                     "is probably not based on the lesion and should not be trusted.")
    if bands:
        band = next((b for b in bands if b["min"] <= p < b["max"] or (p >= 1.0 and b["max"] >= 1.0)), None)
        if band and band["n"]:
            lines.append(f"- **How reliable this confidence level has been:** on the {test['n_test']} held-out "
                         f"test X-rays, the model was correct for {100 * band['accuracy']:.0f}% of the "
                         f"{band['n']} predictions in the same {band['band']}-confidence range "
                         f"({100 * band['min']:.0f}-{100 * min(band['max'], 1.0):.0f}%).")
        if test:
            if label == "malignant":
                lines.append(f"- Overall, when the model says *malignant* it is right {100 * test['precision']:.0f}% "
                             f"of the time, and it detects {100 * test['recall']:.0f}% of malignant tumors.")
            else:
                cm = test["confusion_matrix"]
                npv = cm[0][0] / max(cm[0][0] + cm[1][0], 1)
                lines.append(f"- Overall, when the model says *benign* it is right {100 * npv:.0f}% of the time, "
                             f"but it misses {cm[1][0]} of {sum(cm[1])} malignant tumors in the test set.")
    lines += ["",
              "*What this explanation cannot tell you:* the model learned visual patterns from 1,867 BTXRD "
              "radiographs. It does not identify specific radiological signs (e.g. periosteal reaction, cortical "
              "destruction, matrix type) or the tumor subtype. This is not a diagnosis - a radiologist must "
              "review the image."]
    return "\n".join(lines)


class Predictor:
    def __init__(self, checkpoint=C.CHECKPOINT, device=None):
        self.device = device or C.get_device()
        self.model = load_checkpoint(checkpoint, self.device)
        self.model.requires_grad_(False)   # inference only: Grad-CAM needs activation gradients, not weight gradients
        self.cam = GradCAM(self.model, self.model.layer4[-1])
        self.tf = eval_transform()

    def analyze(self, img: Image.Image) -> dict:
        # same preprocessing as the training data: standardize -> 3-channel -> eval_transform
        img = standardize_xray(img).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        logits = self.cam.forward(x)
        probs = torch.softmax(logits.detach(), 1)[0].cpu().numpy()
        pred = int(probs.argmax())
        cam = self.cam(x, pred, logits=logits)
        gray = np.asarray(img.convert("L").resize((C.IMG_SIZE, C.IMG_SIZE)))
        return {"probs": {c: float(p) for c, p in zip(C.CLASSES, probs)}, "label": C.CLASSES[pred],
                "heatmap": overlay(img, cam), "cam_info": describe_cam(cam, gray)}

    def __call__(self, img: Image.Image):
        r = self.analyze(img)
        return r["probs"], r["label"], r["heatmap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    args = ap.parse_args()
    r = Predictor().analyze(load_image(args.image))
    print(explain(r["probs"], r["cam_info"], *load_confidence_bands()).replace("**", "").replace("### ", ""))
    out = C.OUTPUT_DIR / f"gradcam_{args.image.stem}.png"
    r["heatmap"].save(out)
    print("Grad-CAM saved to", out)


if __name__ == "__main__":
    main()
