"""Gradio web prototype with local email/password registration and model analysis.

Run from the project root with run_demo.bat, or from src with: python app.py
Open http://127.0.0.1:7860
"""
import sys
import re
import json
import sqlite3
import hashlib
import secrets
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, UnidentifiedImageError

import config as C
from dataset import standardize_xray
from predict import Predictor, explain, load_confidence_bands

MIN_SIDE = 64
READ_ERROR = "Could not read this file as an image. Please upload a PNG, JPEG, BMP or TIFF X-ray."
DB_PATH = C.OUTPUT_DIR / "users.sqlite3"
C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    predictor = Predictor(C.CHECKPOINT)
except (FileNotFoundError, ValueError) as e:
    sys.exit(f"Cannot start the demo: {e}")
print(f"Loaded model {C.CHECKPOINT} on {predictor.device}")
bands, test_metrics = load_confidence_bands()


def _connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, salt TEXT NOT NULL, password_hash TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    conn.commit()
    return conn


def _password_hash(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 200_000).hex()


def register_user(email, password, confirm):
    email = (email or "").strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        return "Please enter a valid email address."
    if not password or len(password) < 8:
        return "Password must contain at least 8 characters."
    if password != confirm:
        return "Passwords do not match."
    salt = secrets.token_hex(16)
    conn = _connect_db()
    try:
        conn.execute("INSERT INTO users(email, salt, password_hash) VALUES (?, ?, ?)", (email, salt, _password_hash(password, salt)))
        conn.commit()
    except sqlite3.IntegrityError:
        return "An account with this email already exists. Please sign in."
    finally:
        conn.close()
    return "Account created. Go to the Sign In tab and log in."


def login_user(email, password):
    email = (email or "").strip().lower()
    conn = _connect_db()
    try:
        row = conn.execute("SELECT salt, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    finally:
        conn.close()
    if not row or not secrets.compare_digest(_password_hash(password or "", row[0]), row[1]):
        return False, "Email or password is incorrect.", gr.update(visible=True), gr.update(visible=False), ""
    return True, f"Signed in as {email}", gr.update(visible=False), gr.update(visible=True), email


def logout_user():
    return False, gr.update(visible=True), gr.update(visible=False), ""


def classify(path):
    if path is None:
        raise gr.Error("Please upload an X-ray image first.")
    try:
        img = Image.open(path)
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise gr.Error(READ_ERROR)
    if min(img.size) < MIN_SIDE:
        raise gr.Error(f"Image is too small ({img.width}x{img.height} px). Please upload an X-ray at least {MIN_SIDE}x{MIN_SIDE} px.")
    if np.asarray(standardize_xray(img)).std() < 2:
        raise gr.Error("The image appears blank (no visible detail). Please upload an X-ray.")
    try:
        result = predictor.analyze(img)
    except Exception as e:
        raise gr.Error(f"Prediction failed: {e}")
    return result["probs"], result["heatmap"], explain(result["probs"], result["cam_info"], bands, test_metrics)


def make_analysis():
    if not test_metrics:
        return "Evaluation metrics are not available. Check outputs/test_metrics.json."
    m = test_metrics
    cm = m["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)
    return f'''## Model evaluation: held-out test set

The test set contains **{m['n_test']} X-rays** that were kept separate from training. The results describe this dataset and do not guarantee performance in another hospital or on a new patient.

| Metric | Result | Meaning |
|---|---:|---|
| Accuracy | **{m['accuracy']*100:.1f}%** | {tn+tp} of {m['n_test']} test images were classified correctly. |
| Malignant precision | **{m['precision']*100:.1f}%** | Of images predicted malignant, this proportion really were malignant in the test labels. |
| Malignant recall / sensitivity | **{m['recall']*100:.1f}%** | The model found {tp} of {tp+fn} malignant cases; it missed {fn}. |
| Malignant F1-score | **{m['f1']*100:.1f}%** | Balance between malignant precision and recall. |
| ROC-AUC | **{m['roc_auc']:.3f}** | Measures how well scores rank malignant cases above benign ones across thresholds; it is not accuracy. |
| Benign specificity | **{specificity*100:.1f}%** | Correctly identified {tn} of {tn+fp} benign cases. |
| Benign predictive value | **{npv*100:.1f}%** | Of benign predictions, this proportion were benign in this test set. |

### Confusion matrix (actual rows, predicted columns)

| Actual class | Predicted benign | Predicted malignant |
|---|---:|---:|
| Benign | **{tn}** correct | {fp} false alarms |
| Malignant | {fn} missed malignant cases | **{tp}** correct |

### What the results mean

- The overall accuracy is high, but accuracy alone hides class imbalance: there are many more benign than malignant images.
- **The most important safety limitation is the {fn} malignant cases missed out of {tp+fn}.** A benign result cannot rule out cancer.
- There are {fp} benign cases incorrectly flagged as malignant. Such false alarms can cause concern and need expert review.
- Augmentation comparison in the project files: the augmented model achieved F1 {m['f1']:.3f} and AUC {m['roc_auc']:.3f}; the no-augmentation version achieved F1 0.639 and AUC 0.901. This is a comparison from one training run, not proof that the improvement will always repeat.

### How to read a single-image result

1. **Prediction scores:** these are model softmax scores, not calibrated medical probabilities. The larger score determines the displayed class.
2. **Grad-CAM:** red/yellow regions indicate image areas that influenced the model. It is an explanation aid, not a tumor boundary and not proof that a highlighted region is cancer.
3. **Confidence band:** the per-image explanation compares this score range with the model's observed accuracy in that range on the test set. Small groups, dataset differences and score calibration limit how much this can tell you about a new image.
4. **Image suitability:** this model was trained on tumor X-rays for benign-versus-malignant classification. It is not designed to identify normal X-rays or tumor subtypes and may still output a score for unrelated images.

### Limitations to mention in your project viva

- Only 342 malignant images were available for the full tumor dataset, and only 51 malignant images were in the test set.
- The dataset does not provide patient IDs, so multiple views from the same patient may cross the train/test split and make the results optimistic.
- Results are from one dataset and one training run; external hospital validation and repeated-seed experiments are still needed.
- This is a research prototype, **not a medical device or diagnostic tool**. A qualified radiologist must assess the image and clinical context.
'''

def make_consultation_guide(age, pain_location, duration, symptoms, prior_imaging, injury, change_over_time):
    # Create a visit-preparation guide, not a diagnosis or triage decision.
    age = (age or "Not provided").strip()
    pain_location = pain_location or "Not specified"
    duration = duration or "Not specified"
    symptoms = symptoms or []
    prior_imaging = prior_imaging or "Not provided"
    injury = injury or "Not provided"
    change_over_time = change_over_time or "Not provided"
    symptom_text = ", ".join(symptoms) if symptoms else "No symptoms selected"
    urgent = any(x in symptoms for x in ["Sudden severe pain", "Unable to bear weight/use limb", "Rapidly increasing swelling"])
    urgent_note = ("**Seek urgent in-person medical assessment today** if you have sudden severe pain, cannot bear weight/use the limb, or rapidly increasing swelling—especially after an injury. If symptoms are severe or you suspect a fracture, use local emergency services."
                   if urgent else "If symptoms are severe, rapidly worsening, or you cannot use/bear weight on the limb, seek urgent in-person care rather than waiting for an online review.")
    return f'''## Your radiology / doctor visit preparation guide

This guide organises the information you entered so you can discuss it with a qualified clinician. **It does not diagnose a tumour, interpret your scan, or determine how urgent your case is.**

### Summary to tell the clinician
- **Age:** {age}
- **Area of concern:** {pain_location}
- **How long:** {duration}
- **Symptoms selected:** {symptom_text}
- **Previous X-ray / MRI / CT / report:** {prior_imaging}
- **Recent injury:** {injury}
- **Change over time:** {change_over_time}

### Who can help?
- Start with your treating doctor or an orthopaedic clinician for persistent bone pain, swelling, or an abnormal X-ray.
- A **radiologist** interprets medical images and produces a report; bring prior images and reports so they can compare them.
- If imaging raises concern for a bone tumour, ask your doctor whether referral to an **orthopaedic oncology / bone tumour specialist** is appropriate. Do not arrange a biopsy yourself; the specialist team should plan it if needed.

### Questions you can ask
1. What exactly does the radiology report describe, and what remains uncertain?
2. Does the finding look benign, suspicious, or indeterminate—and what features support that wording?
3. Is MRI, CT, comparison with older images, or another test recommended? What is the reason and timeframe?
4. Do I need referral to an orthopaedic specialist or a bone tumour centre?
5. If a biopsy is being considered, who should plan it and how will the result change management?
6. What symptoms should prompt earlier or urgent reassessment?

### What to bring
- The original X-ray / MRI / CT images (not only phone screenshots), image disc or download link if available
- All radiology reports and previous scans for comparison
- A list of symptoms, when they began, what makes them better/worse, medicines, and relevant medical history
- Referral letter and questions you want answered

### Safety note
{urgent_note}

The AI classifier in this student project only scores an image as benign or malignant based on its training data. Its output is **not a radiologist's report**, cannot explain the cause of symptoms, and must not be used to rule out cancer or choose treatment. Please discuss the image and symptoms with a qualified clinician.'''

CUSTOM_CSS = """
:root { --ink:#142b43; --teal:#087f78; --mint:#d9f7f0; --paper:#f3f6fa; --muted:#52657a; --line:#dce5ed; }
.gradio-container { width: min(1240px, calc(100% - 32px)) !important; max-width: 1240px !important; margin: 24px auto !important; background: transparent !important; font-family: Inter, "Segoe UI", system-ui, sans-serif !important; color:var(--ink) !important; }
body { background:var(--paper) !important; }
footer { display:none !important; }
#login-shell, #app-shell { background:#fff !important; border:1px solid var(--line) !important; border-radius:22px !important; overflow:hidden; box-shadow:0 16px 48px rgba(20,43,67,.08) !important; padding:20px !important; }
#login-title { background:linear-gradient(120deg,#102a43 0%,#164e63 58%,#087f78 100%) !important; color:#fff !important; padding:30px 34px !important; border-radius:16px !important; margin-bottom:22px !important; }
#login-title h1 { color:#fff !important; font-size:clamp(30px,3vw,42px) !important; font-weight:750 !important; letter-spacing:-1px !important; line-height:1.15 !important; margin:10px 0 8px !important; }
#login-title p, #login-title .prose p { color:#e7f8f5 !important; font-size:16px !important; line-height:1.6 !important; margin:0 !important; }
#eyebrow { color:#86f0d9 !important; text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:800; line-height:1.5; }
#login-shell .row { gap:28px !important; align-items:flex-start !important; }
#login-shell .prose, #app-shell .prose { color:#243b53 !important; font-size:15px !important; line-height:1.7 !important; }
#login-shell h2, #login-shell h3, #app-shell h2, #app-shell h3 { color:#142b43 !important; line-height:1.3 !important; }
#login-shell .tabs, #login-shell .tab-nav { border-color:var(--line) !important; }
#login-shell .tab-nav button { font-size:14px !important; font-weight:650 !important; padding:12px 14px !important; }
#login-shell label, #app-shell label { color:#314960 !important; font-size:13px !important; font-weight:650 !important; }
#login-shell input, #app-shell input, #login-shell textarea, #app-shell textarea { border:1px solid #d4dee8 !important; border-radius:10px !important; min-height:46px !important; background:#fff !important; color:#142b43 !important; box-shadow:none !important; }
#login-shell input:focus, #app-shell input:focus { border-color:#087f78 !important; box-shadow:0 0 0 3px rgba(8,127,120,.12) !important; }
button.primary { background:#087f78 !important; border:1px solid #087f78 !important; border-radius:10px !important; color:white !important; font-weight:700 !important; min-height:46px !important; }
button.primary:hover { background:#066b65 !important; }
#warning-card { background:#fff8e8; border:1px solid #f0d69b; border-radius:12px; padding:15px 18px; color:#654b12 !important; line-height:1.55 !important; }
#warning-card b { color:#654b12 !important; }
#brand-banner { background:linear-gradient(120deg,#102a43 0%,#164e63 58%,#087f78 100%); color:white; padding:28px 32px; border-radius:16px; margin-bottom:16px; }
#brand-banner h1 { color:white !important; font-size:clamp(28px,3vw,38px) !important; letter-spacing:-.7px; line-height:1.2; margin:10px 0; }
#brand-banner p, #brand-banner .prose p { color:#e7f8f5 !important; font-size:15px !important; line-height:1.6 !important; }
#metric-card { background:white; border:1px solid var(--line); border-radius:14px; padding:16px; min-height:92px; }
#metric-card h3 { color:#52657a !important; font-size:12px; text-transform:uppercase; letter-spacing:.8px; margin:0; }
#metric-card strong { color:#102a43 !important; font-size:28px; }
#soft-card { background:white; border:1px solid var(--line); border-radius:15px; padding:18px; }
.tab-nav button { font-weight:650 !important; white-space:normal !important; }
.prose table { width:100%; border-collapse:collapse; }
.prose th { background:#edf5f6; color:#142b43 !important; }
.prose th, .prose td { padding:9px 11px !important; border:1px solid #dce5ed !important; text-align:left; }
img { max-width:100%; }
@media (max-width: 800px) {
  .gradio-container { width:calc(100% - 16px) !important; margin:8px auto !important; }
  #login-shell, #app-shell { padding:12px !important; border-radius:16px !important; }
  #login-title { padding:22px 20px !important; }
  #login-shell .row { gap:14px !important; }
  #brand-banner { padding:22px 20px !important; }
}
"""

TUMOR_EDUCATION = r"""## Bone tumours: the essentials

A **bone tumour** is an abnormal growth of cells in or around a bone. Some are **benign** (non-cancerous) and remain local; others are **malignant** (cancerous) and can invade nearby tissue or spread. A tumour is not automatically cancer.

### Why do bone tumours develop?

Often there is no single identifiable cause. Changes in cell DNA can allow cells to grow abnormally. Some rare inherited conditions, previous radiation exposure, and certain underlying bone disorders can raise risk for particular tumour types. Many people diagnosed with a bone tumour have no known risk factor. Ordinary injuries do not generally cause a bone tumour, although an injury may draw attention to symptoms.

### Symptoms that may need medical assessment

- Persistent or worsening bone pain, including pain at night
- Swelling or a lump near a bone
- Reduced movement or difficulty using a limb
- A fracture after relatively minor force

These symptoms can also have other causes. A clinician may consider examination, X-rays, MRI/CT, blood tests, and sometimes a biopsy. The correct work-up depends on the case.

### Benign vs malignant — a simplified distinction

| Feature | Benign | Malignant |
|---|---|---|
| General meaning | Non-cancerous growth | Cancerous growth |
| Behaviour | Often stays local; behaviour varies by type | Can invade tissue and may spread |
| Assessment | Depends on tumour type, size, location and symptoms | Specialist assessment is important |

**Important:** this table is an educational simplification. Imaging alone does not always establish the diagnosis; radiology findings and clinical history must be interpreted by qualified professionals.

### What this project does — and does not do

This research prototype uses a ResNet50-based image classifier to score a supplied image as **benign** or **malignant** and generates a Grad-CAM heatmap to visualise image regions that influenced the score. It does not identify the exact tumour subtype, determine the cause, stage a cancer, or recommend treatment. It was trained on a curated dataset of tumour radiographs and is not designed to screen normal X-rays or every possible image type.
"""

with gr.Blocks(title="Bone Tumor Classification | Research & Consultation Guide", theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue", neutral_hue="slate"), css=CUSTOM_CSS) as demo:
    authenticated = gr.State(False)
    user_email = gr.State("")

    login_page = gr.Column(visible=True, elem_id="login-shell")
    with login_page:
        gr.Markdown("<div id='login-title'><div id='eyebrow'>AI RESEARCH LAB · FINAL YEAR PROJECT</div><h1>Bone Tumor Classification</h1><p>ResNet50 deep learning with Grad-CAM visual explanations</p></div>")
        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                gr.Markdown("""### Bone X-ray research and consultation preparation

Explore bone-tumour X-ray classification, visual explanations, model evaluation, and a guided checklist to prepare for a radiologist or doctor visit.

- **Image analysis** — benign vs malignant model scores
- **Visual explanation** — Grad-CAM overlay
- **Evaluation** — metrics, confusion matrix and error analysis
- **Doctor visit guide** — organise symptoms, reports and questions for your clinician

<div id='warning-card'><b>Research use only</b><br>This is not a diagnostic service. Do not use its output to make medical decisions.</div>""")
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Sign in"):
                        login_email = gr.Textbox(label="Email address", placeholder="student@example.com")
                        login_password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
                        login_button = gr.Button("Sign in to workspace", variant="primary", size="lg")
                        login_message = gr.Markdown()
                    with gr.Tab("Create account"):
                        register_email = gr.Textbox(label="Email address", placeholder="student@example.com")
                        register_password = gr.Textbox(label="Create password · 8+ characters", type="password")
                        register_confirm = gr.Textbox(label="Confirm password", type="password")
                        register_button = gr.Button("Create local account", variant="primary", size="lg")
                        register_message = gr.Markdown()
                gr.Markdown("<small>Local educational demo: accounts are stored on this computer. Use a unique demo password, not a password from another service.</small>")

    dashboard = gr.Column(visible=False, elem_id="app-shell")
    with dashboard:
        gr.Markdown("<div id='brand-banner'><div id='eyebrow'>FINAL YEAR PROJECT · DEEP LEARNING</div><h1>Bone Tumor Classification</h1><p>Bone-tumour X-ray classification · ResNet50 transfer learning · Grad-CAM explainability</p></div>")
        with gr.Row():
            welcome = gr.Markdown("### Research workspace")
            logout_button = gr.Button("Log out", variant="secondary")
        with gr.Tabs():
            with gr.Tab("01 · X-ray analysis"):
                gr.Markdown("## Analyse an X-ray\nUpload a bone-tumour radiograph. The model returns two class scores and a Grad-CAM overlay. Use a sample image first to verify the workflow.")
                with gr.Row():
                    with gr.Column(scale=5, elem_id="soft-card"):
                        xray_input = gr.Image(type="filepath", image_mode=None, label="Input radiograph", height=360)
                        predict_button = gr.Button("Run model analysis", variant="primary", size="lg")
                    with gr.Column(scale=5, elem_id="soft-card"):
                        prediction_output = gr.Label(num_top_classes=2, label="Model class scores")
                        heatmap_output = gr.Image(type="pil", label="Grad-CAM overlay", height=360)
                explanation_output = gr.Markdown()
                predict_button.click(classify, inputs=xray_input, outputs=[prediction_output, heatmap_output, explanation_output])
                gr.Markdown("<div id='warning-card'><b>Clinical safety note:</b> a benign prediction does not rule out malignancy. A qualified radiologist must interpret the image with clinical history and other investigations.</div>")

            with gr.Tab("02 · Model performance"):
                m = test_metrics or {}
                if m:
                    with gr.Row():
                        gr.Markdown(f"<div id='metric-card'><h3>Test accuracy</h3><strong>{m['accuracy']*100:.1f}%</strong></div>")
                        gr.Markdown(f"<div id='metric-card'><h3>ROC-AUC</h3><strong>{m['roc_auc']:.3f}</strong></div>")
                        gr.Markdown(f"<div id='metric-card'><h3>Malignant recall</h3><strong>{m['recall']*100:.1f}%</strong></div>")
                        gr.Markdown(f"<div id='metric-card'><h3>Test images</h3><strong>{m['n_test']}</strong></div>")
                gr.Markdown(make_analysis())
                for image_file, caption in [
                    (C.OUTPUT_DIR / "training_curves.png", "Training and validation curves"),
                    (C.OUTPUT_DIR / "confusion_matrix.png", "Confusion matrix"),
                    (C.OUTPUT_DIR / "roc_curve.png", "ROC curve"),
                    (C.OUTPUT_DIR / "misclassified.png", "Misclassified test examples"),
                    (C.OUTPUT_DIR / "comparison.png", "Augmentation comparison"),
                ]:
                    if image_file.exists():
                        gr.Markdown(f"### {caption}")
                        gr.Image(value=str(image_file), label=caption, interactive=False)

            with gr.Tab("03 · Tumour guide"):
                gr.Markdown(TUMOR_EDUCATION)

            with gr.Tab("04 · Doctor / radiologist guide"):
                gr.Markdown("""## Prepare for a radiology consultation
This section helps you organise what to tell your doctor and what to ask about your imaging. It does **not** diagnose your symptoms or provide a medical urgency assessment.
""")
                with gr.Row():
                    consult_age = gr.Textbox(label="Age (optional)", placeholder="e.g., 20")
                    consult_location = gr.Dropdown(label="Area of concern", choices=["Upper arm / shoulder", "Forearm / wrist", "Hip / pelvis", "Thigh / knee", "Lower leg / ankle", "Foot / hand", "Other / not sure"], value="Other / not sure")
                with gr.Row():
                    consult_duration = gr.Dropdown(label="How long have symptoms been present?", choices=["New / today", "Several days", "Several weeks", "Months or longer", "No symptoms; finding on imaging", "Not sure"], value="Not sure")
                    consult_change = gr.Dropdown(label="Change over time", choices=["Improving", "Stable", "Getting worse", "Comes and goes", "Not sure"], value="Not sure")
                consult_symptoms = gr.CheckboxGroup(label="Symptoms (select any that apply)", choices=["Persistent bone pain", "Pain at night", "Swelling or lump", "Reduced movement", "Sudden severe pain", "Unable to bear weight/use limb", "Rapidly increasing swelling", "Fever or feeling unwell", "No symptoms"], value=[])
                with gr.Row():
                    consult_imaging = gr.Dropdown(label="Previous imaging / report", choices=["None yet", "X-ray only", "MRI", "CT", "More than one type", "Have a report but unsure", "Not provided"], value="Not provided")
                    consult_injury = gr.Dropdown(label="Recent injury in the area?", choices=["No", "Yes", "Not sure / prefer not to say"], value="Not sure / prefer not to say")
                consult_button = gr.Button("Create my doctor-visit checklist", variant="primary", size="lg")
                consult_output = gr.Markdown()
                consult_button.click(make_consultation_guide, inputs=[consult_age, consult_location, consult_duration, consult_symptoms, consult_imaging, consult_injury, consult_change], outputs=consult_output)
                gr.Markdown("**Important:** This is visit preparation, not medical advice. A clinician must assess symptoms and review the original images.")

            with gr.Tab("05 · Methodology"):
                gr.Markdown("""## Research methodology

**Task.** Binary classification of labelled bone-tumour radiographs into benign (class 0) and malignant (class 1).

**Pipeline.** Input validation → grayscale standardisation / 3-channel conversion → resize and normalisation → ResNet50 feature extraction → class scores → Grad-CAM visualisation.

**Transfer learning.** ResNet50 is a convolutional neural network pretrained on ImageNet and adapted to this two-class task. The model is not trained from scratch; the pretrained visual features are fine-tuned for the target dataset.

**Evaluation.** The held-out test set contains 281 images: 230 benign and 51 malignant. Report accuracy alongside malignant precision, malignant recall, F1-score, ROC-AUC and the confusion matrix. Recall matters because a false-benign prediction can miss a malignant case.

**Limitations.** The test set is small, the classes are imbalanced, external validation has not been demonstrated, and patient identifiers are not available to confirm patient-level independence between splits. Results may therefore be optimistic and may not generalise to different hospitals, scanners, patient populations or normal X-rays.

**Responsible use.** This application is an academic prototype for model exploration, not a medical device. Do not use predictions to diagnose, rule out cancer or select treatment.""")

    login_button.click(login_user, inputs=[login_email, login_password], outputs=[authenticated, login_message, login_page, dashboard, user_email]).then(
        lambda email: f"### Welcome to the workspace\nSigned in as **{email}**. Choose a section above to analyse images, inspect model performance, or learn about bone tumours.", inputs=user_email, outputs=welcome)
    register_button.click(register_user, inputs=[register_email, register_password, register_confirm], outputs=register_message)
    logout_button.click(logout_user, outputs=[authenticated, login_page, dashboard, user_email])

if __name__ == "__main__":
    demo.launch(share="google.colab" in sys.modules)
