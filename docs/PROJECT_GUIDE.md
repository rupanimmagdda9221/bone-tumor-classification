# Bone Tumor Classification: Running Instructions and Project Walkthrough

Malla Reddy University, Dept. of AI & ML. Major Project, 2023–2027.
Team: N. Nandini (2311CS020469), N. Rupa Lakshmi (2311CS020480), P. Varshitha (2311CS020499).
Guide: Asst. Prof. Vineela.

This guide has two parts:
- **Part A** explains how to run the project on Mac, Windows or Google Colab.
- **Part B** walks through the whole project: data, code, model, results and demo, in the order you would present it.

> Research prototype for preliminary screening only. It is not a medical diagnosis tool.

---

## Part A: Running the project

### A.1 What you need

| | Demo / prediction | Re-evaluate or retrain |
|---|---|---|
| Python 3.12 (Windows) or the existing `.venv` (Mac) | yes | yes |
| `outputs/best_model.pt` (trained model, ~95 MB) | yes | created by training |
| BTXRD dataset in `data/raw/` | no | yes |
| GPU | no, CPU is fine | strongly recommended |

### A.2 Mac (this laptop, already set up)

Open **Terminal** and run:

```bash
cd ~/Desktop/bone_tumor_classification/src
../.venv/bin/python app.py
```

Wait for `Loaded model ...` and `Running on local URL: http://127.0.0.1:7860`, then open **http://127.0.0.1:7860** in a browser.
Keep the Terminal window open while you use the demo. To stop the server, press **Ctrl + C**.

Other commands, run from the same `src` folder:

```bash
../.venv/bin/python predict.py ../samples/malignant_IMG001650.png   # one image, text output + heatmap file
../.venv/bin/python evaluate.py      # recompute test metrics and plots into outputs/
../.venv/bin/python compare.py       # with/without augmentation table and chart
../.venv/bin/python train.py         # retrain (~25 min on Apple GPU) - overwrites outputs/best_model.pt
```

**Setting up on a different Mac:**
```bash
cd bone_tumor_classification
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### A.3 Windows

Full details and troubleshooting are in [`../WINDOWS_SETUP.md`](../WINDOWS_SETUP.md). The short version:

1. Install **Python 3.12 (64-bit)** from python.org and tick **"Add python.exe to PATH"**.
2. Extract `bone_tumor_classification_windows.zip` to a short path such as `C:\Projects\bone_tumor_classification`.
3. Double-click **`setup_windows.bat`**. You only do this once; it takes 5–15 minutes and needs internet.
4. Double-click **`run_demo.bat`**. The browser opens http://127.0.0.1:7860 after about 20 seconds.
   Keep the black console window open.

These scripts are optional and only needed to re-evaluate or retrain:

| Script | What it does |
|---|---|
| `download_dataset.bat` | downloads BTXRD into `data\raw\` |
| `evaluate.bat` | recomputes test metrics and plots |
| `train.bat` | retrains the model (back up `outputs\` first) |

If Windows shows "Windows protected your PC", click *More info*, then *Run anyway*.

### A.4 Google Colab (no installation)

1. Open `notebooks/bone_tumor_colab.ipynb` in Colab.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Upload `bone_tumor_classification_colab.zip` when asked.
4. Run the cells from top to bottom. The notebook downloads BTXRD, trains, evaluates, and starts the demo with a public `gradio.live` link.

### A.5 Using the demo

1. Click the **Bone X-ray** box and choose an image. You can also drag and drop a file, or copy an image and paste it with the clipboard icon.
   The file can be anywhere on the computer; it does not have to be in the project folder.
2. Click **Submit**.
3. Read the three outputs:
   - **Prediction:** the benign and malignant scores.
   - **Grad-CAM:** a heatmap of the image regions that drove the prediction.
   - **Explanation:** the confidence level, where the evidence is, and how reliable that confidence level was on the test set.

Accepted formats are PNG, JPEG, BMP, TIFF and WEBP. Unreadable, blank or very small images (under 64×64 px) are rejected with a message.

**Using images from Google:** the app will analyse them, but the model was trained only on BTXRD X-rays.
- Use plain X-rays that actually show a bone tumor, not MRI, CT or photos.
- Crop out arrows, text and watermarks.
- Prefer sources with a confirmed diagnosis, such as Radiopaedia.

Results on outside images are not covered by the reported test accuracy. The app also cannot detect non-X-ray images; it will give a score for any picture.

### A.6 Quick troubleshooting

| Problem | Fix |
|---|---|
| Browser says "This site can't be reached" | The server isn't running. Start `app.py` or `run_demo.bat` again and keep the window open. |
| `Address already in use` (port 7860) | An old demo is still running. Close that window, or end `python` in Activity Monitor / Task Manager. |
| "Cannot start the demo: ... not found" | `outputs/best_model.pt` is missing. Copy it from the zip or the backup. |
| Page loads but shows no upload box | Hard-refresh the page (Cmd/Ctrl + Shift + R). If that doesn't help, restart the server. |
| `No module named ...` | Run with `.venv` Python (`../.venv/bin/python` or `..\.venv\Scripts\python`), not the system Python. |

---

## Part B: Project walkthrough

### B.1 The problem in one paragraph

Bone tumors can be **benign** (for example osteochondroma or simple bone cyst) or **malignant** (for example osteosarcoma). X-ray is the first imaging test, but reading X-rays takes time and specialist experience, and the subtle differences between the two classes cause inconsistency between readers.

This project builds a deep-learning system that:
- classifies a bone-tumor X-ray as benign (0) or malignant (1);
- shows **why**, with a Grad-CAM heatmap and a written explanation.

### B.2 Objectives from Review 1 and where each is delivered

| Objective / promised outcome | Delivered in |
|---|---|
| Deep learning system for bone tumor classification | `src/train.py`, `src/predict.py`, `src/app.py` |
| Preprocessing: resizing, normalisation, augmentation | `src/dataset.py` |
| ResNet50 model for benign vs malignant | `src/model.py`, trained model `outputs/best_model.pt` |
| Evaluation with accuracy, precision, recall, F1 | `src/evaluate.py`, `outputs/test_metrics.json` |
| Confusion matrix | `outputs/confusion_matrix.png` |
| Working prototype | Gradio web app (`src/app.py`) |
| Better performance through augmentation | Ablation: F1 0.639 → 0.699, AUC 0.901 → 0.945 |
| Error analysis and model comparison | `outputs/misclassified.png`, `outputs/comparison.png` |

### B.3 Pipeline overview

```
BTXRD X-ray ──► standardize_xray ──► augmentation ──► ResNet50 ──► softmax ──► benign / malignant
 (or upload)    (grayscale, ≤512px,   (training only)   (ImageNet    (2 classes)        │
                224×224, ImageNet                       weights,                         ▼
                normalisation)                          fine-tuned)          Grad-CAM heatmap + explanation
```

### B.4 Dataset

- **BTXRD** (Bone Tumor X-ray Radiograph Dataset): Yao et al., *Scientific Data* 12, 88 (2025), CC BY 4.0.
- It has 3,746 radiographs. We use the **1,867 tumor images: 1,525 benign and 342 malignant (18.3%)**. Normal images are out of scope.
- The data is split 70 / 15 / 15, stratified, with seed 42:

| Split | Images | Benign | Malignant |
|---|---|---|---|
| Train | 1,306 | 1,066 | 240 |
| Validation | 280 | 229 | 51 |
| Test | 281 | 230 | 51 |

The split is saved in `outputs/splits.json`, so every run uses exactly the same images.
`src/prepare_btxrd.py` sorts the downloaded dataset into `data/raw/benign` and `data/raw/malignant`.

### B.5 Code tour (read in this order)

| File | What to say about it |
|---|---|
| `config.py` | One place for paths, image size (224), class names, hyper-parameters and device choice (CUDA → Apple MPS → CPU). |
| `dataset.py` | `standardize_xray()` fixes orientation, converts 16-bit to 8-bit grayscale and limits the size to 512 px. `make_splits()` creates the stratified split. `get_loaders()` applies resize, augmentation (train only) and ImageNet normalisation. |
| `model.py` | Loads ImageNet ResNet50 and replaces the last layer with Dropout 0.4 → Linear 2048→256 → ReLU → Dropout 0.3 → Linear 256→2. `freeze_backbone()` / `unfreeze_top()` control which layers train. `load_checkpoint()` refuses a model trained for different classes. |
| `train.py` | **Stage 1:** head only, 5 epochs, lr 1e-3. **Stage 2:** layer3 + layer4 + head, up to 20 epochs, lr 1e-4. AdamW, batch 32, class-weighted loss (0.613 / 2.721), LR halved on plateau, early stopping (patience 6). Keeps the best model by validation F1. |
| `metrics.py` | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, training curves, confidence bands. |
| `evaluate.py` | Runs the saved model once on the 281 test images and writes all plots and `test_metrics.json`. |
| `compare.py` | Builds the with/without augmentation table (`comparison.md`) and chart. |
| `predict.py` | `GradCAM` class (hooks on ResNet50 `layer4`), `describe_cam()` (where and how large the evidence is), `explain()` (the text shown in the app). Also works from the command line. |
| `app.py` | Gradio interface: upload → validate → predict → heatmap → explanation, with friendly error messages. |

### B.6 Training behaviour

- The best model was at **epoch 18** (13th fine-tuning epoch), with validation F1 0.743. Training stopped after epoch 24.
- Training accuracy reached ~0.99, while validation accuracy stayed at 0.87–0.90. That gap is mild overfitting, which is why we use early stopping, dropout, weight decay and augmentation.
- See `outputs/training_curves.png` and `outputs/train_log.txt`.

### B.7 Results (test set, 281 images, never used in training)

| Metric | Value |
|---|---|
| Accuracy | **0.890** |
| ROC-AUC | **0.945** |
| Malignant precision | 0.692 |
| Malignant recall (sensitivity) | 0.706 |
| Malignant F1 | 0.699 |
| Benign F1 / macro F1 | 0.933 / 0.816 |
| Specificity | 0.930 |

Confusion matrix (rows are actual, columns are predicted):

| | Pred. benign | Pred. malignant |
|---|---|---|
| **Benign (230)** | 214 | 16 (false alarms) |
| **Malignant (51)** | 15 (missed) | 36 |

**Ablation: does augmentation help?** The same split and settings were used; only augmentation differs.

| Model | Acc. | Prec. | Recall | F1 | AUC | Missed malignant |
|---|---|---|---|---|---|---|
| With augmentation (proposed) | 0.890 | 0.692 | **0.706** | **0.699** | **0.945** | 15 / 51 |
| Without augmentation | 0.875 | 0.674 | 0.608 | 0.639 | 0.901 | 20 / 51 |

**How reliable is the confidence?** Test accuracy at each confidence level (the app quotes these figures):

| Confidence | Images | Correct |
|---|---|---|
| High (≥ 90%) | 239 | 95.4% |
| Moderate (70–90%) | 28 | 67.9% |
| Low (< 70%) | 14 | 21.4% |

### B.8 Explainability (Grad-CAM)

- Grad-CAM takes the gradient of the predicted class score with respect to ResNet50's last feature maps. It weights each map by its average gradient, sums them and applies ReLU. The result is upsampled and overlaid on the X-ray.
- **Sanity check passed:** with randomised weights the heatmap changes completely (correlation −0.29), so it really comes from the trained model.
- **Honest observations:**
  - In one benign pelvis image the heatmap partly covers the "R" side marker, which is possible shortcut learning.
  - IMG001690 is a malignant tumor predicted benign at 93.9%. Grad-CAM shows *where* the model looked, not whether it was right.

### B.9 Suggested live-demo order (about 3 minutes)

These predictions were checked on 25 Oct 2026 and come from the images in `samples/`:

| # | Image | Expected result | Point to make |
|---|---|---|---|
| 1 | `malignant_IMG001650.png` | malignant 100% | Correct and confident; the heatmap sits on the lesion. |
| 2 | `benign_IMG001092.png` | benign 97.7% | Correct benign case; explain the explanation text. |
| 3 | `malignant_IMG000160.png` | malignant 75.3% | Moderate confidence. The app says ~68% of such predictions were correct, so the result needs review. |
| 4 | `hardcase_missed_malignant_IMG001690.png` | benign 93.9% (**wrong**, it is malignant) | Known failure: shows the limitation and why a radiologist must review. |
| 5 | (optional) a PDF or tiny image | error message | Input validation. |

Other samples are `benign_IMG000463`, `benign_IMG000636`, `benign_IMG001165`, `malignant_IMG000206` and `malignant_IMG001532`. All of them are predicted correctly.

### B.10 Limitations and future work

**Limitations**
- Only 342 malignant images, so 15 of 51 malignant test cases are missed.
- BTXRD has no patient IDs, so views of the same patient may appear in both train and test. Scores may be slightly optimistic.
- One dataset only, with no external hospital validation.
- Each configuration was trained once.
- There is no check for non-X-ray images.

**Future work**
- More malignant and multi-centre data.
- Patient-level splits and external validation.
- Threshold tuning for higher sensitivity.
- Comparisons with EfficientNet, DenseNet and Vision Transformers.
- Tumor-subtype classification.
- Localisation using BTXRD's masks.
- An out-of-distribution check to reject non-X-ray images.

### B.11 Project folder map

```
bone_tumor_classification/
├── src/                  all Python code (see B.5)
├── outputs/              trained model, metrics, plots, splits.json, no_augmentation/ ablation run
├── samples/              9 test-set X-rays for the demo
├── data/                 BTXRD dataset (only needed for evaluate/train)
├── notebooks/            Google Colab notebook
├── docs/                 final slides, project report, viva Q&A, to-do list, this guide
├── README.md             project summary
├── WINDOWS_SETUP.md      detailed Windows instructions
├── requirements.txt      pinned library versions
└── *.bat                 Windows one-click scripts
```
