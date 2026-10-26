# Bone Tumor Classification from X-Ray Images (ResNet50)

Malla Reddy University, Dept. of AI & ML. Major Project.
This code implements everything planned in Review 1:
- preprocessing (resizing, normalization, data augmentation)
- a ResNet50 transfer-learning model that classifies bone tumors as **benign** or **malignant**
- evaluation with accuracy, precision, recall, F1-score and a confusion matrix
- a working prediction prototype

> Research prototype for preliminary screening only. It does not give a medical diagnosis.

## Online website deployment

The project includes a Flask website (`web_app.py`) with Supabase Auth, a persistent profile/history database, and a private storage bucket for X-rays/reports. See **`DEPLOY_ONLINE.md`** for the beginner-friendly Supabase + GitHub + Render deployment steps. The local Gradio interface remains available in `src/app.py`, but `run_demo.bat` launches the new Flask website.


## Results (held-out test set, 281 X-rays)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Missed malignant | False alarms |
|---|---|---|---|---|---|---|---|
| **ResNet50 + augmentation (proposed)** | **0.890** | **0.692** | **0.706** | **0.699** | **0.945** | 15 / 51 | 16 / 230 |
| ResNet50, no augmentation (ablation) | 0.875 | 0.674 | 0.608 | 0.639 | 0.901 | 20 / 51 | 15 / 230 |

Precision, recall and F1 are for the malignant class. For the benign class, the proposed model has F1 0.933 and
macro-F1 is 0.816. Both models used the same data split and the same training settings. The only difference is
whether augmentation was used, so the gain comes from augmentation: malignant recall rose from 0.61 to 0.71 and
ROC-AUC from 0.90 to 0.95.

| File in `outputs/` | Shows |
|---|---|
| `training_curves.png` | train vs validation loss and accuracy per epoch |
| `confusion_matrix.png` | class-wise predictions (214/230 benign and 36/51 malignant correct) |
| `roc_curve.png` | sensitivity vs false-positive rate (AUC 0.945) |
| `misclassified.png` | error analysis: every misclassified test X-ray with its Grad-CAM heatmap |
| `comparison.png`, `comparison.md` | with vs without augmentation |
| `test_metrics.json`, `train_log.txt` | raw numbers and the full training log |
| `no_augmentation/` | the same files for the ablation model |

## What the demo shows for each X-ray

- **Prediction:** softmax scores for benign / malignant. The class above 50% is reported.
- **Grad-CAM heatmap:** computed from the same trained model on the uploaded image.
- **Explanation:** built only from measured quantities:
  - how far the score is from the 50% boundary
  - where the Grad-CAM evidence is concentrated and how large it is
  - a warning if the heatmap falls mostly on dark background
  - how often the model was correct on the test set at the same confidence level (high ≥90%: 95% of 239; moderate 70–90%: 68% of 28; low <70%: 21% of 14)
  - overall precision and recall
  
  It does not name radiological signs or tumor subtypes, and it is not a diagnosis.

Uploads go through exactly the same preprocessing as the training data (`standardize_xray` in `dataset.py`).
Original full-resolution, colour or 16-bit X-rays therefore get the same result as the dataset copies.
Unreadable, blank or tiny images are rejected with a message.

## Dataset

BTXRD (Bone Tumor X-ray Radiograph Dataset): Yao et al., *Scientific Data* 2025,
[figshare 10.6084/m9.figshare.27865398](https://doi.org/10.6084/m9.figshare.27865398), CC BY 4.0.
It has 3,746 radiographs. This project uses the **1,867 tumor images: 1,525 benign and 342 malignant**.
Normal images are excluded, as defined in the project scope.

The split is stratified 70 / 15 / 15: 1,306 train, 280 validation and 281 test images. It is fixed in `outputs/splits.json`.

## Method

1. **Preprocessing:** grayscale X-rays are converted to 3 channels, resized to 224×224 and normalized with ImageNet mean/std.
2. **Augmentation (training only):** random resized crop, horizontal flip, ±15° rotation, small translation,
   and brightness/contrast jitter.
3. **Model:** ResNet50 pretrained on ImageNet. The final layer is replaced with Dropout → Linear(2048→256) → ReLU → Dropout → Linear(256→2).
4. **Training:**
   - Stage 1 (5 epochs): only the new head is trained (backbone frozen, AdamW, lr 1e-3).
   - Stage 2 (up to 20 epochs): `layer3` and `layer4` are also fine-tuned (lr 1e-4).
   - Class imbalance is handled with a weighted cross-entropy loss.
   - The learning rate is halved when validation F1 plateaus, and training stops early after 6 epochs without improvement.
   - The checkpoint with the best validation F1 is kept.
5. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix and ROC curve on the unseen test set.
6. **Explainability:** Grad-CAM heatmaps on the last ResNet50 block show which regions drove each prediction.

## Project structure

```
src/
  config.py         paths, image size, split ratios, hyperparameters
  dataset.py        stratified split, preprocessing + augmentation
  model.py          ResNet50 with the new classification head
  train.py          two-stage training, then a test-set evaluation (--no-augment for the ablation)
  evaluate.py       test metrics, confusion matrix, ROC curve, misclassified gallery
  compare.py        comparison table + chart across runs
  metrics.py        metric computation and plots
  predict.py        single-image prediction with a Grad-CAM heatmap
  app.py            Gradio web prototype
  prepare_btxrd.py  sorts the downloaded BTXRD into data/raw/{benign,malignant}
notebooks/bone_tumor_colab.ipynb   the full pipeline on Google Colab (GPU)
samples/            test-set X-rays for the demo (the model never saw them in training)
outputs/            trained model, metrics and plots
```

## Run it on Windows

See **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)**. In short: install Python 3.12, double-click `setup_windows.bat`, then `run_demo.bat`.

## Run it (Mac)

```bash
cd ~/Desktop/bone_tumor_classification/src
../.venv/bin/python app.py                                   # web demo at http://127.0.0.1:7860
../.venv/bin/python predict.py ../samples/malignant_IMG001650.png
../.venv/bin/python evaluate.py                              # re-compute test metrics and plots
../.venv/bin/python compare.py                               # comparison table + chart
```

Retrain from scratch, which takes about 25 minutes on Apple Silicon:

```bash
../.venv/bin/python train.py                                             # proposed model -> outputs/
../.venv/bin/python train.py --no-augment --output-dir ../outputs/no_augmentation
```

On a new machine, set up first:
1. Create the environment: `python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`
2. Download BTXRD from figshare.
3. Run `python src/prepare_btxrd.py --src /path/to/BTXRD`.

## Run it on Google Colab

Open `notebooks/bone_tumor_colab.ipynb` in Colab and set the runtime to **T4 GPU**.
Upload `bone_tumor_classification_colab.zip`, which contains the code, the data split and the samples, then run the cells top to bottom.
The notebook downloads BTXRD, trains, evaluates, compares, and launches the demo with a public link.

## Mapping to Review 1

| Declared in Review 1 | Delivered |
|---|---|
| Obj 1: deep learning system for bone tumor classification | `train.py`, `predict.py`, `app.py` |
| Obj 2: resizing, normalization, augmentation | `dataset.py` |
| Obj 3: ResNet50, benign vs malignant | `model.py` and the trained `outputs/best_model.pt` |
| Obj 4: accuracy, precision, recall, F1 | `evaluate.py`, `test_metrics.json` |
| Outcome: functional prototype | `app.py` (Gradio) and `predict.py` |
| Outcome: improved performance through preprocessing/augmentation | ablation: F1 0.639 → 0.699, AUC 0.901 → 0.945 |
| Outcome: confusion matrix for class-wise performance | `confusion_matrix.png` |
| Plan: analyze errors, compare model performance | `misclassified.png`, `comparison.png` |
| Tools: Python, PyTorch, PIL, GPU / Google Colab | Apple GPU (MPS) locally and the Colab notebook |

## Limitations

- Only 342 malignant images, so 15 of 51 malignant test cases are still missed.
- BTXRD has no patient IDs, so different views of one patient may appear in both train and test. This can make
  the scores slightly optimistic.
- Each configuration was trained once (seed 42), so differences between runs include some random variation.
- The model was tested on one dataset only, with no external validation from another hospital.


### Refined interface update
The login page now uses a responsive layout, higher-contrast typography, clearer project naming, improved form fields and mobile-width styling. The project brand is Bone Tumor Classification; the task title is Bone Tumor Classification.
