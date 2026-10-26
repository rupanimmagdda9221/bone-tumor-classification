# Running the project on Windows

Requirements: Windows 10 / 11 (64-bit) with Python 3.12. An NVIDIA GPU is optional: it only matters for training.
The demo and prediction run fine on any laptop CPU.

## What you need

| | Needed for demo | Needed for training |
|---|---|---|
| Python 3.12 (64-bit) | yes | yes |
| ~3 GB free disk (PyTorch + packages) | yes | yes |
| Internet (first setup) | yes | yes |
| BTXRD dataset (~840 MB, auto-downloaded) | no | yes |
| NVIDIA GPU | no | strongly recommended (CPU training takes several hours) |

## Step 1: Install Python (once)

1. Download **Python 3.12** (64-bit) from https://www.python.org/downloads/windows/
   ("Windows installer (64-bit)").
2. On the first installer screen, **tick "Add python.exe to PATH"**, then click *Install Now*.
3. Check it: open **Command Prompt** (Start → type `cmd`) and run `python --version`.
   It should print `Python 3.12.x`.

> Use 3.12 rather than the newest Python. PyTorch and the other packages are best supported on 3.12.

## Step 2: Copy the project

Copy the `bone_tumor_classification` folder to the Windows machine, for example to `C:\Projects\bone_tumor_classification`.
If you received `bone_tumor_classification_windows.zip`, right-click it → *Extract All*.

- **Do not copy the `.venv` folder from the Mac.** It only works on the Mac. Setup creates a new one.
- Avoid folder paths that contain OneDrive or very long names. `C:\Projects\...` is the safest choice.
- `outputs\best_model.pt` (~95 MB) is the trained model. It must be present for the demo.

## Step 3: Setup (once)

Double-click **`setup_windows.bat`**. This step:
- creates a virtual environment in `.venv`
- installs PyTorch, using the **GPU (CUDA) version automatically if an NVIDIA GPU is detected**, otherwise the CPU version
- installs the remaining requirements (Gradio, scikit-learn, matplotlib, ...)

It takes 5–15 minutes depending on internet speed. The last lines should say
`PyTorch 2.x.x | GPU available: True/False` and then `Setup complete`.

If Windows shows a blue **"Windows protected your PC"** box, click *More info* → *Run anyway*.
This happens because the .bat files are not signed.

## Step 4: Run the demo

Double-click **`run_demo.bat`**.
- A black console window starts the server. After about 20 seconds the browser opens **http://127.0.0.1:7860**.
- Upload an X-ray from the `samples` folder. You'll see the benign / malignant probabilities and a Grad-CAM heatmap.
- **Keep the console window open** while using the demo. Closing it stops the server, and the browser will then show
  "This site can't be reached".
- If Windows Firewall asks about Python, you can click *Cancel*. The demo only uses this computer (127.0.0.1).

## Optional: re-evaluate or retrain

| File | What it does | Time |
|---|---|---|
| `download_dataset.bat` | downloads BTXRD and fills `data\raw\benign` and `data\raw\malignant` | 5–15 min |
| `evaluate.bat` | recomputes the test metrics, confusion matrix, ROC curve and comparison into `outputs\` | 1–5 min |
| `train.bat` | retrains the model and **overwrites** `outputs\best_model.pt` | ~20–40 min on GPU, several hours on CPU |

Run `download_dataset.bat` before `evaluate.bat` or `train.bat`. Back up the `outputs` folder before retraining
if you want to keep the current model and results.

`train.bat` accepts the same options as `train.py`:
```bat
train.bat --batch-size 16
train.bat --no-augment --output-dir ..\outputs\no_augmentation
```

## Manual commands (instead of the .bat files)

Run in **Command Prompt** from the project folder:
```bat
cd C:\Projects\bone_tumor_classification
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
REM GPU (NVIDIA) machines:
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
REM CPU-only machines instead:   .venv\Scripts\python -m pip install torch torchvision
.venv\Scripts\python -m pip install -r requirements.txt

cd src
..\.venv\Scripts\python app.py
..\.venv\Scripts\python predict.py ..\samples\malignant_IMG001650.png
..\.venv\Scripts\python evaluate.py
..\.venv\Scripts\python train.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick **Add python.exe to PATH**, or use `py -3.12` instead of `python`. |
| `GPU available: False` on a machine with an NVIDIA GPU | Update the NVIDIA driver (https://www.nvidia.com/drivers), delete the `.venv` folder, and run `setup_windows.bat` again. |
| `CUDA out of memory` during training | Use `train.bat --batch-size 16` (or 8). |
| Training is extremely slow | It is running on CPU. Use a machine with an NVIDIA GPU, or Google Colab (`notebooks\bone_tumor_colab.ipynb`). |
| `Address already in use` / port 7860 busy | Another demo window is still open. Close it, or end `python.exe` in Task Manager. |
| Browser shows "This site can't be reached" | The console window from `run_demo.bat` was closed. Start it again. |
| `No module named ...` | Run the scripts through `.venv\Scripts\python` (the .bat files do this), not the global `python`. |
| `SSL: CERTIFICATE_VERIFY_FAILED` while downloading | Company or college networks sometimes block downloads. Try another network or a phone hotspot. |
| `No images found under data\raw\...` | Run `download_dataset.bat` first (only needed for evaluate/train). |
