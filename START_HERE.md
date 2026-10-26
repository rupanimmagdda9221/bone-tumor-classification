# Start here — Bone Tumor Classification

This folder contains the final-year AIML research project and an online-ready website.

## To run the website on your Windows laptop

1. Install Python 3.12 (64-bit) if needed: https://www.python.org/downloads/windows/ (tick **Add Python to PATH**).
2. Double-click `setup_windows.bat` and wait for **Setup complete**.
3. Double-click `run_demo.bat` and keep the black window open.
4. Open http://127.0.0.1:7860 in Chrome.
5. For sign-up, saved profiles, consultations and private uploads, configure the Supabase environment variables from `.env.example` in the Windows environment before starting the app. The easiest online setup is described in `DEPLOY_ONLINE.md`.

If the website says Supabase is not configured, the model code may still be present, but online-style account persistence will not work until you configure the backend.

## To make a public website link

Follow `DEPLOY_ONLINE.md` from top to bottom. It covers Supabase (accounts/database/private file storage), GitHub and Render deployment. You must create these accounts and add your own secret environment variables; this project cannot be published from your laptop merely by opening `127.0.0.1`.

## Included research components

- ResNet50 binary classifier (benign vs malignant)
- Grad-CAM visual explanation
- Existing test metrics and evaluation plots
- Bone tumour education
- Doctor/radiologist consultation preparation
- Persistent profile, consultation history, and private uploads when Supabase is configured

**Medical safety:** this is a research prototype, not a diagnostic tool. A benign model output does not rule out cancer; a qualified clinician must interpret medical images and symptoms.
