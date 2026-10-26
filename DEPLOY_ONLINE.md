# Put Bone Tumor Classification online (beginner guide)

This project now has a Flask website, Supabase Auth, a persistent Postgres database, private file storage, saved consultation history, profile settings, the existing ResNet50 checkpoint, Grad-CAM, and the existing model-evaluation results.

**I cannot publish the website or create your final public URL from this chat** because deployment requires accounts and secrets that only you should control. Follow these steps to create the live link.

## Before you start

- You need a GitHub account and a Supabase account.
- You need a Render account for hosting. The model file is about 97 MB, and loading PyTorch can take several minutes on first start.
- Do not put passwords, Supabase service-role keys, medical files, or `.env` files into a public GitHub repository.
- This is an academic prototype, not a clinical service. Do not use real patients' identifiable medical records for a college demo.

## Step 1 — Create the Supabase backend

1. Open https://supabase.com and create a new project. Choose a strong database password and save it somewhere safe.
2. In the project, open **SQL Editor → New query**.
3. Open `supabase_schema.sql` from this project, copy the whole file into the SQL editor, and click **Run**.
4. Open **Storage** and verify that a bucket named `private-medical-files` exists and is marked **Private**. The SQL file attempts to create it.
5. Open **Project Settings → API** (the labels can vary). Copy the Project URL, publishable/anon key, and service-role/secret key.
   - `SUPABASE_URL` = Project URL
   - `SUPABASE_ANON_KEY` = publishable/anon key
   - `SUPABASE_SERVICE_ROLE_KEY` = service-role/secret key
6. Never put the service-role key in browser JavaScript or GitHub. It belongs only in Render's server environment variables.
7. In **Authentication → URL Configuration**, later add your Render URL as the site URL if you use email confirmation. For an online deployment, keep email verification enabled if possible and test the verification link.

## Step 2 — Put the project on GitHub

1. Extract the project ZIP on your computer.
2. Create a **private** GitHub repository named `bone-tumor-classification`.
3. Upload the project files to that repository. Include `src/`, `outputs/best_model.pt`, `outputs/test_metrics.json`, `web_app.py`, `requirements.txt`, `render.yaml`, and `supabase_schema.sql`.
4. Do not upload `.venv/`, `.env`, `users.sqlite3`, or any real patient files. The project `.gitignore` should exclude them; check before committing.
5. The trained model checkpoint must be present in `outputs/best_model.pt` for inference to work.

## Step 3 — Deploy on Render

1. Open https://render.com and sign in using GitHub.
2. Choose **New → Web Service** and connect your private repository.
3. Render should detect `render.yaml`. If asked, use:
   - Runtime: Python
   - Build command: `pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`
     (installs the CPU-only PyTorch build first; the default Linux build includes CUDA and runs out of memory on the 512 MB Starter instance)
   - Start command: `gunicorn --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT web_app:app`
4. Add these environment variables under the service's **Environment** settings:
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `SUPABASE_STORAGE_BUCKET` = `private-medical-files`
   - `FLASK_SECRET_KEY` = a long random value (Render can generate this)
5. Choose a plan with enough RAM for PyTorch + ResNet50. Free/low-memory tiers may crash or be too slow. PyTorch dependencies can make the first build take a while.
6. Click **Create Web Service** and wait for the deploy log to show success. The public URL will look like `https://your-service-name.onrender.com`; use the exact URL Render gives you.
7. Visit `https://YOUR-RENDER-URL/health`. It should return JSON showing `status: ok`, `supabase_configured: true`, and ideally `model_loaded: true`.

## Step 4 — Configure email authentication

1. In Supabase **Authentication → URL Configuration**, set the Site URL to your Render URL.
2. Add the Render URL to the allowed redirect URLs if your project settings require it.
   - For **Forgot password** to work, add `https://YOUR-RENDER-URL/reset-password` to **Redirect URLs**. Without it, Supabase sends the reset link to the Site URL instead and the new-password form never appears.
   - Test it: click **Forgot password?** on the sign-in page, open the email, set a new password, then sign in with it. Supabase's built-in email sender only allows a few emails per hour, so add custom SMTP under **Authentication → Emails** if reset emails stop arriving.
3. Register using your own test email. If Supabase asks for email verification, open the verification email and then sign in.
4. If sign-up reports that confirmation is required, that is expected with email verification enabled. Do not disable verification for a real public website just to make the demo easier.

## Step 5 — Test all features before sharing the URL

- Create a test account, sign out, and sign back in.
- Edit and save the profile, then check it remains saved after a new sign-in.
- Create a doctor-visit checklist and verify it appears in Saved history.
- Upload a sample X-ray, run the model, and test both one-time analysis and the optional private-save checkbox.
- Upload a sample report in Saved history and download it through the private link.
- Sign in as a second test user and verify that the first user's history/files are not visible.
- Check `/health` and Render logs if a feature fails.

## What is saved, and where?

- **Email and password:** Supabase Auth manages credentials and password hashing; this app does not store plaintext passwords.
- **Profile:** Supabase Postgres (`user_profiles`).
- **Consultation and analysis history:** Supabase Postgres (`consultations`). The summary may contain health information; keep access limited.
- **X-rays and reports:** Supabase Storage private bucket (`private-medical-files`), with ownership metadata in `uploaded_files`.
- **Web session:** an opaque random session ID is stored in a HttpOnly cookie; the matching session record is stored in Postgres. The cookie is not a password.

## Important security and medical limitations

This is a student project scaffold, not a production medical-record system. Before accepting real patient data, obtain appropriate institutional/privacy review, add account deletion and data-retention controls, logging/monitoring, rate limiting, robust email/password reset, backups, and security testing. Never claim the AI is clinically validated. A benign prediction cannot rule out malignancy, and Grad-CAM is not a tumour boundary. A qualified radiologist/clinician must review findings.

## Local testing

On Windows, run `setup_windows.bat` once, then `run_demo.bat`. Open `http://127.0.0.1:7860`. Local login/storage features require the Supabase environment variables to be set in the terminal or Windows environment; online deployment is configured using Render's Environment page.
