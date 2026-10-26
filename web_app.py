"""Online Bone Tumor Classification research website.
Uses Supabase Auth for passwords, Supabase Postgres for records, and a private Storage bucket for files.
"""
import os, sys, uuid, secrets, json, re, io
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps

from flask import Flask, request, redirect, url_for, render_template_string, session, flash, send_file, abort
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import numpy as np
from supabase import create_client

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import config as C

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-only-change-me')
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024
BUCKET = os.environ.get('SUPABASE_STORAGE_BUCKET', 'private-medical-files')
SUPABASE_URL = os.environ.get('SUPABASE_URL', '').strip()
SUPABASE_ANON_KEY = os.environ.get('SUPABASE_ANON_KEY', '').strip()
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '').strip()
CONFIGURED = all([SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY])
admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) if CONFIGURED else None
    
predictor = None
bands = None
test_metrics = None


def get_predictor():
    global predictor, explain, bands, test_metrics

    if predictor is None:
        if not C.CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {C.CHECKPOINT}"
            )

        from predict import Predictor, explain, load_confidence_bands

        predictor = Predictor(C.CHECKPOINT)
        bands, test_metrics = load_confidence_bands()

        print(
            f'[Bone Tumor Classification] Model loaded on {predictor.device}'
        )

    return predictor
r

BASE_CSS = r'''
:root{--navy:#102a43;--teal:#087f78;--mint:#d9f7f0;--paper:#f3f6fa;--ink:#172b3e;--muted:#607286;--line:#dce5ed;--white:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.6}
a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}.wrap{max-width:1180px;margin:28px auto;padding:0 18px}.hero{background:linear-gradient(120deg,#102a43,#164e63 58%,#087f78);color:#fff;padding:28px 32px;border-radius:18px;margin-bottom:20px}.eyebrow{color:#86f0d9;text-transform:uppercase;letter-spacing:1.8px;font-size:11px;font-weight:800}.hero h1{font-size:clamp(28px,4vw,40px);line-height:1.15;letter-spacing:-.8px;margin:10px 0}.hero p{margin:0;color:#e7f8f5}.card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:22px;margin-bottom:18px;box-shadow:0 8px 30px #142b4308}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.nav{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:12px 0 20px}.nav a{padding:9px 12px;background:#fff;border:1px solid var(--line);border-radius:10px;font-weight:650}.nav .right{margin-left:auto}.muted{color:var(--muted);font-size:14px}.label{display:block;font-size:13px;font-weight:700;margin:10px 0 5px;color:#314960}input,select,textarea{width:100%;padding:12px 13px;border:1px solid #cdd9e4;border-radius:10px;font:inherit;background:white;color:var(--ink)}textarea{min-height:90px;resize:vertical}input:focus,select:focus,textarea:focus{outline:3px solid #087f7822;border-color:var(--teal)}button,.button{display:inline-block;background:var(--teal);color:#fff;border:0;border-radius:10px;padding:12px 16px;font:inherit;font-weight:750;cursor:pointer}.button.secondary,button.secondary{background:#eaf3f5;color:var(--navy);border:1px solid var(--line)}.full{width:100%;margin-top:14px}.alert{padding:12px 15px;border-radius:10px;margin:12px 0;background:#fff8e8;border:1px solid #f0d69b;color:#654b12}.alert.success{background:#eafaf3;border-color:#b6e5ce;color:#14532d}.metric{background:#fff;border:1px solid var(--line);border-radius:13px;padding:15px}.metric small{display:block;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;font-size:11px;font-weight:750}.metric strong{font-size:26px;color:var(--navy)}table{width:100%;border-collapse:collapse}th,td{padding:10px;border:1px solid var(--line);text-align:left;vertical-align:top}th{background:#edf5f6}.warning{background:#fff8e8;border:1px solid #f0d69b;border-radius:12px;padding:15px;color:#654b12}.result{white-space:pre-wrap}.filelist li{margin:8px 0}img.result-img{max-width:100%;height:auto;border-radius:12px;border:1px solid var(--line)}.footer{color:var(--muted);font-size:12px;text-align:center;padding:20px}
@media(max-width:760px){.wrap{margin:10px auto;padding:0 10px}.hero{padding:22px 20px}.card{padding:16px}.grid,.grid3{grid-template-columns:1fr}.nav .right{margin-left:0}}

/* Refined responsive UI layer */
:root{
 --navy:#0b1d33;--teal:#087f8c;--mint:#b8f4e7;--paper:#f4f7fb;
 --ink:#16263a;--muted:#63748a;--line:#dce5ef;--white:#fff;
 --shadow:0 18px 55px rgba(16,42,67,.08);
}
html{scroll-behavior:smooth}
body{
 background:
  radial-gradient(ellipse at 8% 2%,rgba(8,127,140,.10),transparent 34rem),
  radial-gradient(ellipse at 96% 18%,rgba(89,128,210,.09),transparent 30rem),
  var(--paper);
 color:var(--ink);letter-spacing:-.012em;
}
.wrap{max-width:1240px;margin:30px auto;padding:0 24px}
.hero{
 position:relative;overflow:hidden;
 background:linear-gradient(118deg,#091b31 0%,#123e59 54%,#087f78 100%);
 padding:clamp(26px,4vw,44px);border-radius:24px;margin-bottom:24px;
 box-shadow:0 22px 60px rgba(8,36,57,.18);
 border:1px solid rgba(255,255,255,.14)
}
.hero:after{
 content:"";position:absolute;width:300px;height:300px;right:-70px;top:-145px;
 border:1px solid rgba(184,244,231,.22);border-radius:50%;
 box-shadow:0 0 0 35px rgba(184,244,231,.045),0 0 0 75px rgba(184,244,231,.035);
 pointer-events:none
}
.hero .eyebrow{display:inline-flex;align-items:center;gap:8px;color:#a6f4e3;
 background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14);
 border-radius:999px;padding:7px 11px;letter-spacing:1.4px}
.hero h1{position:relative;z-index:1;font-size:clamp(30px,4.6vw,48px);font-weight:780;letter-spacing:-1.8px;margin:18px 0 12px;max-width:850px}
.hero p{position:relative;z-index:1;max-width:760px;font-size:clamp(14px,1.8vw,17px);line-height:1.7;color:#e1f3f5}
h2,h3,h4{color:var(--navy);letter-spacing:-.045em;line-height:1.25}
h2{font-size:clamp(22px,3vw,29px);margin:0 0 10px}
h3{font-size:20px;margin:0 0 10px}
p{margin-top:8px}
.card{
 border:1px solid rgba(210,222,235,.9);border-radius:20px;padding:clamp(19px,3vw,28px);
 margin-bottom:20px;box-shadow:var(--shadow);transition:box-shadow .2s ease,border-color .2s ease;
}
.card:hover{box-shadow:0 20px 60px rgba(16,42,67,.105);border-color:#c7d9e7}
.grid,.grid3{gap:20px}
.nav{gap:9px;margin:16px 0 24px}
.nav a{padding:10px 14px;border-radius:12px;background:rgba(255,255,255,.88);font-size:14px;color:#18334b;border-color:#dce5ed;transition:all .18s ease}
.nav a:hover{background:#e9f8f6;border-color:#9adbd1;text-decoration:none;transform:translateY(-1px)}
.nav .right{display:flex;align-items:center}
.label{font-size:13px;letter-spacing:.01em;color:#31475f;margin:16px 0 7px}
input,select,textarea{
 min-height:46px;border:1px solid #ccd8e5;border-radius:12px;padding:12px 14px;
 box-shadow:0 1px 2px rgba(16,42,67,.025);transition:border-color .15s ease,box-shadow .15s ease;
}
input[type=file]{padding:11px;background:#f8fafc;min-height:auto}
input[type=checkbox]{min-height:0;accent-color:var(--teal)}
input:focus,select:focus,textarea:focus{outline:none;border-color:#0b9b9b;box-shadow:0 0 0 4px rgba(8,127,140,.13)}
button,.button{
 border-radius:12px;padding:12px 18px;background:linear-gradient(135deg,#087f78,#0b8c9a);
 box-shadow:0 7px 16px rgba(8,127,120,.16);transition:transform .16s ease,box-shadow .16s ease,filter .16s ease;
 text-decoration:none
}
button:hover,.button:hover{filter:brightness(1.04);transform:translateY(-1px);box-shadow:0 10px 22px rgba(8,127,120,.2);text-decoration:none}
button:active,.button:active{transform:translateY(0)}
.button.secondary,button.secondary{background:#edf4f8;color:#19344b;box-shadow:none}
.button.secondary:hover,button.secondary:hover{background:#e1eef4;box-shadow:none}
.full{margin-top:18px;min-height:48px}
.metric{border-radius:16px;padding:18px 19px;background:linear-gradient(145deg,#fff,#f8fbfd);box-shadow:0 5px 18px rgba(16,42,67,.035)}
.metric small{letter-spacing:1px;color:#718196}
.metric strong{display:block;font-size:clamp(24px,3vw,31px);letter-spacing:-1px;margin-top:5px}
.warning{border-radius:15px;padding:17px 19px;line-height:1.65;background:#fff8e9;border-color:#f2d89d}
.alert{border-radius:13px;box-shadow:0 5px 16px rgba(16,42,67,.04)}
table{display:block;max-width:100%;overflow-x:auto;white-space:normal;border-collapse:separate;border-spacing:0;border-radius:12px}
th,td{padding:12px 14px;border-width:0 1px 1px 0;min-width:100px}
th:first-child,td:first-child{border-left:1px solid var(--line)}
th{border-top:1px solid var(--line);color:#244157;font-size:13px}
th:first-child{border-top-left-radius:10px}
th:last-child{border-top-right-radius:10px}
.result-img{box-shadow:0 8px 26px rgba(16,42,67,.1)}
pre,.result{overflow-wrap:anywhere;line-height:1.75}
a{font-weight:600}
a:focus-visible,button:focus-visible,.button:focus-visible{outline:3px solid #45b8ad;outline-offset:3px}
.footer{padding:26px 8px 18px;line-height:1.7}
@media(max-width:760px){
 .wrap{margin:12px auto;padding:0 12px}
 .hero{padding:25px 20px;border-radius:19px}
 .hero h1{letter-spacing:-1.1px}
 .card{padding:18px;border-radius:16px}
 .grid,.grid3{grid-template-columns:minmax(0,1fr);gap:14px}
 .nav{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
 .nav a{text-align:center;padding:10px 8px;font-size:13px}
 .nav .right{grid-column:1/-1;margin:0}
 .nav .right button{width:100%}
 .full{width:100%}
 th,td{padding:10px;font-size:13px}
}
@media(prefers-reduced-motion:reduce){
 html{scroll-behavior:auto}
 *,*:before,*:after{transition:none!important;animation:none!important}
}

'''

LAYOUT = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{{ title or 'Bone Tumor Classification' }}</title><style>''' + BASE_CSS + '''</style></head><body><main class="wrap"><header class="hero"><div class="eyebrow">FINAL YEAR AIML PROJECT · RESEARCH PROTOTYPE</div><h1>Bone Tumor Classification</h1><p>ResNet50 image classification · Grad-CAM visual explanation · Consultation preparation</p></header>{% with messages=get_flashed_messages(with_categories=true) %}{% for category,message in messages %}<div class="alert {{ 'success' if category=='success' else '' }}">{{ message }}</div>{% endfor %}{% endwith %}{{ body|safe }}<div class="footer">Academic research prototype only. Not a diagnostic service. A qualified clinician must review symptoms and medical images.</div></main></body></html>'''

def page(body, title='Bone Tumor Classification', **kwargs):
    return render_template_string(LAYOUT, body=render_template_string(body, **kwargs), title=title)

def db_ready():
    return CONFIGURED

def current_record():
    sid = request.cookies.get('btc_session')
    if not sid or not CONFIGURED: return None
    try:
        rows = admin.table('web_sessions').select('*').eq('session_id', sid).limit(1).execute().data
        if not rows: return None
        rec = rows[0]
        # Opaque random cookie only; no auth tokens are placed in the browser cookie.
        if rec.get('expires_at') and datetime.fromisoformat(rec['expires_at'].replace('Z','+00:00')) < datetime.now(timezone.utc):
            admin.table('web_sessions').delete().eq('session_id', sid).execute()
            return None
        return rec
    except Exception as exc:
        print('Session lookup failed:', repr(exc)); return None

def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        rec = current_record()
        if not rec:
            flash('Please sign in to access your private workspace.')
            return redirect(url_for('login'))
        request.user_session = rec
        return fn(*args, **kwargs)
    return wrapped

def create_server_session(auth_session, user):
    sid = secrets.token_urlsafe(40)
    expires = (datetime.now(timezone.utc) + __import__('datetime').timedelta(days=7)).isoformat()
    admin.table('web_sessions').insert({'session_id':sid,'user_id':user.id,'expires_at':expires}).execute()
    return sid

@app.context_processor
def inject_csrf_token():
    def csrf_token():
        token = session.get('_csrf_token')
        if not token:
            token = secrets.token_urlsafe(32)
            session['_csrf_token'] = token
        return token
    return {'csrf_token': csrf_token}

@app.before_request
def protect_post_requests():
    if request.method == 'POST':
        expected = session.get('_csrf_token', '')
        supplied = request.form.get('csrf_token', '')
        if not expected or not supplied or not secrets.compare_digest(expected, supplied):
            abort(400, description='Invalid or expired form token. Refresh the page and try again.')

@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['X-Frame-Options']='DENY'
    response.headers['Referrer-Policy']='same-origin'
    response.headers['Cache-Control']='no-store'
    if request.is_secure:
        response.headers['Strict-Transport-Security']='max-age=31536000; includeSubDomains'
    return response

@app.route('/')
def index():
    rec=current_record()
    if rec: return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/health')
def health():
    return {'status':'ok','supabase_configured':CONFIGURED,'model_loaded':predictor is not None}, 200

@app.route('/login', methods=['GET','POST'])
def login():
    if not CONFIGURED:
        body='''<section class="card"><h2>Connect your database first</h2><p>The website code is ready, but online accounts need a Supabase project. Follow <code>DEPLOY_ONLINE.md</code> to set the three Supabase environment variables on your hosting service.</p><p><b>Required:</b> SUPABASE_URL, SUPABASE_ANON_KEY and SUPABASE_SERVICE_ROLE_KEY. Never share the service-role key publicly.</p></section>'''
        return page(body, title='Setup · Bone Tumor Classification')
    if request.method=='POST':
        email=(request.form.get('email') or '').strip().lower(); password=request.form.get('password') or ''
        try:
            auth_client=create_client(SUPABASE_URL,SUPABASE_ANON_KEY)
            result=auth_client.auth.sign_in_with_password({'email':email,'password':password})
            if not result.user or not result.session: raise ValueError('Email or password was not accepted. If email verification is enabled, verify your email first.')
            sid=create_server_session(result.session,result.user)
            admin.table('user_profiles').upsert({'user_id':result.user.id,'email':email},on_conflict='user_id').execute()
            resp=redirect(url_for('dashboard')); resp.set_cookie('btc_session',sid,httponly=True,secure=(request.is_secure or bool(os.environ.get('RENDER'))),samesite='Lax',max_age=60*60*24*7,path='/'); return resp
        except Exception as exc:
            flash('Sign in failed. Check your email/password and confirm your email if requested. Details: '+str(exc)[:220])
    body='''<div class="grid"><section class="card"><h2>Welcome back</h2><p>Sign in to access your private analysis workspace, saved consultations, profile and uploaded files.</p><div class="warning"><b>Research use only</b><br>This tool cannot diagnose cancer or rule it out.</div></section><section class="card"><h2>Sign in</h2><form method="post"><input type="hidden" name="csrf_token" value="{{ csrf_token() }}"><label class="label">Email address</label><input type="email" name="email" autocomplete="email" required><label class="label">Password</label><input type="password" name="password" autocomplete="current-password" required><button class="full" type="submit">Sign in to workspace</button></form><p class="muted">New here? <a href="{{ url_for('register') }}">Create an account</a></p></section></div>'''
    return page(body)

@app.route('/register', methods=['GET','POST'])
def register():
    if not CONFIGURED: return redirect(url_for('login'))
    if request.method=='POST':
        email=(request.form.get('email') or '').strip().lower(); password=request.form.get('password') or ''; confirm=request.form.get('confirm') or ''; name=(request.form.get('name') or '').strip()
        if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+',email): flash('Enter a valid email address.')
        elif len(password)<10: flash('Use a password with at least 10 characters.')
        elif password != confirm: flash('Passwords do not match.')
        else:
            try:
                auth_client=create_client(SUPABASE_URL,SUPABASE_ANON_KEY)
                result=auth_client.auth.sign_up({'email':email,'password':password,'options':{'data':{'display_name':name}}})
                if result.session and result.user:
                    sid=create_server_session(result.session,result.user)
                    admin.table('user_profiles').upsert({'user_id':result.user.id,'email':email,'display_name':name},on_conflict='user_id').execute()
                    resp=redirect(url_for('dashboard')); resp.set_cookie('btc_session',sid,httponly=True,secure=(request.is_secure or bool(os.environ.get('RENDER'))),samesite='Lax',max_age=60*60*24*7,path='/'); return resp
                flash('Account request received. Check your email to verify the address, then sign in.')
                return redirect(url_for('login'))
            except Exception as exc: flash('Could not create account: '+str(exc)[:220])
    body='''<section class="card" style="max-width:650px;margin:auto"><h2>Create your account</h2><p class="muted">Accounts are managed by Supabase Auth. Passwords are not stored in this application database.</p><form method="post"><input type="hidden" name="csrf_token" value="{{ csrf_token() }}"><label class="label">Name (optional)</label><input name="name" autocomplete="name" maxlength="100"><label class="label">Email address</label><input type="email" name="email" autocomplete="email" required><label class="label">Password (10+ characters)</label><input type="password" name="password" minlength="10" autocomplete="new-password" required><label class="label">Confirm password</label><input type="password" name="confirm" minlength="10" autocomplete="new-password" required><button class="full" type="submit">Create secure account</button></form><p class="muted">Already registered? <a href="{{ url_for('login') }}">Sign in</a></p></section>'''
    return page(body)

@app.route('/logout', methods=['POST'])
def logout():
    sid=request.cookies.get('btc_session')
    if sid and CONFIGURED:
        try: admin.table('web_sessions').delete().eq('session_id',sid).execute()
        except Exception: pass
    resp=redirect(url_for('login')); resp.delete_cookie('btc_session',path='/'); return resp

@app.route('/dashboard')
@login_required
def dashboard():
    rec=request.user_session; uid=rec['user_id']
    profile={}
    try:
        rows=admin.table('user_profiles').select('*').eq('user_id',uid).limit(1).execute().data
        if rows: profile=rows[0]
    except Exception: pass
    metrics=test_metrics or {}
    body='''<nav class="nav"><a href="{{ url_for('dashboard') }}">Dashboard</a><a href="{{ url_for('analyze') }}">X-ray analysis</a><a href="{{ url_for('consultation') }}">Doctor guide</a><a href="{{ url_for('history') }}">Saved history</a><a href="{{ url_for('profile') }}">Profile</a><form class="right" method="post" action="{{ url_for('logout') }}"><input type="hidden" name="csrf_token" value="{{ csrf_token() }}"><button class="secondary" type="submit">Sign out</button></form></nav><section class="card"><h2>Welcome{% if profile.get('display_name') %}, {{ profile.get('display_name')|e }}{% endif %}</h2><p class="muted">Signed in as {{ profile.get('email','')|e }}. Your saved consultation records and files are private to your account in this application.</p><div class="grid3"><div class="metric"><small>Test accuracy</small><strong>{{ '%.1f'|format(metrics.get('accuracy',0)*100) }}%</strong></div><div class="metric"><small>ROC-AUC</small><strong>{{ '%.3f'|format(metrics.get('roc_auc',0)) }}</strong></div><div class="metric"><small>Model status</small><strong style="font-size:20px">{{ 'Loaded' if model_loaded else 'Unavailable' }}</strong></div></div></section><div class="grid"><section class="card"><h3>Analyse an X-ray</h3><p>Upload an image to view benign/malignant model scores and a Grad-CAM overlay.</p><a class="button" href="{{ url_for('analyze') }}">Open image analysis</a></section><section class="card"><h3>Prepare for a consultation</h3><p>Organise symptoms, previous imaging, and questions to discuss with a qualified clinician.</p><a class="button" href="{{ url_for('consultation') }}">Create doctor-visit guide</a></section><section class="card"><h3>Saved history</h3><p>Review consultation summaries and files associated with your account.</p><a class="button secondary" href="{{ url_for('history') }}">View my history</a></section><section class="card"><h3>About the model</h3><p>ResNet50 transfer learning with Grad-CAM explainability. Test metrics describe the held-out dataset only, not clinical performance for an individual.</p><a class="button secondary" href="{{ url_for('model_info') }}">Model analysis</a></section></div><div class="warning"><b>Important medical limitation:</b> this academic prototype is not a diagnostic service. A benign model output does not rule out malignancy. Seek clinical assessment for persistent or worsening symptoms.</div>'''
    return page(body, profile=profile, metrics=metrics, model_loaded=predictor is not None)

@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    rec=request.user_session; uid=rec['user_id']
    rows=admin.table('user_profiles').select('*').eq('user_id',uid).limit(1).execute().data
    p=rows[0] if rows else {'email':''}
    if request.method=='POST':
        name=(request.form.get('display_name') or '').strip()[:100]; phone=(request.form.get('phone') or '').strip()[:30]
        try:
            admin.table('user_profiles').upsert({'user_id':uid,'email':p.get('email',''),'display_name':name,'phone':phone,'updated_at':datetime.now(timezone.utc).isoformat()},on_conflict='user_id').execute(); flash('Profile saved.','success'); return redirect(url_for('profile'))
        except Exception as exc: flash('Could not save profile: '+str(exc)[:180])
    body='''<nav class="nav"><a href="{{ url_for('dashboard') }}">← Dashboard</a><a href="{{ url_for('history') }}">Saved history</a></nav><section class="card"><h2>My profile</h2><p class="muted">Avoid adding sensitive medical details to profile fields.</p><form method="post"><input type="hidden" name="csrf_token" value="{{ csrf_token() }}"><label class="label">Email</label><input value="{{ p.get('email','')|e }}" disabled><label class="label">Display name</label><input name="display_name" value="{{ p.get('display_name','')|e }}" maxlength="100"><label class="label">Phone (optional)</label><input name="phone" value="{{ p.get('phone','')|e }}" maxlength="30"><button class="full" type="submit">Save profile</button></form></section>'''
    return page(body,p=p)

@app.route('/analyze', methods=['GET','POST'])
@login_required
def analyze():
    rec=request.user_session; uid=rec['user_id']; result=None
    if request.method=='POST':
        f=request.files.get('xray'); consent=request.form.get('save_file')=='yes'
        
        if not f or not f.filename:
            flash('Choose an X-ray image first.')
        elif not C.CHECKPOINT.exists():
            flash('The model checkpoint is not available on this deployment.')
        else:
            try:

                data=f.read()
                if len(data)>15*1024*1024: raise ValueError('File exceeds 15 MB limit.')
                img=Image.open(io.BytesIO(data)); img.load()
                if min(img.size)<64: raise ValueError('Image must be at least 64×64 pixels.')
                if np.asarray(standardize_xray(img)).std()<2: raise ValueError('Image appears blank. Please choose a visible X-ray.')
                prediction = get_predictor().analyze(img)
                explanation=explain(prediction['probs'],prediction['cam_info'],bands,test_metrics)
                # Store a temporary rendered heatmap in this response; persist uploaded original only if user chooses.
                buf=io.BytesIO(); prediction['heatmap'].save(buf,format='PNG'); heatmap_b64=__import__('base64').b64encode(buf.getvalue()).decode()
                saved_msg='Original X-ray not saved; you can choose to save it below.'
                if consent:
                    ext=Path(secure_filename(f.filename)).suffix.lower()
                    if ext not in ['.png','.jpg','.jpeg','.tif','.tiff','.bmp']: ext='.png'
                    path=f'{uid}/{uuid.uuid4().hex}{ext}'
                    # Supabase private bucket, user-scoped path. Content-type inferred from extension.
                    admin.storage.from_(BUCKET).upload(path,data,{'content-type':f.mimetype or 'application/octet-stream','upsert':'false'})
                    admin.table('uploaded_files').insert({'user_id':uid,'storage_path':path,'original_filename':secure_filename(f.filename)[:180],'content_type':f.mimetype or 'application/octet-stream','file_size':len(data),'file_kind':'xray'}).execute()
                    saved_msg='Original X-ray saved privately to your account.'
                admin.table('consultations').insert({'user_id':uid,'record_type':'image_analysis','title':'X-ray analysis','summary':json.dumps({'scores':prediction['probs'],'predicted_class':max(prediction['probs'],key=prediction['probs'].get),'explanation':explanation,'saved_image':consent}), 'created_at':datetime.now(timezone.utc).isoformat()}).execute()
                result={'scores':prediction['probs'],'heatmap':heatmap_b64,'explanation':explanation,'saved_msg':saved_msg}
                flash('Analysis completed and added to your history.','success')
            except (UnidentifiedImageError,OSError,Image.DecompressionBombError): flash('Could not read this file as an image. Upload a PNG, JPEG, BMP or TIFF X-ray.')
            except Exception as exc: flash('Analysis or saving failed: '+str(exc)[:220])
    body='''<nav class="nav"><a href="{{ url_for('dashboard') }}">← Dashboard</a><a href="{{ url_for('history') }}">Saved history</a></nav><section class="card"><h2>X-ray analysis</h2><p>Upload a bone-tumour radiograph. The model produces benign/malignant class scores and a Grad-CAM overlay. Scores are not calibrated medical probabilities.</p><form method="post" enctype="multipart/form-data"><input type="hidden" name="csrf_token" value="{{ csrf_token() }}"><label class="label">X-ray image (PNG/JPG/TIFF/BMP, max 15 MB)</label><input type="file" name="xray" accept="image/png,image/jpeg,image/tiff,image/bmp" required><label style="display:flex;gap:10px;align-items:flex-start;margin-top:14px;font-weight:500"><input type="checkbox" name="save_file" value="yes" style="width:auto;margin-top:5px"><span>Save the original X-ray privately to my account for later access. Leave unchecked for one-time analysis.</span></label><button class="full" type="submit">Analyse X-ray</button></form></section>{% if result %}<section class="card"><h2>Model output</h2><div class="grid">{% for label,score in result.scores.items() %}<div class="metric"><small>{{ label }} model score</small><strong>{{ '%.1f'|format(score*100) }}%</strong></div>{% endfor %}</div><h3>Grad-CAM overlay</h3><img class="result-img" alt="Grad-CAM overlay" src="data:image/png;base64,{{ result.heatmap }}"><p class="muted">Red/yellow regions show image areas that influenced the prediction; they are not confirmed tumour boundaries.</p><div class="result">{{ result.explanation|e }}</div><p class="muted">{{ result.saved_msg }}</p></section>{% endif %}<div class="warning">A benign result cannot rule out cancer. Discuss symptoms and original images with a qualified clinician.</div>'''
    return page(body,result=result)

@app.route('/consultation', methods=['GET','POST'])
@login_required
def consultation():
    rec=request.user_session; uid=rec['user_id']
    if request.method=='POST':
        data={k:(request.form.get(k) or '').strip() for k in ['age','location','duration','change','imaging','injury','symptoms','notes']}
        data['symptoms_list']=request.form.getlist('symptoms_list')
        # User-generated fields are escaped by Jinja when displayed. Avoid collecting identifying info unnecessarily.
        summary=make_guide(data)
        try:
            admin.table('consultations').insert({'user_id':uid,'record_type':'doctor_guide','title':'Doctor consultation preparation','summary':json.dumps({'answers':data,'guide':summary}), 'created_at':datetime.now(timezone.utc).isoformat()}).execute()
            flash('Consultation guide saved to your account.','success')
        except Exception as exc: flash('Guide created but could not be saved: '+str(exc)[:180])
        return page('''<nav class="nav"><a href="{{ url_for('dashboard') }}">← Dashboard</a><a href="{{ url_for('history') }}">Saved history</a></nav><section class="card"><h2>Your doctor-visit preparation guide</h2><div class="result">{{ guide }}</div><p><a class="button" href="{{ url_for('consultation') }}">Create another guide</a></p></section>''',guide=summary)
    body='''<nav class="nav"><a href="{{ url_for('dashboard') }}">← Dashboard</a><a href="{{ url_for('history') }}">Saved history</a></nav><section class="card"><h2>Doctor / radiologist consultation guide</h2><p>This form prepares a summary and questions to take to a clinician. It does not diagnose symptoms or determine urgency.</p><form method="post"><input type="hidden" name="csrf_token" value="{{ csrf_token() }}"><div class="grid"><div><label class="label">Age (optional)</label><input name="age" maxlength="3" inputmode="numeric"><label class="label">Area of concern</label><select name="location"><option>Upper arm / shoulder</option><option>Forearm / wrist</option><option>Hip / pelvis</option><option>Thigh / knee</option><option>Lower leg / ankle</option><option>Foot / hand</option><option>Other / not sure</option></select><label class="label">How long?</label><select name="duration"><option>New / today</option><option>Several days</option><option>Several weeks</option><option>Months or longer</option><option>No symptoms; finding on imaging</option><option>Not sure</option></select><label class="label">Change over time</label><select name="change"><option>Improving</option><option>Stable</option><option>Getting worse</option><option>Comes and goes</option><option>Not sure</option></select></div><div><label class="label">Symptoms (select any)</label>{% for s in symptoms %}<label style="display:flex;gap:8px;align-items:flex-start;margin:8px 0;font-weight:500"><input type="checkbox" name="symptoms_list" value="{{ s }}" style="width:auto;margin-top:5px">{{ s }}</label>{% endfor %}<label class="label">Previous imaging/report</label><select name="imaging"><option>None yet</option><option>X-ray only</option><option>MRI</option><option>CT</option><option>More than one type</option><option>Have a report but unsure</option><option>Not provided</option></select><label class="label">Recent injury?</label><select name="injury"><option>No</option><option>Yes</option><option>Not sure / prefer not to say</option></select></div></div><label class="label">Anything else you want to remember to tell the doctor? (optional)</label><textarea name="notes" maxlength="1500" placeholder="Keep this brief; do not enter national ID, payment details, or unnecessary personal identifiers."></textarea><button class="full" type="submit">Generate and save consultation guide</button></form></section>'''
    return page(body,symptoms=['Persistent bone pain','Pain at night','Swelling or lump','Reduced movement','Sudden severe pain','Unable to bear weight/use limb','Rapidly increasing swelling','Fever or feeling unwell','No symptoms'])

def make_guide(d):
    sym=', '.join(d.get('symptoms_list',[])) or 'None selected'
    urgent=any(x in d.get('symptoms_list',[]) for x in ['Sudden severe pain','Unable to bear weight/use limb','Rapidly increasing swelling'])
    urgency='If you have sudden severe pain, cannot bear weight/use the limb, or rapidly increasing swelling, seek urgent in-person medical assessment; use local emergency services for severe symptoms.' if urgent else 'If symptoms become severe or rapidly worsen, or you cannot use/bear weight on the limb, seek urgent in-person medical assessment.'
    return f'''CONSULTATION PREPARATION — NOT A DIAGNOSIS\n\nSummary to tell the clinician\n• Age: {d.get('age') or 'Not provided'}\n• Area: {d.get('location') or 'Not specified'}\n• Duration: {d.get('duration') or 'Not specified'}\n• Change: {d.get('change') or 'Not specified'}\n• Symptoms: {sym}\n• Prior imaging/report: {d.get('imaging') or 'Not provided'}\n• Recent injury: {d.get('injury') or 'Not provided'}\n• Additional notes: {d.get('notes') or 'None'}\n\nQuestions to ask\n1. What exactly does the radiology report describe, and what remains uncertain?\n2. Does the finding appear benign, suspicious, or indeterminate, and what features support that wording?\n3. Is MRI, CT, comparison with older images, or another test appropriate? Why?\n4. Should I see an orthopaedic clinician or orthopaedic oncology/bone-tumour specialist?\n5. If biopsy is considered, which specialist team should plan it?\n6. Which symptoms should prompt earlier reassessment?\n\nWho does what\n• Radiologist: interprets medical images and produces a report.\n• Treating doctor/orthopaedic clinician: considers symptoms, examination and imaging together.\n• Orthopaedic oncology specialist: may assess a suspected bone tumour when referral is appropriate.\n\nBring original images (not only screenshots), all reports, prior scans for comparison, a symptom timeline, medicines and referral letters.\n\nSafety: {urgency}\n\nThe AI classifier only scores an image as benign or malignant based on training data. It cannot determine the cause of symptoms, confirm cancer, or recommend treatment. A qualified clinician must review the images and clinical context.'''

@app.route('/history')
@login_required
def history():
    uid=request.user_session['user_id']
    try:
        consultations=admin.table('consultations').select('id,title,record_type,summary,created_at').eq('user_id',uid).order('created_at',desc=True).limit(50).execute().data
        files=admin.table('uploaded_files').select('id,original_filename,file_size,file_kind,created_at').eq('user_id',uid).order('created_at',desc=True).limit(50).execute().data
    except Exception as exc:
        flash('Could not load history: '+str(exc)[:180]); consultations=[]; files=[]
    # Show only selected safe metadata; summary JSON can contain user-provided medical details.
    body='''<nav class="nav"><a href="{{ url_for('dashboard') }}">← Dashboard</a><a href="{{ url_for('analyze') }}">X-ray analysis</a><a href="{{ url_for('consultation') }}">Doctor guide</a></nav><section class="card"><h2>My saved consultation history</h2>{% if consultations %}<table><tr><th>Date</th><th>Type</th><th>Title</th><th>View</th></tr>{% for c in consultations %}<tr><td>{{ c.get('created_at','')[:19].replace('T',' ') }}</td><td>{{ c.get('record_type','')|e }}</td><td>{{ c.get('title','')|e }}</td><td><a href="{{ url_for('view_consultation',record_id=c.get('id')) }}">Open</a></td></tr>{% endfor %}</table>{% else %}<p class="muted">No saved consultations yet.</p>{% endif %}</section><section class="card"><h2>My uploaded files</h2><p class="muted">Files are stored in a private bucket under your account. Downloads are generated only after ownership is checked.</p><form method="post" enctype="multipart/form-data" action="{{ url_for('upload_report') }}"><input type="hidden" name="csrf_token" value="{{ csrf_token() }}"><label class="label">Upload a radiology report (PDF or image, max 15 MB)</label><input type="file" name="report" accept="application/pdf,image/*" required><button class="full" type="submit">Save report privately</button></form>{% if files %}<table><tr><th>Date</th><th>Filename</th><th>Type</th><th>Size</th><th>Download</th></tr>{% for f in files %}<tr><td>{{ f.get('created_at','')[:19].replace('T',' ') }}</td><td>{{ f.get('original_filename','')|e }}</td><td>{{ f.get('file_kind','')|e }}</td><td>{{ (f.get('file_size',0)/1024)|round|int }} KB</td><td><a href="{{ url_for('download_file',file_id=f.get('id')) }}">Create private download link</a></td></tr>{% endfor %}</table>{% else %}<p class="muted">No uploaded files. On the analysis page, choose the private-save checkbox if you want to keep an original X-ray.</p>{% endif %}</section>'''
    return page(body,consultations=consultations,files=files)

@app.route('/history/consultation/<record_id>')
@login_required
def view_consultation(record_id):
    uid=request.user_session['user_id']
    rows=admin.table('consultations').select('*').eq('id',record_id).eq('user_id',uid).limit(1).execute().data
    if not rows: abort(404)
    r=rows[0]; raw=r.get('summary','')
    try:
        d=json.loads(raw); text=d.get('guide') or d.get('explanation') or json.dumps(d,indent=2)
    except Exception: text=raw
    body='''<nav class="nav"><a href="{{ url_for('history') }}">← Saved history</a></nav><section class="card"><h2>{{ record.get('title','Saved record')|e }}</h2><p class="muted">{{ record.get('created_at','')[:19].replace('T',' ') }}</p><pre style="white-space:pre-wrap;font:inherit">{{ content }}</pre></section>'''
    return page(body,record=r,content=text)

@app.route('/files/<file_id>/download')
@login_required
def download_file(file_id):
    uid=request.user_session['user_id']
    rows=admin.table('uploaded_files').select('*').eq('id',file_id).eq('user_id',uid).limit(1).execute().data
    if not rows: abort(404)
    f=rows[0]
    try:
        signed=admin.storage.from_(BUCKET).create_signed_url(f['storage_path'],60)
        url=signed.get('signedURL') or signed.get('signedUrl')
        if not url: raise ValueError('No signed URL returned')
        return redirect(url)
    except Exception as exc:
        flash('Could not create a private download link: '+str(exc)[:160]); return redirect(url_for('history'))

@app.route('/files/upload-report', methods=['POST'])
@login_required
def upload_report():
    uid=request.user_session['user_id']; f=request.files.get('report')
    if not f or not f.filename:
        flash('Choose a report file first.'); return redirect(url_for('history'))
    data=f.read(); name=secure_filename(f.filename); ext=Path(name).suffix.lower()
    allowed={'.pdf': 'application/pdf', '.png':'image/png', '.jpg':'image/jpeg', '.jpeg':'image/jpeg', '.tif':'image/tiff', '.tiff':'image/tiff', '.bmp':'image/bmp'}
    if ext not in allowed:
        flash('Report must be PDF or an image (PDF, PNG, JPG, TIFF, BMP).'); return redirect(url_for('history'))
    if not data or len(data)>15*1024*1024:
        flash('File is empty or exceeds the 15 MB limit.'); return redirect(url_for('history'))
    path=f'{uid}/{uuid.uuid4().hex}{ext}'
    try:
        admin.storage.from_(BUCKET).upload(path,data,{'content-type':allowed[ext],'upsert':'false'})
        admin.table('uploaded_files').insert({'user_id':uid,'storage_path':path,'original_filename':name[:180],'content_type':allowed[ext],'file_size':len(data),'file_kind':'report'}).execute()
        flash('Report uploaded privately to your account.','success')
    except Exception as exc: flash('Report upload failed: '+str(exc)[:180])
    return redirect(url_for('history'))

@app.route('/model-analysis')
@login_required
def model_info():
    m=test_metrics or {}; cm=m.get('confusion_matrix',[[0,0],[0,0]]); tn,fp=cm[0]; fn,tp=cm[1]
    body='''<nav class="nav"><a href="{{ url_for('dashboard') }}">← Dashboard</a></nav><section class="card"><h2>Model analysis and limitations</h2>{% if m %}<div class="grid3"><div class="metric"><small>Test accuracy</small><strong>{{ '%.1f'|format(m.get('accuracy',0)*100) }}%</strong></div><div class="metric"><small>ROC-AUC</small><strong>{{ '%.3f'|format(m.get('roc_auc',0)) }}</strong></div><div class="metric"><small>Malignant recall</small><strong>{{ '%.1f'|format(m.get('recall',0)*100) }}%</strong></div></div><h3>Confusion matrix (actual rows, predicted columns)</h3><table><tr><th>Actual class</th><th>Predicted benign</th><th>Predicted malignant</th></tr><tr><th>Benign</th><td>{{ tn }} correct</td><td>{{ fp }} false alarms</td></tr><tr><th>Malignant</th><td>{{ fn }} missed cases</td><td>{{ tp }} correct</td></tr></table><p>The test set contains {{ m.get('n_test',0) }} images. The model detected {{ tp }} of {{ tp+fn }} malignant cases and missed {{ fn }}. This illustrates why accuracy alone is insufficient.</p>{% else %}<p>Evaluation metrics are not available in outputs/test_metrics.json.</p>{% endif %}<h3>Methodology</h3><p>Input validation → X-ray standardisation and resizing → ResNet50 transfer learning → benign/malignant scores → Grad-CAM overlay.</p><h3>Limitations</h3><ul><li>Test set is small and class-imbalanced.</li><li>Patient IDs are not available to verify patient-level independence across splits.</li><li>External hospital validation has not been demonstrated.</li><li>Model scores are not calibrated medical probabilities; Grad-CAM is not a tumour boundary.</li><li>This is an academic prototype, not a diagnostic device.</li></ul></section>'''
    return page(body,m=m,tn=tn,fp=fp,fn=fn,tp=tp)

@app.errorhandler(413)
def too_large(_): return page('<section class="card"><h2>File too large</h2><p>Maximum upload size is 15 MB.</p><a href="/analyze">Return to analysis</a></section>'),413

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT','7860')),debug=False)
