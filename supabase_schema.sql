-- Run this entire file in Supabase Dashboard > SQL Editor > New query.
-- The web server uses the SERVICE ROLE key only on the server. Never expose it in frontend code.
create table if not exists public.user_profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  email text not null,
  display_name text not null default '',
  phone text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.consultations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  record_type text not null check (record_type in ('doctor_guide','image_analysis')),
  title text not null default 'Saved consultation',
  summary text not null default '{}',
  created_at timestamptz not null default now()
);
create index if not exists consultations_user_date_idx on public.consultations(user_id, created_at desc);

create table if not exists public.uploaded_files (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  storage_path text not null unique,
  original_filename text not null,
  content_type text not null default 'application/octet-stream',
  file_size bigint not null default 0,
  file_kind text not null default 'xray' check (file_kind in ('xray','report')),
  created_at timestamptz not null default now()
);
create index if not exists uploaded_files_user_date_idx on public.uploaded_files(user_id, created_at desc);

-- Server-side sessions: browser receives only an opaque random session ID.
-- The browser receives only an opaque random session ID, not a Supabase token.
create table if not exists public.web_sessions (
  session_id text primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  expires_at timestamptz not null,
  created_at timestamptz not null default now()
);
create index if not exists web_sessions_user_idx on public.web_sessions(user_id);

-- The app uses the server-only service-role key for database operations and checks user_id
-- on every query. Enable RLS as defense-in-depth; do not create public/anon policies.
alter table public.user_profiles enable row level security;
alter table public.consultations enable row level security;
alter table public.uploaded_files enable row level security;
alter table public.web_sessions enable row level security;

-- Create a private bucket. If it already exists, this insert does nothing.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('private-medical-files', 'private-medical-files', false, 15728640,
        array['image/png','image/jpeg','image/tiff','image/bmp','application/pdf'])
on conflict (id) do update set public = false, file_size_limit = 15728640;
