# Deploying MediHaven

This guide deploys the **synthetic demo**: 100 generated patients, the 4-model ensemble, the vault and
the QR scanner. It is a student project and is not a medical device. Never put real patient data into it.

**Research mode (`DATA_MODE=mimic`) cannot be deployed.** MIMIC-IV data and models trained on it are
covered by a data use agreement, so they must stay on the researcher's own machine. `wsgi.py` refuses to
start in research mode, and research mode itself only binds to loopback. To run it, use `run_demo.py`
locally (see `docs/research/phase3_integration_security.md`).

## 1. What runs in production

| Piece | Choice |
| :--- | :--- |
| WSGI server | Waitress (`waitress-serve ... wsgi:app`): pure Python, runs on Linux and Windows. One process with threads, which suits SQLite and the in-memory alert state and rate limits. |
| Entry point | `wsgi.py`. It defaults `FLASK_ENV=production` and creates any missing demo assets on first start: it seeds the database, builds the feature store and trains the models. |
| Database | SQLite at `DATABASE_PATH` (default `database/medihaven.db`). Seeding commits once at the end, so an empty or interrupted database is seeded again on the next start. |
| Health check | `GET /api/health` |

Run a single instance. Several processes or replicas would each keep separate alert acknowledgements
and rate-limit counters, and SQLite is not built for multi-host writes.

## 2. Required settings

Set these in the platform's environment or secret settings, never in the image or the repository.

| Variable | Required | Notes |
| :--- | :--- | :--- |
| `FLASK_SECRET_KEY` | yes | Signs session cookies. At least 32 random characters: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `JWT_SECRET_KEY` | yes | Signs QR passes. Generate it separately, the same way. |
| `PASSWORD_HASH_VIDHI`, `PASSWORD_HASH_NIRUPAM`, `PASSWORD_HASH_PARAM` | for admin access | Each value is a password hash; generate one with `python -m src.api.auth`. In production the development admin passwords are refused, and an admin account without a hash cannot log in. |
| `DEMO_LOGINS` | optional | `1` enables the public demo accounts `physician1` and `patient1` and shows their credentials on the login page. The default is off in production. Anyone can then sign in as the demo physician, which is the point of a public demo. Leave it off otherwise and set `PASSWORD_HASH_PHYSICIAN1` / `PASSWORD_HASH_PATIENT1`. |
| `TRUST_PROXY_HOPS` | behind a proxy | Number of reverse proxies in front of the app. Most PaaS platforms use 1. The app then sees the real client address (used by the login rate limit) and the HTTPS scheme. |
| `GEMINI_API_KEY` | optional | Enables live narratives. Without it a local, deterministic fallback text is used. Narrative calls are limited to 30 per 5 minutes per client. |

The app **refuses to start** when `FLASK_SECRET_KEY` or `JWT_SECRET_KEY` is missing, a placeholder,
the public development default, or shorter than 32 characters. The same check applies to any
non-loopback `FLASK_HOST`, whichever launcher is used.

Production defaults, which you can override but normally should not:

- `SESSION_COOKIE_SECURE=1`: the session cookie is only sent over HTTPS. Set `0` only to test over plain HTTP.
- `FLASK_DEBUG` is ignored: the debugger is never enabled in production.
- `DEMO_MODE=0`: clinicians cannot mint QR passes for patients; only patients share their own vault.
  `DEMO_MODE=1` re-enables the scanner's "instant demo pass" buttons, but it bypasses patient consent.
- `SESSION_HOURS=8`: sign-in lifetime.

Serve over **HTTPS**. Browsers only allow the scanner camera and the "copy" button on secure origins.

## 3. Option A: Docker

```bash
docker build -t medihaven .
docker run -d --name medihaven -p 8000:8000 \
  -e FLASK_SECRET_KEY=... -e JWT_SECRET_KEY=... \
  -e PASSWORD_HASH_PARAM='scrypt:...' -e DEMO_LOGINS=1 \
  -e SESSION_COOKIE_SECURE=0 \
  -v medihaven-db:/app/database \
  medihaven
```

- `SESSION_COOKIE_SECURE=0` is only needed when testing at `http://localhost:8000`. Drop it behind HTTPS.
- The first start takes a few minutes while the demo data is seeded and the models are trained. The
  container health check allows for this.
- The volume keeps the database across restarts. Without it, each new container starts from a freshly
  seeded demo, which is fine for a demo.
- The image runs as an unprivileged user. It contains no `.env`, no local database and no model files
  (see `.dockerignore`).

## 4. Option B: a Python host or PaaS (Render, Railway, a VM)

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export FLASK_SECRET_KEY=... JWT_SECRET_KEY=... TRUST_PROXY_HOPS=1
python -m src.bootstrap                            # optional: prepare demo assets ahead of time
waitress-serve --host=0.0.0.0 --port=${PORT:-8000} --threads=8 wsgi:app
```

On a PaaS:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `waitress-serve --host=0.0.0.0 --port=$PORT --threads=8 wsgi:app`
- **Health check path:** `/api/health`

Use Python 3.12. On Linux, the optional `pyzbar` QR decoder needs the system library (`apt install
libzbar0`); without it, OpenCV decodes QR images instead.

## 5. Check after deploying

1. `GET /api/health` returns `"status": "OPERATIONAL"` and `"models_online": true`.
2. The response headers include `Content-Security-Policy`, `X-Frame-Options: DENY` and (over HTTPS)
   `Strict-Transport-Security`.
3. Logging in with a development admin password fails; your `PASSWORD_HASH_*` password works.
4. Open a patient chart on `/dashboard`: all four model votes show a tier and a score.

## 6. Security properties and known limits

Already in place:

- Signing secrets are checked at startup.
- Passwords are stored only as hashes.
- Login is rate-limited to 10 attempts per 5 minutes per client.
- Sessions use HttpOnly, SameSite=Lax, Secure cookies and expire after 8 hours.
- Security headers are set, including a CSP that allows only same-origin scripts.
- QR passes are bounded (at most 30 days and 1000 uses).
- Use counts are claimed atomically, so a single-use pass grants access once.
- Pass scope is read from the stored grant.
- Audit logs record the signed-in clinician.
- Image uploads to the scanner are size-capped.
- Error responses do not include internal details.

Known limits of this demo:

- Logout clears the browser's cookie. A copied session cookie stays valid until it expires (8 hours),
  because sessions are stateless.
- Alert acknowledgements and the admin activity feed are kept in memory and reset on restart.
- `GET /api/patients/<id>/predict` stores a prediction row on every call, as an audit history, so the
  table grows with use.
- Failed or forged pass scans are recorded with status "REVOKED" in the database audit table, whose
  schema allows only four statuses.
- The admin passwords used before this change are in the public Git history. Treat them as known and
  never reuse them.
