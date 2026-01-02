# RaceManager — Cloud Run + Firestore migration notes
- Cloud Run & Gunicorn notes
- For production, run with Gunicorn (already used in `Dockerfile`). Recommended command:
```bash
# local test with Gunicorn
export PORT=8000
export WEB_CONCURRENCY=3
gunicorn -w ${WEB_CONCURRENCY} -b :${PORT} --timeout 120 --access-logfile - --error-logfile - "RaceManager.app:app"
```

When deploying to Cloud Run, set environment variables like `FLASK_SECRET`, `USE_FIRESTORE`, `LOG_LEVEL`, and `WEB_CONCURRENCY`.

- Quick overview
- The Flask app supports per-race data. Use the `race` query parameter (for example `/?race=PA-1249`) to switch between race datasets.
- Optional Firestore backend: set `USE_FIRESTORE=1` in the environment and grant the service account access to Firestore (Cloud Run Workload Identity recommended).

Files added
- `Dockerfile` — builds a container using `gunicorn` for Cloud Run.
- `firestore.rules` — example Firestore rules for per-race access and role enforcement.
- `RaceManager/static/firebase-client.html` — small client example to obtain Firebase ID tokens.

Running locally
1. (Optional) enable service account JSON and set `GOOGLE_APPLICATION_CREDENTIALS` if you want Firestore locally.
2. Install Python deps:
```bash
python -m pip install -r RaceManager/requirements.txt
```
3. Run locally (dev server):
```bash
python RaceManager/app.py
```

Windows notes
- Gunicorn is not supported natively on Windows (it requires `fcntl`). For Windows development you can either:
	- Use WSL or Docker and run Gunicorn as shown above, or
	- Install `waitress` and run the app with `waitress-serve` (there is a convenience requirements file `RaceManager/requirements-windows.txt`):

```bash
# from your venv
python -m pip install -r RaceManager/requirements-windows.txt
set PORT=8000   # PowerShell: $Env:PORT=8000
waitress-serve --port=$PORT "RaceManager.app:app"
```

Build and deploy to Cloud Run
1. Build image:
```bash
docker build -t gcr.io/PROJECT-ID/race-manager:latest .
```
2. Push and deploy (or use `gcloud builds submit`):
```bash
docker push gcr.io/PROJECT-ID/race-manager:latest
gcloud run deploy race-manager --image gcr.io/PROJECT-ID/race-manager:latest --platform managed --region YOUR_REGION --allow-unauthenticated
```

Environment variables
- `DEFAULT_RACE` — the default race id to load on startup (default: `default`).
- `USE_FIRESTORE` — set to `1` to use Firestore instead of local JSON files.

Firestore notes
- The Firestore document path used is `races/{raceId}` and the whole dataset is stored as fields on that document.
- Use `firestore.rules` as a starting point for security rules.

Next steps
- (Optional) move server-side role/management endpoints into `server/index.js` (Node) or add Flask endpoints to manage members and custom claims with the Firebase Admin SDK.
- Add admin UI to create races and assign members.

Bootstrap checklist (one-time tasks before production)
- Create a Firebase project and enable Firestore (Native mode).
- Create a service account for your Cloud Run service with roles: `roles/datastore.user` (or `roles/datastore.owner` as appropriate), `roles/iam.serviceAccountUser`, and `roles/logging.logWriter`.
- Locally: download the service account JSON and set `FIREBASE_CREDENTIALS` to its path or set `GOOGLE_APPLICATION_CREDENTIALS` to that file for testing.
- On Cloud Run: use Workload Identity / attach the service account to the service so the container can access Firestore without embedding keys.
- Create an initial top-level admin user:
	- Option A (recommended): use a short script with the Firebase Admin SDK and your service account to call `auth.setCustomUserClaims(uid, {'isTopAdmin': true})` for your bootstrap account.
	- Option B: use the Firebase Console (Users) and a cloud script to set custom claims.
- Add at least one race document or create a per-race data file `race_data_<raceId>.json` for each initial race.

Config values and parameters to change before production
- `DEFAULT_RACE` (env): default race id loaded on startup. Set to your actual production race id or remove default for explicit selection.
- `USE_FIRESTORE` (env): set to `1` in production to use Firestore instead of local JSON files.
- `FIREBASE_CREDENTIALS` (env): path to service account JSON for local testing. In Cloud Run prefer Workload Identity instead.
- `auth_config.json`: remove hard-coded tokens and use Firebase Auth for all users. Migrate owner/judge tokens to Firebase accounts or disable legacy tokens.
- `RaceManager/static/firebase-client.html`: replace `firebaseConfig` with your real Firebase Web SDK configuration before deploying.
- `firestore.rules`: review and tighten rules to match your exact security requirements (the shipped rules are a starting point).
- `requirements.txt`: pin stable package versions and update as needed.
- `Dockerfile`: review the base image (python:3.11-slim) and apply any organization-specific hardening.

Session and secret configuration (important)
- `app.secret_key`: the Flask app currently generates a random secret on startup. In production you must set a reproducible secret via an environment variable (for example `FLASK_SECRET` or `APP_SECRET`) and assign it to `app.secret_key` instead of the auto-generated value.
- `SESSION_COOKIE_SECURE`: set to `True` in production so cookies are only sent over HTTPS.
- `SESSION_COOKIE_SAMESITE`: set to `Lax` or `Strict` depending on your cross-site requirements. If your client is on a different domain, configure accordingly and use proper CORS.
- Consider `PERMANENT_SESSION_LIFETIME` to limit session duration.


Bootstrap commands (examples)
```bash
# Optional: set credentials locally
export FIREBASE_CREDENTIALS=/path/to/service-account.json

# Install deps
python -m pip install -r RaceManager/requirements.txt

# Run locally
python RaceManager/app.py

# Set an initial top admin via a short script (example using Firebase Admin):
# python set_top_admin.py --uid <UID> --project <PROJECT_ID>
```

Summary:
- API expects Firebase ID tokens in Authorization: Bearer <token>.
- Firestore structure: races/{raceId}, races/{raceId}/members/{uid}, races/{raceId}/data/{...}
- Top-level admins: set custom claim isTopAdmin on users (via admin SDK).

Local run:
1. Set GOOGLE_APPLICATION_CREDENTIALS to service account json.
2. npm install
3. npm start

Build & deploy to Cloud Run:
gcloud builds submit --tag gcr.io/PROJECT-ID/race-manager
gcloud run deploy race-manager --image gcr.io/PROJECT-ID/race-manager --platform managed --allow-unauthenticated --region YOUR_REGION

Notes:
- For production, use Workload Identity or attach IAM to allow service account access to Firestore without embedding keys.
- Create links for users by sharing URLs like https://<service>/race/PA-1249 or with query ?race=PA-1249.
