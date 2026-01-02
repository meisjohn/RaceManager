## Building and Running

### Local Development (Flask Dev Server)

1.  **Install dependencies**:
    ```bash
    python -m pip install -r requirements.txt
    ```
2.  **Run the application**:
    ```bash
    python src/app.py
    ```
    *Note: The application runs on port 5000 by default.*

### Docker

The included `Dockerfile` builds a production-ready image using Gunicorn.

1.  **Build the image**:
    ```bash
    docker build -t racemanager .
    ```
2.  **Run the container**:
    ```bash
    # Example running with local credentials mounted and environment variables set
    docker run -d -p 8080:8080 \
      --env FLASK_SECRET="your-secret" \
      --env USE_FIRESTORE=1 \
      --env GOOGLE_CLOUD_PROJECT="racemanager-482418" \
      --env FB_API_KEY="your-api-key" \
      --env FB_APP_ID="your-app-id" \
      --env GOOGLE_APPLICATION_CREDENTIALS="/app/config/service-account.json" \
      --env FIREBASE_CREDENTIALS="/app/config/service-account.json" \
      --volume /path/to/local/config:/app/config \
      --volume /path/to/local/qr:/app/src/static/qr \
      racemanager
    ```

### Production Execution (Without Docker)

For production environments not using Docker, use a WSGI server.

#### Linux / macOS (Gunicorn)

Gunicorn is included in `requirements.txt`.

```bash
export PORT=8080
export WEB_CONCURRENCY=3
gunicorn -w ${WEB_CONCURRENCY} -b 0.0.0.0:${PORT} --timeout 120 --access-logfile - --error-logfile - "src.app:app"
```

#### Windows (Waitress)

Gunicorn does not run on Windows. Use `waitress` instead.

1.  **Install Waitress**:
    ```bash
    pip install waitress
    ```
2.  **Run the application**:
    ```bash
    # PowerShell
    $Env:PORT=8080
    waitress-serve --port=$Env:PORT "src.app:app"
    
    # Command Prompt
    set PORT=8080
    waitress-serve --port=%PORT% "src.app:app"
    ```

## Admin Access

To access the administrative interface for the first time:

1.  Ensure `auth_config.json` is present in your `CONFIG_DIR` and contains an `admin_token`.
2.  Navigate to `/login?token=<your-admin-token>` (e.g., `http://localhost:5000/login?token=secret-admin-token`).
3.  Once logged in as Admin, you can access the dashboard at `/admin/ui` to manage races, users, and settings.

## Configuration

### Environment Variables

The application is configured via the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_SECRET` or `APP_SECRET` | Random string | Secret key for Flask sessions. **Must be set in production.** |
| `SESSION_COOKIE_SECURE` | `0` | Set to `1`, `true`, or `True` to require HTTPS for session cookies. |
| `SESSION_COOKIE_SAMESITE` | `Lax` | SameSite attribute for session cookies. |
| `LOG_LEVEL` | `INFO` | Logging level (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `DEFAULT_RACE` | `demo` | The default race ID to load if none is specified. |
| `CONFIG_DIR` | `config` | Directory path for local JSON configuration and data files. |
| `USE_FIRESTORE` | `0` | Set to `1`, `true`, or `yes` to use Google Cloud Firestore for storage. |
| `NUM_LANES` | `4` | Number of lanes on the track. |
| `CACHE_TTL` | `30` | In-memory cache duration in seconds for race data. |
| `FIREBASE_CREDENTIALS` | None | Path to Firebase Admin SDK service account JSON (required for local dev; Cloud Run uses ADC). |
| `FIREBASE_API_CREDENTIALS` | None | Path to a JSON file containing Firebase Web SDK config (apiKey, appId, etc.). |
| `FB_API_KEY` | `DEFAULT_API_KEY` | Fallback Firebase Web API Key if credentials file is missing. |
| `FB_APP_ID` | `DEFAULT_APP_ID` | Fallback Firebase Web App ID if credentials file is missing. |
| `FLASK_DEBUG` or `DEBUG` | `0` | Set to `1`, `true`, or `yes` to enable Flask debug mode. |

### Configuration Files

The application looks for the following JSON files in the `CONFIG_DIR` (default: `config/`):

*   **`patrol_config.json`**: Defines the mapping of patrol IDs to names.
    ```json
    { "1": "Foxes", "2": "Hawks", ... }
    ```
*   **`auth_config.json`**: Contains legacy authentication tokens and optional Firebase configuration.
    ```json
    {
      "owner_token": "secret-owner-token",
      "judge_token": "secret-judge-token",
      "admin_token": "secret-admin-token",
      "firebase": { "apiKey": "...", ... }
    }
    ```
*   **`race_data_{race_id}.json`**: Stores the complete state for a specific race (participants, heats, times) when `USE_FIRESTORE` is false.
*   **`race_members_{race_id}.json`**: Stores user roles (VIEWER, OWNER, JUDGE, ADMIN) for a specific race when `USE_FIRESTORE` is false.
*   **`race_archive_{race_id}.json`**: Stores archived race data when `USE_FIRESTORE` is false.

## Cloud Configuration

### Google Cloud Platform (GCP) Services

This application leverages the following GCP services:

*   **Cloud Run**: Serverless container execution.
*   **Artifact Registry**: Storage for Docker container images.
*   **Firestore (Native Mode)**: NoSQL database for race data and user roles.
*   **Secret Manager**: Secure storage for application secrets.

### Prerequisites

1.  **GCP Project**: Ensure you have a project (e.g., `racemanager-482418`).
2.  **Storage Buckets**: Create two buckets: one for private config and one for public QR codes.
    *   `racemanager-config` (Private)
    *   `racemanager-qr` (Public/Static assets)
2.  **Service Account**: Create a service account (e.g., `RaceManager-svc-acct`) with the following IAM roles:
    *   `Cloud Run Invoker`: To allow access to the service.
    *   `Cloud Datastore User`: To read/write to Firestore.
    *   `Storage Object Viewer`: To pull images from Artifact Registry (if private).
    *   `Secret Manager Secret Accessor`: If using Secret Manager.

### Deployment

#### 1. Build and Push Container

```bash
# Enable services
gcloud services enable run.googleapis.com artifactregistry.googleapis.com firestore.googleapis.com

# Create repository (one-time)
gcloud artifacts repositories create racemanager-images --repository-format=docker --location=us-east4

# Configure auth
gcloud auth configure-docker us-east4-docker.pkg.dev

# Build and Push
export PROJECT_ID="racemanager-482418"
docker build -t us-east4-docker.pkg.dev/${PROJECT_ID}/racemanager-images/racemanager:latest .
docker push us-east4-docker.pkg.dev/${PROJECT_ID}/racemanager-images/racemanager:latest
```

#### 2. Deploy to Cloud Run

```bash
gcloud run deploy race-manager \
  --image us-east4-docker.pkg.dev/${PROJECT_ID}/racemanager-images/racemanager:latest \
  --platform managed \
  --region us-east4 \
  --allow-unauthenticated \
  --service-account RaceManager-svc-acct@${PROJECT_ID}.iam.gserviceaccount.com \
  --set-env-vars "USE_FIRESTORE=1" \
  --set-secrets "FLASK_SECRET=flask-secret:latest,FB_API_KEY=firebase-api-key:latest,FB_APP_ID=firebase-app-id:latest" \
  --add-volume name=config-storage,type=cloud-storage,bucket=racemanager-config \
  --add-volume name=qr-storage,type=cloud-storage,bucket=racemanager-qr \
  --add-volume-mount volume=config-storage,mount-path=/app/config \
  --add-volume-mount volume=qr-storage,mount-path=/app/src/static/qr
```

### Firebase Integration

1.  **Project Link**: Link your GCP project to Firebase in the Firebase Console.
2.  **Authentication**: Enable Authentication providers (Google, Email/Password).
3.  **Firestore**: Ensure the database is created in Native mode.
    *   *Note: The application stores data at `races/{raceId}`.*
4.  **Security Rules**: Deploy `firestore.rules` to secure your database.
    ```bash
    firebase deploy --only firestore:rules
    ```
5.  **Client Config**: Update `config/auth_config.json` or set `FB_API_KEY` and `FB_APP_ID` environment variables so the frontend can initialize the Firebase SDK.

### Secrets Management

For production, avoid passing secrets as plain environment variables.

1.  Create a secret in Secret Manager (e.g., `flask-secret`).
2.  Grant the Cloud Run service account access to the secret.
3.  Mount the secret as an environment variable in Cloud Run:
    ```bash
    gcloud run deploy race-manager ... --update-secrets=FLASK_SECRET=flask-secret:latest
    ```
