from glob import glob
import logging
import sys
from typing import Dict, List, Any
from flask import Flask, render_template, request, redirect, send_from_directory, url_for, \
    jsonify, Response, flash, session, g
import itertools
import json
import os
import os.path
from enum import Enum
import csv
from io import StringIO
import traceback
import uuid
import secrets
import qrcode
import re
from functools import wraps
import threading
import time

import firebase_admin
from firebase_admin import auth as fb_auth
from firebase_admin import credentials as fb_credentials


app = Flask(__name__)
# Use environment-provided secret in production; fall back to a random value for dev.
app.secret_key = os.environ.get('FLASK_SECRET') or os.environ.get('APP_SECRET') or secrets.token_urlsafe(16)

# Session cookie security recommended for production
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', '0') in ('1', 'true', 'True')
app.config['SESSION_COOKIE_SAMESITE'] = os.environ.get('SESSION_COOKIE_SAMESITE', 'Lax')

# Configure JSON logging to stdout so Cloud Run / GCP logging picks it up
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "lineno": record.lineno,
        }
        # include request id if present
        if hasattr(record, 'request_info'):
            log_record['request_info'] = dict(record.request_info)
        if hasattr(record, 'request_id') and record.request_id:
            log_record['request_id'] = record.request_id
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_record)


class RequestIDFilter(logging.Filter):
    def filter(self, record):
        try:
            record.request_id = getattr(g, 'request_id', None)
        except Exception:
            record.request_id = None
        return True


handler = logging.StreamHandler(sys.stdout)
handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
handler.setFormatter(JSONFormatter())
handler.addFilter(RequestIDFilter())

# Root logger configuration
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
root_logger.handlers = []
root_logger.addHandler(handler)

# App logger
logger = logging.getLogger('RaceManager')
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.propagate = False
logger.addHandler(handler)

# Make werkzeug use the same handler
logging.getLogger('werkzeug').handlers = [handler]
logging.getLogger('werkzeug').setLevel(logging.INFO)

logging.getLogger('urllib3.connectionpool').handlers = [handler]
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)

# Basic startup log (CURRENT_RACE_ID is set later)
logger.info("Starting RaceManager app; DEFAULT_RACE not yet resolved; LOG_LEVEL=%s", LOG_LEVEL)

# Optional Firestore support. Enable by setting USE_FIRESTORE=1 and
# providing Cloud Run service account or GOOGLE_APPLICATION_CREDENTIALS.
try:
    from google.cloud import firestore as _gcf
    FIRESTORE_AVAILABLE = True
except Exception:
    _gcf = None
    FIRESTORE_AVAILABLE = False

DEFAULT_RACE_ID = os.environ.get("DEFAULT_RACE", "demo")
CURRENT_RACE_ID = DEFAULT_RACE_ID
CONFIG_DIR = os.environ.get("CONFIG_DIR", "config")
USE_FIRESTORE = os.environ.get("USE_FIRESTORE", "0").lower() in ("1", "true", "yes")
DEFAULT_API_KEY="DEFAULT_API_KEY"
DEFAULT_APP_ID="DEFAULT_APP_ID"

# Config File templates and constants
DATA_FILE_TEMPLATE = "race_data_{race_id}.json"
ARCHIVE_FILE_TEMPLATE = "race_archive_{race_id}.json"
RACE_MEMBERS_FILE_TEMPLATE = "race_members_{race_id}.json"
PATROL_CONFIG_FILE = "patrol_config.json"
AUTH_CONFIG_FILE = "auth_config.json"

# Messaging constants
TOPIC_RACE_DATA = "racemanager/race_data"
TOPIC_INVALIDATE_CACHE = "racemanager/invalidate_cache"
SENDER_ID = uuid.uuid4().hex # This will be unique for each PID
ZENOH_CONFIG = os.environ.get("ZENOH_CONFIG")
ZENOH_ENABLED = os.environ.get("ZENOH_ENABLED", "0").lower() in ("1", "true", "yes")

if ZENOH_ENABLED:
    try:
        import zenoh
        zenoh.init_log_from_env_or("info")
        if ZENOH_CONFIG and os.path.exists(ZENOH_CONFIG):
            logger.info(f"Loading zenoh config from {ZENOH_CONFIG}")
            z_conf = zenoh.Config.from_file(ZENOH_CONFIG)
        else:
            logger.info(f"Loading default zenoh config")
            z_conf = zenoh.Config()
        zenoh_session = zenoh.open(z_conf)
    except:
        logger.exception("Failed to initialize Zenoh")
        zenoh_session = None
else:
    logger.info("Zenoh messaging has been disabled")
    zenoh_session = None


def data_filename_for_race(race_id):
    return os.path.join(CONFIG_DIR, DATA_FILE_TEMPLATE.format(race_id=str(race_id)))

def archive_filename_for_race(race_id):
    return os.path.join(CONFIG_DIR, ARCHIVE_FILE_TEMPLATE.format(race_id=str(race_id)))

def members_filename_for_race(race_id):
    return os.path.join(CONFIG_DIR, RACE_MEMBERS_FILE_TEMPLATE.format(race_id=str(race_id)))

def patrol_config_filename():
    return os.path.join(CONFIG_DIR, PATROL_CONFIG_FILE)

def auth_config_filename():
    return os.path.join(CONFIG_DIR, AUTH_CONFIG_FILE)

NUM_LANES = os.environ.get("NUM_LANES", 4)

# Initialize Firebase Admin SDK for auth (used to verify ID tokens and set custom claims).
FIREBASE_ADMIN_AVAILABLE = False
try:
    if os.environ.get('FIREBASE_CREDENTIALS'):
        # Local Dev path
        cred = fb_credentials.Certificate(os.environ.get('FIREBASE_CREDENTIALS'))
        firebase_admin.initialize_app(cred)
    else:
        # Cloud Run (ADC) path
        firebase_admin.initialize_app()
    FIREBASE_ADMIN_AVAILABLE = True
except Exception as e:
    logger.warning("Firebase Admin init failed or not configured: %s", e, exc_info=True)
    FIREBASE_ADMIN_AVAILABLE = False

# In-memory per-race cache (default TTL seconds). Use Redis later for cross-instance caching.
CACHE_TTL = int(os.environ.get('CACHE_TTL', '30'))
_race_cache = {}  # race_id -> {'data': ..., 'ts': float}
_race_cache_locks = {}
_race_cache_master_lock = threading.Lock()

default_patrol_names = {
    "1": "Foxes", 
    "2": "Hawks", 
    "3": "Mountain Lions", 
    "4": "Navgators", 
    "5": "Adventurers", 
    "Exhibition": "Exhibition"
}
fn = patrol_config_filename()
if fn and os.path.exists(fn):
    with open(fn, 'r') as f:
        default_patrol_names = json.load(f)
else:
    logger.warning("No patrol_config.json file found; using default patrol names.")

auth_config = {}
fn = auth_config_filename()
if fn and os.path.exists(fn):
    with open(fn, 'r') as f:
        auth_config = json.load(f)
else:
    logger.warning("No auth_config.json file found; using empty config.")


def _get_race_lock(rid):
    with _race_cache_master_lock:
        if rid not in _race_cache_locks:
            _race_cache_locks[rid] = threading.Lock()
        return _race_cache_locks[rid]

def get_race_cached(race_id, ttl=None):
    ttl = CACHE_TTL if ttl is None else ttl
    entry = _race_cache.get(race_id)
    if not entry:
        logger.info('Cache miss (no entry) for race %s', race_id)
        return None
    if (time.time() - entry.get('ts', 0)) > ttl:
        # expired
        try:
            with _get_race_lock(race_id):
                # double-check while holding lock
                entry2 = _race_cache.get(race_id)
                if not entry2:
                    logger.info('Cache miss (evicted concurrently) for race %s', race_id)
                    return None
                if (time.time() - entry2.get('ts', 0)) > ttl:
                    _race_cache.pop(race_id, None)
                    logger.info('Cache expired for race %s', race_id)
                    return None
                age = time.time() - entry2.get('ts', 0)
                logger.debug('Cache hit for race %s after lock (age=%.1fs)', race_id, age)
                return entry2.get('data')
        except Exception:
            logger.exception(f"Exception caught while reading cache, race_id={race_id}")
            return None
    age = time.time() - entry.get('ts', 0)
    #logger.debug(f"Loaded data: {entry.get('data')}")
    logger.debug('Cache hit for race %s (age=%.1fs)', race_id, age)
    return entry.get('data')

def set_race_cached(race_id, data: Dict[str, Any], publish=True):
    try:
        with _get_race_lock(race_id):
            #logger.debug(f"data: {data}")
            _race_cache[race_id] = {'data': data, 'ts': time.time()}
            logger.info('Set in-memory cache for race %s', race_id)
    except Exception:
        logger.exception('Failed to set in-memory cache for race %s', race_id)

    if publish:
        publish_race_data(race_id, data)

def invalidate_race_cache(race_id, publish=True):
    try:
        with _get_race_lock(race_id):
            _race_cache.pop(race_id, None)
            logger.info('Invalidated in-memory cache for race %s', race_id)
    except Exception:
        logger.exception('Failed to invalidate cache for race %s', race_id)
    if publish:
        publish_invalidated_cache(race_id)

def read_race_doc(race_id, include_archive=False):
    """Return the race-level document/data dict from Firestore or local file."""
    archived=False
    try:
        if USE_FIRESTORE and FIRESTORE_AVAILABLE:
            db = _gcf.Client()
            doc = db.collection('races').document(str(race_id)).get()
            if not doc.exists and include_archive:
                doc = db.collection('race_archives').document(str(race_id)).get()
                archived=True
            data = doc.to_dict() if doc.exists else {}
            if data:
                data['archived'] = archived
                return data
            else:
                logger.warning(f"Race data not found")
                return {}
        else:
            fn = data_filename_for_race(race_id)
            try:
                with open(fn, 'r') as f:
                    data = json.load(f)
                    data['archived'] = False
                    return data
            except FileNotFoundError:
                if include_archive:
                    archive_fn = archive_filename_for_race(race_id)
                    try:
                        with open(archive_fn, 'r') as f:
                            data = json.load(f)
                            data['archived'] = True
                            return data
                    except FileNotFoundError:
                        logger.warning(f"Race data file '{fn}' and archive file '{archive_fn}' not found")
                        return {}
                logger.warning(f"Race data file '{fn}' not found")
                return {}
    except Exception as e:
        logger.exception('Error reading race doc %s, exception: %s', race_id, str(e))
        return {}

def write_race_doc(race_id, data, merge=False):
    """Write race-level data to Firestore or local file.

    If merge is True, perform a shallow merge of keys into existing data.
    Returns True on success, False on failure.
    """

    try:
        if USE_FIRESTORE and FIRESTORE_AVAILABLE:
            try:
                db = _gcf.Client()
                if merge:
                    db.collection('races').document(str(race_id)).set(data, merge=True)
                else:
                    db.collection('races').document(str(race_id)).set(data)
                return True
            except Exception as e:
                logger.exception(f"Error writing data to Firebase for race {race_id}: {e}")
                return False
        else:
            fn = data_filename_for_race(race_id)
            existing = {}
            if merge and os.path.exists(fn):
                try:
                    with open(fn, 'r') as f:
                        existing = json.load(f)
                except Exception as e:
                    logger.exception(f"Error loading json file {fn}: {e}")
                    existing = {}
            if merge:
                for k, v in data.items():
                    existing[k] = v
                to_write = existing
            else:
                to_write = data
            try:
                with open(fn+".tmp", 'w') as f:
                    json.dump(to_write, f, default=str)
                os.replace(fn+".tmp", fn) # atomic replace
                return True
            except Exception as e:
                logger.exception(f'Error writing json file {fn}: {e}')

    except Exception:
        logger.exception(f'Error writing race doc {race_id}: {e}')
        return False

def read_members_local(race_id):
    fname = members_filename_for_race(race_id)
    try:
        with open(fname, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def write_members_local(race_id, members):
    fname = members_filename_for_race(race_id)
    with open(fname, 'w') as f:
        json.dump(members, f)

def publish_race_data(race_id, race_data: Dict[str, Any]):
    if zenoh_session:
        try:
            payload={'race_id': race_id, 'data': race_data}
            zenoh_session.put(TOPIC_RACE_DATA, json.dumps(payload), attachment=json.dumps({"SENDER_ID": SENDER_ID}))
            logger.debug(f"Sender {SENDER_ID} published race id {race_id} to Zenoh with topic {TOPIC_RACE_DATA}")
        except:
            logger.exception(f"Sender {SENDER_ID} unable to publish race id {race_id} to Zenoh with topic {TOPIC_RACE_DATA}")
    else:
        logger.debug("Zenoh session invalid, not publishing race id {race_id}")

def publish_invalidated_cache(race_id):
    if zenoh_session:
        try:
            zenoh_session.put(TOPIC_INVALIDATE_CACHE, race_id, attachment=json.dumps({"SENDER_ID": SENDER_ID}))
            logger.debug(f"Sender {SENDER_ID} published invalidate cache for race id {race_id} to Zenoh with topic {TOPIC_INVALIDATE_CACHE}")
        except:
            logger.exception(f"Sender {SENDER_ID} unable to publish invalidate cache for race id {race_id} to Zenoh with topic {TOPIC_INVALIDATE_CACHE}")
    else:
        logger.debug("Zenoh session invalid, not publishing invalidate cache message for race id {race_id}.")

def receive_race_data(sample: zenoh.Sample):
    sample_sender_id = json.loads(sample.attachment.to_string()).get("SENDER_ID")
    if SENDER_ID == sample_sender_id:
        return
    payload = json.loads(sample.payload.to_string())
    logging.info(f"Sender ID {SENDER_ID} received race data for race_id {payload['race_id']} from {sample_sender_id}")

    set_race_cached(payload['race_id'], payload['data'], publish=False)

def receive_invalidate_cache(sample: zenoh.Sample):
    sample_sender_id = json.loads(sample.attachment.to_string()).get("SENDER_ID")
    if SENDER_ID == sample_sender_id:
        return
    race_id = sample.payload.to_string()
    logging.info(f"Sender ID {SENDER_ID} received race data for race_id {'race_id'} from {sample_sender_id}")

    invalidate_race_cache(race_id, publish=False)

# This needs to be after the declaration of the functions above:
if zenoh_session:
    zenoh_session.declare_subscriber(TOPIC_RACE_DATA, receive_race_data)
    zenoh_session.declare_subscriber(TOPIC_INVALIDATE_CACHE, receive_invalidate_cache)

@app.before_request
def ensure_request_id():
    # propagate incoming request id or generate one
    rid = request.headers.get('X-Request-ID') or uuid.uuid4().hex
    try:
        g.request_id = rid
    except Exception:
        pass

@app.before_request
def log_request_info():
    if '/static/' in request.url:
        # Don't log static artifact queries
        return
    if request.is_json:
        logger.info(f'Request received for url: {request.url}, method: {request.method}', extra={'request_info':{
            'url': request.url,
            'path': request.path,
            'request_args': dict(request.args),
            'method': request.method,
            'headers': dict(request.headers),
            'uid': session.get('fb_uid') or None,
            'role': session.get('role') or None,
            'role_race_id': session.get('role_race_id') or None,
            'race_id': session.get('current_race_id') or CURRENT_RACE_ID,
            'form_data': request.form.to_dict() if request.form else None,
            'json_body': request.get_json(silent=True) # Use silent=True to avoid errors if not a valid JSON
        }})
    else:
        logger.info(f'Request received (non-JSON) for url: {request.url}, method: {request.method}', extra={'request_info': {
            'url': request.url,
            'path': request.path,
            'request_args': dict(request.args),
            'method': request.method,
            'headers': dict(request.headers),
            'uid': session.get('fb_uid') or None,
            'role': session.get('role') or None,
            'role_race_id': session.get('role_race_id') or None,
            'race_id': session.get('current_race_id') or CURRENT_RACE_ID,
            'form_data': request.form.to_dict() if request.form else None,
            'data': request.get_data(as_text=True)
        }})

@app.after_request
def log_response(response:Response):
    try:
        status = response.status_code
    except Exception:
        return response
    if status >= 500:
        logger.error('Response %s %s returned %s', request.method, request.path, status)
    elif status >= 400:
        logger.warning('Response %s %s returned %s', request.method, request.path, status)
    return response

def verify_firebase_token(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        m = re.match(r"Bearer\s+(.+)", auth_header)
        if not m:
            logger.warning('Missing Authorization header for request %s %s', request.method, request.path)
            return jsonify({'error': 'Missing Authorization header with Bearer token'}), 401
        token = m.group(1)
        if not FIREBASE_ADMIN_AVAILABLE:
            logger.error('Server not configured to verify tokens while handling %s %s', request.method, request.path)
            return respond_error('Server not configured to verify tokens', 500, 'admin_ui')
        try:
            decoded = fb_auth.verify_id_token(token)
            request.fb_user = decoded
        except Exception as e:
            logger.warning('Token verify error: %s', e, exc_info=True)
            return jsonify({'error': 'Invalid token'}), 401
        return func(*args, **kwargs)
    return wrapper

def require_race_context(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'race_context' not in kwargs:
            if 'race_id' in kwargs:
                race_id = kwargs['race_id']
            else:
                race_id = request.args.get('race') or request.form.get("race") or session.get('current_race_id') or CURRENT_RACE_ID
            context = load_data(race_id=race_id)
            kwargs['race_context'] = context
        return func(*args, **kwargs)
    return wrapper

def require_role(min_role, redirect='login'):
    roles_order = {'VIEWER': 1, 'OWNER': 2, 'ADMIN': 3, 'JUDGE': 4}
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # session owner bypass: allow only when owner is acting on their current race
            role = session.get('role')
            role_race_id = session.get('role_race_id')
            race_id = request.args.get('race') or request.form.get('race') or session.get('current_race_id') or CURRENT_RACE_ID
            uid = session.get('fb_uid')
            email = session.get('fb_email')
            logger.debug(f"require_role({min_role}) checking: role:{role} for {role_race_id}, race_id: {race_id}, current_race_id: {session.get('current_race_id')}")

            # Check for viewer:
            if min_role == 'VIEWER':
                logger.debug("All actions approved for Viewer role")
                return func(*args, **kwargs)

            # Check for Judge
            if min_role in ['JUDGE'] and role and Role[role.upper()] == Role.JUDGE:
                # ensure judge is operating on their current race only
                if str(race_id) == role_race_id \
                   and str(race_id) == str(role_race_id):
                    logger.debug(f"Judge {session.get('judge_name')} approved for this race {race_id}")
                    return func(*args, **kwargs)

            # Check for Owner
            if min_role in ['VIEWER', 'OWNER'] and role and Role[role.upper()] == Role.OWNER:
                # ensure owner is operating on their current race only
                if str(race_id) == str(session.get('current_race_id')) \
                   and str(race_id) == role_race_id:
                    logger.debug(f"Owner approved for this race {race_id}")
                    return func(*args, **kwargs)

            # Check for Admin
            if min_role in ['VIEWER', 'OWNER', 'ADMIN']:
                user = getattr(request, 'fb_user', None)
                if role and Role[role.upper()] == Role.ADMIN or \
                    session.get('isTopAdmin') or \
                    user and user.get('isTopAdmin'):
                    logger.debug(f"Admin approved for race {race_id}")
                    return func(*args, **kwargs)

            # check firebase token and custom claims

            # lookup member role
            member_role = None
            if uid:
                if USE_FIRESTORE and FIRESTORE_AVAILABLE:
                    try:
                        db = _gcf.Client()
                        mdoc = db.collection('races').document(str(race_id)).collection('members').document(uid).get()
                        if mdoc.exists:
                            member_role = mdoc.to_dict().get('role')
                    except Exception as e:
                        logger.exception('Error reading member role from firestore: %s', e)
                else:
                    local = read_members_local(race_id)
                    member_role = local.get(uid, {}).get('role')

            logger.debug(f"require_role({min_role}) checking: {email} ({uid}) member_role:{member_role}")
            if member_role and roles_order.get(member_role, 0) >= roles_order.get(min_role, 0):
                logger.debug("Member approved for this race")
                return func(*args, **kwargs)
            logger.warning('Insufficient role for request %s: required=%s member_role=%s uid=%s', request.path, min_role, member_role, uid)
            return respond_error(f'Insufficient role; minimum authentication required for action: {min_role}', 403, redirect_endpoint=redirect)
        return wrapper
    return decorator

def respond_error(message, status=400, redirect_endpoint=None, redirect_kwargs=None, flash_category='warning'):
    """Return a flash+redirect for HTML browsers or JSON+status for API clients.

    - If the request accepts HTML (and is not JSON), flash the message and redirect to
      `redirect_endpoint` (or referrer/index).
    - Otherwise return a JSON error with the given HTTP status.
    """
    try:
        wants_html = request.accept_mimetypes.accept_html and not request.is_json
    except Exception:
        wants_html = False
    if wants_html:
        try:
            flash(message, flash_category)
        except Exception:
            logger.exception('flash() failed when responding with HTML error')
        if redirect_endpoint:
            try:
                target = url_for(redirect_endpoint, **(redirect_kwargs or {}))
            except Exception:
                target = request.referrer or url_for('index')
        else:
            target = request.referrer or url_for('index')
        return redirect(target)
    return jsonify({'error': message}), status

class Role(Enum):
    VIEWER = 0
    JUDGE = 1
    OWNER = 2
    ADMIN = 3

class Rounds(Enum):
    NONE = 0
    FIRST = 1
    SEMI = 2
    FINAL = 3

class RaceScheduleType(Enum):
    PAIRED=1
    ALL_LANES=2

class Participant:
    def __init__(self, first_name, last_name, patrol):
        self.first_name = first_name
        self.last_name = last_name
        self.patrol = patrol
        self.car_weight_oz = 0
        self.times: List[float] = []
        self.average_time = 0
        self.best_time = float('inf')
        self.best_time_race_number = None
        self.car_name = None
        self.car_number = None
        self.participant_id = uuid.uuid4().hex
    def __str__(self):
        return f"P({self.car_name}): {self.patrol},{self.first_name} {self.last_name}, {self.car_weight_oz}"
    def toJSON(self):
        return self.__dict__

class Race:
    def __init__(self, patrol, race_number):
        self.patrol = patrol
        self.race_number = race_number
        self.heats: List[Heat] = []
        self.round = Rounds.NONE
    def __str__(self):
        my_str=f"R({self.patrol} {self.race_number}): Round: {self.round}" + os.linesep
        my_str+=os.linesep.join([f"Heat: {h}" for h in self.heats])
        return my_str
    def toJSON(self):
        return { "patrol": self.patrol,
                 "race_number": self.race_number,
                 "round": self.round.value,
                 "heats": [ h.toJSON() for h in self.heats]
                }

class Heat:
    def __init__(self, heat_number):
        self.heat_id = uuid.uuid4().hex
        self.heat_number = heat_number
        self.lanes: Dict[int, Participant] = {}
        self.times: Dict[int, float] = {}
    def __str__(self):
        my_str=f"H({self.heat_id} {self.heat_number}):" + os.linesep
        my_str+=os.linesep.join([f"Lane {l}: {p}" for l,p in sorted(self.lanes.items())])
        return my_str
    def toJSON(self):
        return {
                "heat_id": self.heat_id,
                "heat_number": self.heat_number,
                "lanes": {str(lane): p.participant_id if p else None for lane, p in self.lanes.items()},
                "times": {str(lane): t for lane, t in self.times.items()}
                }

class Design:
    def __init__(self, participant):
        self.participant: Participant = participant
        self.scores: Dict[str, float] = {}  # {judge_id: score}

    def toJSON(self):
        return {
            "participant_id": self.participant.participant_id,
            "scores": self.scores,
        }

class RaceContext:
    def __init__(self, race_id=DEFAULT_RACE_ID, name=DEFAULT_RACE_ID, participants=[], races=[], designs={}, judges={},
                 initial_races_completed={}, semi_final_races_completed={},
                 judging_active=True, patrol_names = default_patrol_names, archived=False):
        self.race_id: str = race_id
        self.name: str = name
        self.participants: List[Participant] = participants
        self.races: List[Race] = races
        self.designs: Dict[str, Design] = designs
        self.judges: Dict[str, Dict[str, str]] = judges
        self.initial_races_completed: Dict[str, bool] = initial_races_completed
        self.semi_final_races_completed: Dict[str, bool] = semi_final_races_completed
        self.judging_active: bool = judging_active
        self.patrol_names: Dict[str, str] = patrol_names
        self.archived: bool = archived


def _race_id_from_name(name: str) -> str:
    rid = re.sub(r"[^A-Za-z0-9_-]+", '_', name)
    rid = re.sub(r'_+', '_', rid).strip('_')
    return rid[:120] or name.replace(' ', '_')

def create_race(name):

    # If race_id not provided, derive from name
    if not name:
        raise RuntimeError('Must specify name whenc creating a race')
    # create a simple URL-friendly id: keep alphanum, dash, underscore; replace other chars/spaces with underscore
    race_id = _race_id_from_name(name)
    url = request.url_root.rstrip('/') + url_for('index', race=race_id)
    race_qr = generate_qr(url, f'race-{race_id}')

    payload = {
        'name': name,
        'race_id': race_id,
        'race_qr': race_qr,
        'participants': [],
        'races': [],
        'designs': {},
        'judges': {},
        'judging_active': True,
        'initial_races_completed': {p: False for p in default_patrol_names},
        'semi_final_races_completed': {p: False for p in default_patrol_names if p != 'Exhibition'},
        'patrol_names': default_patrol_names
    }

    if USE_FIRESTORE and FIRESTORE_AVAILABLE:
        payload['createdAt'] = _gcf.SERVER_TIMESTAMP
    
    return payload

def load_data(race_id=None, name=None):
    """Load race data for given race_id. If USE_FIRESTORE is enabled and Firestore
    client is available it will attempt to load from Firestore first; otherwise uses
    a per-race JSON file on disk (race_data_<race_id>.json)."""
    resave = False
    archived_race=False
    global CURRENT_RACE_ID

    if race_id is None:
        race_id = request.args.get('race') or request.form.get("race") or session.get('current_race_id') or CURRENT_RACE_ID
    CURRENT_RACE_ID = str(race_id)

    if name is None:
        try:
            name = session.get('current_race_name', str(race_id))
        except RuntimeError:
            name = str(race_id)

    logger.info(f"Loading Race {name} ({race_id})")

    # Try in-memory cache first
    cached = get_race_cached(CURRENT_RACE_ID)
    if cached is not None:
        logger.debug('Loaded race %s from in-memory cache', CURRENT_RACE_ID)
        data = cached
    else:
        data = read_race_doc(CURRENT_RACE_ID)

        if not data:
            archive = read_race_doc(CURRENT_RACE_ID, include_archive=True)
            if archive:
                logger.warning(f"Requested Race {race_id} has been archived.")
                data=archive
                archived_race=True
            else:
                logger.warning("Creating new race in load_data")
                data = create_race(name=name)
                resave=True
        else:
            logger.debug('Loaded race %s from storage', CURRENT_RACE_ID)

    context=_convert_race_context(CURRENT_RACE_ID, name, data)

    try:
        session['current_race_id'] = context.race_id
        session['current_race_name'] = context.name
    except Exception:
        logger.exception('Unable to set session current_race_id after firestore create')

    if resave and not archived_race:
        logger.debug(f"Saving triggered for {CURRENT_RACE_ID}")
        save_data(CURRENT_RACE_ID, context=context)

    return context

def _convert_race_context(race_id:str, name:str, data: dict) -> RaceContext:

    context = RaceContext(
        race_id=CURRENT_RACE_ID, 
        name=data.get('name', name),
        participants=[], 
        races=[], 
        designs={}, 
        judges=data.get('judges', {}),
        initial_races_completed=data.get('initial_races_completed', {}), 
        semi_final_races_completed=data.get('semi_final_races_completed', {}),
        judging_active=data.get('judging_active', True), 
        patrol_names=dict(sorted(data.get('patrol_names', default_patrol_names).items())),
        archived=data.get('archived', False))
    

    # Reconstruct participants, races, etc. from loaded data
    if not context.initial_races_completed or \
        not context.semi_final_races_completed or \
        len(context.initial_races_completed) != len(context.patrol_names) or \
        len(context.semi_final_races_completed) != len(context.patrol_names):
        resave = True
        context.initial_races_completed = \
            {p: False for p in context.patrol_names}
        context.semi_final_races_completed = \
            {p: False for p in context.patrol_names if p != 'Exhibition'}

    for p_data in sorted(data.get('participants', []), key=lambda p: p.get('car_number')):
        p = Participant(p_data.get('first_name'), p_data.get('last_name'), p_data.get('patrol'))
        p.__dict__.update(p_data)
        context.participants.append(p)

    for r_data in sorted(data.get('races', []), key=lambda r: r.get('race_number')):
        r = Race(r_data.get('patrol'), int(r_data.get('race_number', 0)))
        r.__dict__.update(r_data)
        try:
            r.round = Rounds(int(r_data.get('round', 0)))
        except Exception:
            r.round = Rounds.NONE
        r.heats = []
        for h_data in r_data.get('heats', []):
            h = Heat(int(h_data.get('heat_number', 0)))
            h.__dict__.update(h_data)
            h.lanes = {}
            for lane_num_str, p_id in h_data.get('lanes', {}).items():
                try:
                    lane_num = int(lane_num_str)
                except Exception:
                    lane_num = int(lane_num_str)
                if p_id:
                    p = next((p for p in context.participants if p.participant_id == p_id), None)
                    h.lanes[lane_num] = p
                else:
                    h.lanes[lane_num] = None
            h.times = {int(k): float(t) for k, t in h_data.get('times', {}).items()}
            r.heats.append(h)
        context.races.append(r)

    for p_id, d_data in data.get('designs', {}).items():
        participant = next((p for p in context.participants if p.participant_id == p_id), None)
        if participant:
            d = Design(participant)
            d.scores = d_data.get('scores', {})
            context.designs[p_id] = d
    for p in context.participants:
        if p.participant_id not in context.designs:
            resave = True
            context.designs[p.participant_id] = Design(p)
    
    return context

def save_data(race_id=None, context: RaceContext=None):
    """Save current in-memory state to Firestore (if enabled) or to per-race JSON file."""
    if race_id is None:
        if context:
            race_id = context.race_id
        else:
            race_id = CURRENT_RACE_ID
    

    if context:
        if context.archived:
            logger.debug(f"Save cancelled for archived race {race_id}")
            return

        payload = {
            # intentionally omitting 'name' to avoid overwriting
            'participants': [p.toJSON() for p in context.participants],
            'races': [r.toJSON() for r in context.races],
            'designs': {p_id: d.toJSON() for p_id, d in context.designs.items()},
            'judges': context.judges,
            'judging_active': context.judging_active,
            'initial_races_completed': context.initial_races_completed,
            'semi_final_races_completed': context.semi_final_races_completed,
            'patrol_names': context.patrol_names
        }
    else:
        logger.error('No context provided to save_data')
        raise RuntimeError('Must provide context to save_data')

    logger.info(f"Saving race {race_id}, {len(payload['participants'])} participants")
    try:
        write_race_doc(race_id=race_id, data=payload, merge=True)
    except Exception as e:
        logger.exception('Failed to save race data %s: %s', race_id, e)
    try:
        set_race_cached(race_id, payload)
    except Exception as e:
        logger.warning('Failed to set cache after local save for %s: %s', race_id, e)

# Ensure a race is loaded at startup and allow switching via ?race=ID
@app.before_request
def populate_fb_user():
    # If a firebase session exists, populate request.fb_user for downstream checks
    try:
        if not hasattr(request, 'fb_user') and session.get('fb_uid'):
            request.fb_user = {'uid': session.get('fb_uid'), 'isTopAdmin': session.get('isTopAdmin', False), 'email': session.get('fb_email')}
    except Exception:
        logger.exception('Failed to populate fb_user from session')

# Inject current race info into all templates
@app.context_processor
def inject_current_race():
    rid = request.args.get('race') or request.form.get("race") or session.get('current_race_id') or CURRENT_RACE_ID

    races_data = get_races_list(include_archived=True)
    race = next((r for r in races_data if r['id'] == rid), None)

    name = race.get('name', rid) if race else rid
    race_qr = race.get('race_qr', None) if race else None
    archived = race.get('archived', False) if race else False
    
    if not race:
        logger.exception('Error determining current race name for %s', rid)
    if race_qr is None:
        url = request.url_root.rstrip('/') + url_for('index', race=rid)
        race_qr = generate_qr(url, f'race-{rid}')
    return {'current_race_id': rid, 'current_race_name': name, 'current_race_qr': race_qr, 'archived': archived}

def get_races_list(include_archived=False) -> List[Dict[str, Any]]:
    races_list = []
    if USE_FIRESTORE and FIRESTORE_AVAILABLE:
        try:
            db = _gcf.Client()
            # list active races
            for doc in db.collection('races').list_documents():
                d = doc.get().to_dict() or {}
                if not d.get('archived') or include_archived:
                    races_list.append({'id': doc.id, 'name': d.get('name', ''), 'race_qr': d.get('race_qr', None), 'archived': bool(d.get('archived', False))})
            # include archives if requested
            if include_archived:
                for doc in db.collection('race_archives').list_documents():
                    d = doc.get().to_dict() or {}
                    races_list.append({'id': doc.id, 'name': d.get('name', ''), 'race_qr': d.get('race_qr', None), 'archived': True})
        except Exception as e:
            logger.exception('Error listing races from firestore: %s', e)
    else:
        # find local files race_data_<id>.json
        for f in os.listdir(CONFIG_DIR):
            if f.startswith('race_data_') and f.endswith('.json'):
                rid = f[len('race_data_'):-len('.json')]
                with open(os.path.join(CONFIG_DIR, f)) as rf:
                    try:
                        data = json.load(rf)
                        name = data.get('name', rid)
                        archived = data.get('archived', False)
                        race_qr = data.get('race_qr', None)
                    except Exception:
                        name = rid
                        archived = False
                if not archived or include_archived:
                    races_list.append({'id': rid, 'name': name, 'race_qr': race_qr, 'archived': bool(archived)})
        # include archive files if requested
        if include_archived:
            for f in os.listdir(CONFIG_DIR):
                if f.startswith('race_archive_') and f.endswith('.json'):
                    rid = f[len('race_archive_'):-len('.json')]
                    with open(os.path.join(CONFIG_DIR, f)) as rf:
                        try:
                            data = json.load(rf)
                            name = data.get('name', rid)
                            race_qr = data.get('race_qr', None)
                        except Exception:
                            name = rid
                    races_list.append({'id': rid, 'name': name, 'race_qr': race_qr, 'archived': True})
    return races_list

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# --- Admin UI and management endpoints ---
@app.route('/admin/ui')
@require_role("OWNER")
def admin_ui():
    races_list = get_races_list(include_archived=True)
    total_races=len(races_list)
    if session.get('role') == Role.OWNER.name:
        races_list = [r for r in races_list if r['id'] == session.get('role_race_id')]
    visible_races=len(races_list)

    # Determine data mode for UI hint
    if USE_FIRESTORE:
        if FIRESTORE_AVAILABLE:
            data_mode = 'Firestore'
        else:
            data_mode = 'Firestore (unavailable — using local files)'
    else:
        data_mode = 'Local files'

    return render_template('admin_races.html', races=races_list, 
                           data_mode=data_mode, firebase_admin_available=FIREBASE_ADMIN_AVAILABLE,
                           total_races=total_races, visible_races=visible_races
                           )


@app.route('/admin/set_owner_token', methods=['POST'])
@require_role('OWNER', redirect='admin_ui')
def admin_set_owner_token():
    # Generate and persist a per-race owner token and optionally a QR image
    race_id = request.args.get('race') or request.form.get("race") or session.get('current_race_id') or CURRENT_RACE_ID
    token_val = uuid.uuid4().hex
    if race_id == DEFAULT_RACE_ID:
        token_val = f'{DEFAULT_RACE_ID}_token' # sample token for the DEFAULT race
    logger.info('Admin generating owner token for race %s', race_id)
    # persist owner token
    if not write_race_doc(race_id, {'owner_token': token_val}, merge=True):
        return respond_error('Failed to set owner token', 500, 'admin_ui')

    # generate QR that links to the login page with the token shown (owner can use page to submit)
    qr_fn = None
    try:
        url = request.url_root.rstrip('/') + url_for('login', token=token_val, race=race_id)
        qr_fn = generate_qr(url, f'owner-{race_id}')
        # store QR filename alongside token
        if not write_race_doc(race_id, {'owner_qr': qr_fn}, merge=True):
            raise RuntimeError('Failed to set owner QR')
    except Exception:
        logger.exception('Failed to generate owner QR for %s', race_id)
        qr_fn = None

    try:
        flash('Owner token generated for race ' + str(race_id), 'info')
    except Exception:
        logger.exception('flash unavailable after creating owner token')
    # redirect back to the members page for the race so UI shows the new token/QR
    return redirect(url_for('admin_race_members', race_id=race_id))


@app.route('/admin/create_race', methods=['POST'])
@require_role('ADMIN', redirect='admin_ui')
def admin_create_race():
    # Accept either form or JSON body, but auto-generate a web-friendly race id
    json_body = request.get_json(silent=True) or {}
    name = (request.form.get('name') or json_body.get('name') or '').strip()
    # If race_id not provided, derive from name
    if not name:
        logger.warning('admin_create_race missing name/raceId; request path=%s', request.path)
        return respond_error('Name is required to create a race', 400, 'admin_ui')

    payload = create_race(name)
    race_id = payload['race_id']

    # accept optional owner_token provided at creation time
    owner_token = (request.form.get('owner_token') or json_body.get('owner_token'))
    if owner_token:
        payload['owner_token'] = owner_token
        # for convenience, allow a fixed DEFAULT race token
        if race_id == DEFAULT_RACE_ID and owner_token == f'{DEFAULT_RACE_ID}_token':
            payload['owner_token'] = f'{DEFAULT_RACE_ID}_token'

    actor = getattr(request, 'fb_user', None) or {'uid': session.get('fb_uid'), 'email': session.get('fb_email'), 'role': session.get('role')}
    logger.info('admin_create_race requested by %s for race_id=%s name=%s', actor.get('uid') or actor.get('role'), race_id, name)
 
    if not write_race_doc(race_id=race_id, data=payload, merge=True):
        return respond_error("Error creating race", 500, 'admin_ui', flash_category='error')
    try:
        set_race_cached(race_id, payload)
    except Exception:
        logger.exception('Failed to set cache after create for %s', race_id)
        logger.exception('create race error: %s', e)
        return respond_error("Error creating race", 500, 'admin_ui', flash_category='error')
    return redirect(url_for('admin_ui'))

@app.route('/admin/races/<race_id>/members', methods=['GET', 'POST'])
@require_role('OWNER', redirect='admin_ui')
def admin_race_members(race_id):
    user = getattr(request, 'fb_user', None)
    logger.info('admin_race_members access for race %s by %s', race_id, (user.get('uid') if user else session.get('role')))

    if request.method == 'POST':
        # normalize JSON body if provided (silent to avoid 415 on form posts)
        json_body = request.get_json(silent=True) or {}
        # --- Handle judge management actions (regen/remove) from admin UI ---
        regen_judge = request.form.get('regen_judge') or json_body.get('regen_judge') or \
            request.form.get('judge_name') or json_body.get('judge_name')
        remove_judge = request.form.get('remove_judge') or json_body.get('remove_judge')
        if regen_judge:
            try:
                data = read_race_doc(race_id)
                judges_data = data.get('judges', {})

                # generate new token and QR for the judge
                token_val = uuid.uuid4().hex
                judges_data.setdefault(regen_judge, {})['token'] = token_val
                judges_data.setdefault(regen_judge, {})['id'] = judges_data.get(regen_judge, {}).get('id') or uuid.uuid4().hex
                url = request.url_root.rstrip('/') + url_for('login', token=token_val, race=race_id)
                try:
                    qr=judges_data[regen_judge].get('qr', None)
                    if qr:
                        try:
                            os.remove(os.path.join(app.static_folder, 'qr', qr))
                        except Exception:
                            logger.exception('Failed to remove QR file %s for removed judge %s', qr, regen_judge)
                    qr_fn = generate_qr(url, judges_data[regen_judge]['id'])
                    judges_data[regen_judge]['qr'] = qr_fn
                except Exception:
                    logger.exception('Failed to generate QR while regen judge %s', regen_judge)

                # persist back
                if not write_race_doc(race_id, {'judges': judges_data}, merge=True):
                    raise RuntimeError('Failed to persist judges data')
                flash(f'Regenerated token for judge {regen_judge}', 'info')
                return redirect(url_for('admin_race_members', race_id=race_id))
            except Exception:
                logger.exception('Error regenerating judge token for %s', regen_judge)
                return respond_error('Failed to regenerate judge token', 500, 'admin_race_members', {'race_id': race_id})

        if remove_judge:
            try:
                data = read_race_doc(race_id)
                judges_data = data.get('judges', {})
                if remove_judge in judges_data:
                    qr=judges_data[remove_judge].get('qr', None)
                    if qr:
                        try:
                            os.remove(os.path.join(app.static_folder, 'qr', qr))
                        except Exception:
                            logger.exception('Failed to remove QR file %s for removed judge %s', qr, remove_judge)
                    judges_data.pop(remove_judge, None)
                data.pop('judges', None)
                data.update({'judges': judges_data})
                if not write_race_doc(race_id, data):
                    raise RuntimeError('Failed to persist judges data')
                flash(f'Removed judge {remove_judge}', 'info')
                return redirect(url_for('admin_race_members', race_id=race_id))
            except Exception:
                logger.exception('Error removing judge %s', remove_judge)
                return respond_error('Failed to remove judge', 500, 'admin_race_members', {'race_id': race_id})

        # add or update member
        uid = request.form.get('uid') or json_body.get('uid')
        email = request.form.get('email') or json_body.get('email')
        role = request.form.get('role') or json_body.get('role')

        # Resolve email to UID if needed
        if not uid and email and FIREBASE_ADMIN_AVAILABLE:
            try:
                u = fb_auth.get_user_by_email(email)
                uid = u.uid
            except Exception:
                flash(f'User with email {email} not found', 'warning')
                return redirect(url_for('admin_race_members', race_id=race_id))

        # soft-delete or restore actions
        delete_uid = request.form.get('delete_uid') or json_body.get('delete_uid')
        restore_uid = request.form.get('restore_uid') or json_body.get('restore_uid')

        # Handle restore
        if restore_uid:
            logger.info('Restore member %s requested for race %s by %s', restore_uid, race_id, user.get('uid'))
            if USE_FIRESTORE and FIRESTORE_AVAILABLE:
                try:
                    db = _gcf.Client()
                    db.collection('races').document(race_id).collection('members').document(restore_uid).update({'deleted': False, 'deletedAt': None, 'updatedAt': _gcf.SERVER_TIMESTAMP})
                    return redirect(url_for('admin_race_members', race_id=race_id))
                except Exception as e:
                    logger.exception('restore member error: %s', e)
                    return 'error', 500
            else:
                m = read_members_local(race_id)
                if restore_uid in m:
                    m[restore_uid]['deleted'] = False
                    m[restore_uid].pop('deletedAt', None)
                    write_members_local(race_id, m)
                return redirect(url_for('admin_race_members', race_id=race_id))

        # Handle soft-delete
        if delete_uid:
            logger.info('Soft-delete member %s requested for race %s by %s', delete_uid, race_id, user.get('uid'))
            if USE_FIRESTORE and FIRESTORE_AVAILABLE:
                try:
                    db = _gcf.Client()
                    db.collection('races').document(race_id).collection('members').document(delete_uid).update({'deleted': True, 'deletedAt': _gcf.SERVER_TIMESTAMP})
                    return redirect(url_for('admin_race_members', race_id=race_id))
                except Exception as e:
                    logger.exception('soft-delete member error: %s', e)
                    return 'error', 500
            else:
                m = read_members_local(race_id)
                if delete_uid in m:
                    m[delete_uid]['deleted'] = True
                    m[delete_uid]['deletedAt'] = str(__import__('datetime').datetime.utcnow().isoformat())
                    write_members_local(race_id, m)
                return redirect(url_for('admin_race_members', race_id=race_id))

        # Add/edit member
        if not uid or not role:
            logger.warning('admin_race_members missing uid or role; request by %s race=%s', user.get('uid') if user else session.get('fb_uid'), race_id)
            return respond_error('UID and role required', 400, 'admin_race_members', {'race_id': race_id})
        
        logger.info('Assigning role %s to uid %s for race %s by %s', role, uid, race_id, user.get('uid'))
        
        update_data = {'role': role}
        if email:
            update_data['email'] = email

        if USE_FIRESTORE and FIRESTORE_AVAILABLE:
            try:
                db = _gcf.Client()
                update_data['updatedAt'] = _gcf.SERVER_TIMESTAMP
                db.collection('races').document(race_id).collection('members').document(uid).set(update_data, merge=True)
                return redirect(url_for('admin_race_members', race_id=race_id))
            except Exception as e:
                logger.exception('assign member error: %s', e)
                return 'error', 500
        else:
            m = read_members_local(race_id)
            existing = m.get(uid, {})
            existing.update(update_data)
            existing.pop('deleted', None)
            existing.pop('deletedAt', None)
            m[uid] = existing
            write_members_local(race_id, m)
            return redirect(url_for('admin_race_members', race_id=race_id))

    # GET: list members
    members = {}
    if USE_FIRESTORE and FIRESTORE_AVAILABLE:
        try:
            db = _gcf.Client()
            cols = db.collection('races').document(race_id).collection('members').stream()
            for doc in cols:
                members[doc.id] = doc.to_dict()
        except Exception as e:
            logger.exception('error reading members: %s', e)
    else:
        members = read_members_local(race_id)

    # Also load race-level info (judges, owner_token) for the admin UI
    archived=False
    rd = read_race_doc(race_id)
    if not rd:
        archived_rd=read_race_doc(race_id, include_archive=True)
        if archived_rd:
            rd = archived_rd
            archived=True
    context = load_data(race_id)

    judges_data = context.judges
    for j_name, jinfo in judges_data.items():
        jinfo['voted'] = any([des.scores[jinfo['id']] for des in context.designs.values() if jinfo['id'] in des.scores.keys()])
    sorted_judges = dict(sorted(judges_data.items()))
    judges_data = dict(sorted_judges)
    owner_token = rd.get('owner_token')
    owner_qr = rd.get('owner_qr')    # filename saved by admin_set_owner_token

    return render_template('admin_members.html', race_id=race_id, members=members, judges=judges_data, owner_token=owner_token, owner_qr=owner_qr, archived=archived)


def _delete_qr_files_from_doc(data_doc):
    try:
        qr_dir = os.path.join(app.static_folder, 'qr')
        files = []
        if isinstance(data_doc.get('race_qr'), str):
            files.append(data_doc.get('race_qr'))
        if isinstance(data_doc.get('owner_qr'), str):
            files.append(data_doc.get('owner_qr'))
        for jname, jinfo in (data_doc.get('judges') or {}).items():
            jq = jinfo.get('qr')
            if jq:
                files.append(jq)
        for fn in files:
            try:
                fp = os.path.join(qr_dir, fn)
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                logger.exception('Failed to remove QR file %s', fn)
    except Exception:
        logger.exception('Error while deleting QR files')


@app.route('/admin/archive_race', methods=['POST'])
@require_role('ADMIN', redirect='admin_ui')
def admin_archive_race():
    race_id = request.form.get('race') or request.json.get('race') 
    logger.info('Archive requested for race %s by %s', race_id, getattr(request, 'fb_user', None) or session.get('role'))
    try:
        if USE_FIRESTORE and FIRESTORE_AVAILABLE:
            db = _gcf.Client()
            doc_ref = db.collection('races').document(str(race_id))
            doc = doc_ref.get()
            if not doc.exists:
                return respond_error('Race not found', 404, 'admin_ui')
            data = doc.to_dict() or {}
            data['archived'] = True
            data['archivedAt'] = _gcf.SERVER_TIMESTAMP
            # write to archive collection
            db.collection('race_archives').document(str(race_id)).set(data)
            # delete original
            doc_ref.delete()
            # optionally remove members subcollection
            try:
                for member in db.collection('races').document(str(race_id)).collection('members').list_documents():
                    member.delete()
            except Exception:
                logger.exception('Failed to remove members subcollection during archive for %s', race_id)
        else:
            fn = data_filename_for_race(race_id)
            if not os.path.exists(fn):
                return respond_error('Race not found', 404, 'admin_ui')
            with open(fn, 'r') as f:
                data = json.load(f)
            data['archived'] = True
            data['archivedAt'] = str(__import__('datetime').datetime.utcnow().isoformat())
            archive_fn = os.path.join(CONFIG_DIR, f"race_archive_{race_id}.json")
            with open(archive_fn + '.tmp', 'w') as f:
                json.dump(data, f, default=str)
            os.replace(archive_fn + '.tmp', archive_fn)
            try:
                os.remove(fn)
            except Exception:
                logger.exception('Failed to remove original race file %s after archiving', fn)

        # delete QR images since they can be regenerated
        try:
            _delete_qr_files_from_doc(data)
        except Exception:
            logger.exception('QR deletion step failed for %s', race_id)

        invalidate_race_cache(race_id)
        flash(f'Race {race_id} archived', 'info')
        return redirect(url_for('admin_ui'))
    except Exception as e:
        logger.exception('Error archiving race %s: %s', race_id, e)
        return respond_error('Failed to archive race', 500, 'admin_ui')


@app.route('/admin/restore_race', methods=['POST'])
@require_role('ADMIN', redirect='admin_ui')
def admin_restore_race():
    race_id = request.form.get('race') or request.json.get('race')
    logger.info('Request to restore race %s by %s', race_id, getattr(request, 'fb_user', None) or session.get('role'))
    if not race_id:
        return respond_error('race required', 400, 'admin_ui')
    try:
        #TODO: Cleanup restore... use common functions: read(archive), update, write, remove archive.
        if USE_FIRESTORE and FIRESTORE_AVAILABLE:
            db = _gcf.Client()
            arc = db.collection('race_archives').document(str(race_id)).get()
            if not arc.exists:
                return respond_error('Archive not found', 404, 'admin_ui')
            data = arc.to_dict() or {}
            data.pop('archived', None)
            data.pop('archivedAt', None)
            db.collection('races').document(str(race_id)).set(data)
            db.collection('race_archives').document(str(race_id)).delete()
        else:
            archive_fn = archive_filename_for_race(race_id)
            if not os.path.exists(archive_fn):
                return respond_error('Archive not found', 404, 'admin_ui')
            with open(archive_fn, 'r') as f:
                data = json.load(f)
            data.pop('archived', None)
            data.pop('archivedAt', None)
            fn = data_filename_for_race(race_id)
            with open(fn + '.tmp', 'w') as f:
                json.dump(data, f, default=str)
            os.replace(fn + '.tmp', fn)
            try:
                os.remove(archive_fn)
            except Exception:
                logger.exception('Failed to remove archive file %s after restore', archive_fn)

        invalidate_race_cache(race_id)
        flash(f'Race {race_id} restored', 'info')
        return redirect(url_for('admin_ui'))
    except Exception as e:
        logger.exception('Error restoring race %s: %s', race_id, e)
        return respond_error('Failed to restore race', 500, 'admin_ui')

@app.route('/admin/delete_race', methods=['POST'])
@require_role('ADMIN', redirect='admin_ui')
def admin_delete_race():
    json_body = request.get_json(silent=True) or {}
    race_id = request.form.get('race') or json_body.get('race')
    
    if not race_id:
        return respond_error('race required', 400, 'admin_ui')
    
    logger.info('Permanent delete requested for archived race %s by %s', race_id, getattr(request, 'fb_user', None) or session.get('role'))

    try:
        if USE_FIRESTORE and FIRESTORE_AVAILABLE:
            db = _gcf.Client()
            arc_ref = db.collection('race_archives').document(str(race_id))
            if not arc_ref.get().exists:
                 return respond_error('Archived race not found', 404, 'admin_ui')
            arc_ref.delete()
        else:
            archive_fn = os.path.join(CONFIG_DIR, f"race_archive_{race_id}.json")
            if not os.path.exists(archive_fn):
                return respond_error('Archived race not found', 404, 'admin_ui')
            os.remove(archive_fn)

        flash(f'Race {race_id} permanently deleted', 'info')
        return redirect(url_for('admin_ui'))
    except Exception as e:
        logger.exception('Error deleting race %s: %s', race_id, e)
        return respond_error('Failed to delete race', 500, 'admin_ui')


@app.route('/admin/download_race', methods=['GET', 'POST'])
@require_role('OWNER', redirect='admin_ui')
def admin_download_race():

    race_id = request.args.get('race') or request.form.get('race')
    if race_id is None:
        return respond_error('Must specify race_id to download', 400, 'admin_ui')

    # Try in-memory cache first
    cached = get_race_cached(race_id)
    if cached is not None:
        logger.debug('Loaded race %s from in-memory cache', race_id)
        data = cached
    else:
        data = read_race_doc(race_id, include_archive=True)
        logger.debug('Loaded race %s from storage', race_id)

    if not data:
        return respond_error('Race not found', 404, 'admin_ui')

    return Response(json.dumps(data), mimetype='application/json', headers={ 'Content-Disposition': f'attachment; filename=race_data_{race_id}.json' })

@app.route('/admin/upload_restore', methods=['POST'])
@require_role('ADMIN', redirect='admin_ui')
def admin_upload_restore():
    # Accept a multipart file and race id to restore
    race_name = request.form.get('race_name') or (request.args.get('race_name'))
    file = request.files.get('file')
    if not file:
        return respond_error('file required', 400, 'admin_ui')
    
    try:
        data = json.load(file)
        if not race_name:
            race_name = data.get('name')
        if not race_name:
            return respond_error('Race name required (in form or file)', 400, 'admin_ui')
        
        race_payload = create_race(race_name)
        race_payload.update({k:v for k,v in data.items() if k not in [ 'name', 'race_id', 'race_qr ']})
        data = race_payload
        race_id = data.get('race_id')

        # write into active race
        if not write_race_doc(race_id, data, merge=False):
            return respond_error('Failed to save imported race file', 500, 'admin_ui')
        invalidate_race_cache(race_id)
        flash(f'Race {race_id} restored from upload', 'info')
        return redirect(url_for('admin_ui'))
    except Exception as e:
        logger.exception('Failed to restore uploaded race %s: %s', race_id, e)
        return respond_error('Failed to import race file', 500, 'admin_ui')


@app.route('/admin/set_top_admin', methods=['POST'])
@require_role('ADMIN', redirect='admin_ui')
def admin_set_top_admin():
    json_body = request.get_json(silent=True) or {}
    uid = request.form.get('uid') or json_body.get('uid')
    make_top = request.form.get('isTop') or json_body.get('isTop')
    if make_top in (None, ''):
        logger.warning('admin_set_top_admin missing isTop parameter; request by %s', getattr(request, 'fb_user', {}).get('uid') or session.get('role'))
        return 'isTop required', 400
    is_top_bool = str(make_top).lower() in ('1', 'true', 'yes')
    if not FIREBASE_ADMIN_AVAILABLE:
        logger.error('admin_set_top_admin called but Firebase Admin SDK not available')
        return respond_error('Server not configured for Firebase admin actions', 500, 'admin_ui')
    try:
        logger.info('Setting top-admin=%s for uid=%s by %s', is_top_bool, uid, getattr(request, 'fb_user', {}).get('uid') or session.get('role'))
        fb_auth.set_custom_user_claims(uid, {'isTopAdmin': is_top_bool})
        fb_auth.revoke_refresh_tokens(uid)
        return redirect(url_for('admin_ui'))
    except Exception as e:
        logger.exception('set_top_admin error: %s', e)
        return 'error', 500


@app.route('/session_login', methods=['POST'])
def session_login():
    """Accepts JSON { idToken: '<firebase id token>' } from client, verifies it,
    and establishes a Flask session (server-side) with uid and isTopAdmin claim."""
    data = request.get_json(silent=True) or {}
    id_token = data.get('idToken')
    race_id = request.args.get('race') or request.form.get("race") or session.get('current_race_id') or CURRENT_RACE_ID
 
    if not FIREBASE_ADMIN_AVAILABLE:
        logger.error('session_login attempted but Firebase Admin SDK not available')
        return jsonify({ 'status': 'error', 'message': 'Server not configured for token verification'}), 500

    try:
        # Verify the token via the Service Account
        decoded_token = fb_auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        email = decoded_token.get('email')

        # set session values
        session['isTopAdmin'] = decoded_token.get('isTopAdmin', False)
        if session['isTopAdmin']:
            session['role'] = Role.ADMIN.name
            session['role_race_id'] = "global"
        # optional: set a friendly username/email
        session['fb_email'] = email

        logger.info('session_login successful for uid=%s isTopAdmin=%s email=%s', uid, session.get('isTopAdmin'), email)

        # If using Firestore/local member files, preload membership roles for convenience
        members = {}
        if uid:
            if USE_FIRESTORE and FIRESTORE_AVAILABLE:
                try:
                    db = _gcf.Client()
                    mdoc = db.collection('races').document(str(race_id)).collection('members').document(uid).get()
                    if mdoc.exists:
                        members[uid] = mdoc.to_dict()
                except Exception:
                    logger.exception('Failed to preload member data for uid %s', uid)
            else:
                local = read_members_local(race_id)
                if uid in local:
                    members[uid] = local[uid]
                else:
                    members[uid] = {"role": "VIEWER"}  # default to viewer if not found
                    local[uid] = {"role": "VIEWER"}  # default to viewer if not found
                    write_members_local(race_id, local)
        session['members'] = members

        if members.get(uid, {}).get('role', None):
            session['role'] = members[uid]['role']
            session['role_race_id'] = race_id
            if members[uid].get('role') == 'JUDGE':
                session['judge_name'] = email
                data=read_race_doc(race_id)
                judges_data = data.get('judges', {})

                # generate new token and QR for the judge
                if email not in judges_data:
                    token_val = uuid.uuid4().hex
                    judges_data.setdefault(email, {})['token'] = token_val
                    judges_data.setdefault(email, {})['id'] = uid
                    url = request.url_root.rstrip('/') + url_for('login', token=token_val, race=race_id)
                    try:
                        qr=judges_data[email].get('qr', None)
                        if qr:
                            try:
                                os.remove(os.path.join(app.static_folder, 'qr', qr))
                            except Exception:
                                logger.exception('Failed to remove QR file %s for removed judge %s', qr, email)
                        qr_fn = generate_qr(url, judges_data[email]['id'])
                        judges_data[email]['qr'] = qr_fn
                    except Exception:
                        logger.exception('Failed to generate QR while regen judge %s', email)
                    if not write_race_doc(race_id, {'judges': judges_data}, merge=True):
                        raise RuntimeError('Failed to persist judges data')

        else:
            session['role'] = Role.VIEWER.name
        
        if session['role'] == Role.VIEWER.name:
            next_URL=url_for('index', race=race_id)
        elif session['role'] == Role.JUDGE.name:
            next_URL=url_for('judge_design', race=race_id)
        elif session['role'] == Role.OWNER.name or session['role'] == Role.ADMIN.name:
            next_URL=url_for('admin_ui', race=race_id)
        else:
            next_URL=url_for('login')

        return jsonify({'ok': True, 
                        'nextURL': next_URL,
                        'isAdminOrOwner': session.get('role', Role.VIEWER.name) in [Role.ADMIN.name, Role.OWNER.name],
                        'message': f'Successfully logged in as role {session.get("role")} for race {race_id}. ' + \
                            f' If an elevated account role is required, contact your race administrator.',
                        'role': session.get('role')}), 200
    except Exception as e:
        logger.warning('session_login verify failed: %s', e, exc_info=True)
        return jsonify({ 'status': 'error', 'message': 'Unable to authenticate user'}), 401


@app.route('/logout')
def logout():
    # Clear both legacy session role and firebase session keys
    logger.info('Logout requested; clearing session for role=%s fb_uid=%s', session.get('role'), session.get('fb_uid'))
    #session.pop('role', None)
    #session.pop('judge_name', None)
    #session.pop('fb_uid', None)
    #session.pop('isTopAdmin', None)
    #session.pop('fb_email', None)
    #session.pop('members', None)
    session.clear()
    return redirect(url_for('index'))



def generate_qr(url,id):
    #TODO: make QR codes in-memory only; generate on-demand and don't save on disk.

    qr = qrcode.QRCode(box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    image = qr.make_image()

    filename = f"{id}.png"
    # Save into Flask static folder so url_for('static', ...) works regardless of cwd
    qr_dir = os.path.join(app.static_folder, 'qr')
    try:
        os.makedirs(qr_dir, exist_ok=True)
    except Exception:
        logger.exception('Failed to create qr directory %s', qr_dir)
    file_path = os.path.join(qr_dir, filename)
    try:
        image.save(file_path)
    except Exception:
        logger.exception('Failed to save QR image to %s', file_path)
        raise
    return filename



@app.route("/login", methods=["GET", "POST"])
def login():
    # Accept token from either GET query or POST form (so QR scans to /login?token=... work)
    token = request.values.get('token')
    race_id = request.values.get('race')
    # If a token is present (from GET or POST), attempt authentication and redirect
    if token:
        try:
            rd = read_race_doc(race_id)
        except Exception:
            logger.exception(f"Failed to load race {race_id}")
            rd = {}
        race_name = rd.get('name', race_id)

        # owner check (per-race then global)
        owner_token = rd.get('owner_token')
        if owner_token and token == owner_token:
            session['role'] = Role.OWNER.name
            session['role_race_id'] = race_id
            session['current_race_id'] = race_id
            session['current_race_name'] = race_name
            try:
                flash(f'Logged in as Owner for {race_name}', 'success')
            except Exception:
                pass
            return redirect(url_for('index', race=race_id))

        if token == auth_config.get('owner_token'):
            session['role'] = Role.OWNER.name
            session['role_race_id'] = race_id
            session['current_race_id'] = race_id
            session['current_race_name'] = race_name
            flash(f'Logged in as Owner for {race_name}', 'success')
            return redirect(url_for('index', race=race_id))

        # judge check (per-race then global)
        judges_data = rd.get('judges', {})
        for jname, jinfo in judges_data.items():
            if jinfo.get('token') == token:
                session['role'] = Role.JUDGE.name
                session['role_race_id'] = race_id
                session['judge_name'] = jname
                session['current_race_id'] = race_id
                session['current_race_name'] = race_name
                flash(f"{jname} logged in as Judge for race {race_name}", 'success')
                return redirect(url_for('judge_design', race=race_id))

        if token == auth_config.get('judge_token'):
            session['role'] = Role.JUDGE.name
            session['role_race_id'] = race_id
            session['current_race_id'] = race_id
            session['current_race_name'] = race_name
            flash(f'Logged in as Judge for {race_name}', 'success')
            return redirect(url_for('judge_design', race=race_id))

        # global admin check
        if token == auth_config.get('admin_token'):
            session['role'] = Role.ADMIN.name
            session['role_race_id'] = "global"
            session['isTopAdmin'] = True
            session['current_race_id'] = race_id
            session['current_race_name'] = race_name
            flash(f'Logged in as Admin', 'success')
            return redirect(url_for('admin_ui', race=race_id))


        flash(f'Invalid token for {race_name}', 'warning')
        return render_template('login.html', races_list=get_races_list())

    # No token provided: render login page
    firebase_cfg = {}
    if USE_FIRESTORE and FIRESTORE_AVAILABLE:
        if os.environ.get('FIREBASE_API_CREDENTIALS', None):
            try:
                with open(os.environ.get('FIREBASE_API_CREDENTIALS')) as f:
                    temp_cfg = json.load(f)
                    if 'apiKey' in temp_cfg:
                        firebase_cfg = temp_cfg
            except:
                pass
        if not firebase_cfg:
            firebase_cfg = auth_config.get('firebase') if isinstance(auth_config, dict) else {}
        # If no firebase config in file, allow env-based config for Cloud Run
        if not firebase_cfg:
            # Automatically get the Project ID from the Admin SDK
            project_id = firebase_admin.get_app().project_id
            FB_API_KEY = os.environ.get('FB_API_KEY', DEFAULT_API_KEY)
            FB_APP_ID = os.environ.get('FB_APP_ID', DEFAULT_APP_ID)

            firebase_cfg = {
                "apiKey": FB_API_KEY,
                "authDomain": f"{project_id}.firebaseapp.com",
                "projectId": project_id,
                "storageBucket": f"{project_id}.appspot.com",
                "messagingSenderId": FB_APP_ID.split(':')[1], # Messaging ID is usually the middle part of App ID
                "appId": FB_APP_ID
            }
    return render_template("login.html",
                           races_list=get_races_list(),
                           firebase_config=firebase_cfg)


@app.route("/race_links")
def race_links():
    races = get_races_list()
    race_id = request.args.get('race') or request.form.get("race") or session.get('current_race_id') or CURRENT_RACE_ID
    race = next((r for r in races if r['id'] == race_id), None)
    if race:
        # Always regenerate, because if redirection is happening for different hosts
        # then I want the QR to be right. 
        qr = race.get('race_qr')
        url = request.url_root.rstrip('/') + url_for('index', race=race_id)
        generate_qr(url, f'race-{race_id}')
        
    return render_template("race_links.html", races_list=races)


@app.route("/", methods=["GET", "POST"])
@require_race_context
def index(race_context: RaceContext):

    message = request.args.get('message', None)

    sorted_participants = sorted(race_context.participants, key=lambda p: (p.patrol, p.car_number))

    if message:
        flash(message, 'success')

    return render_template("index.html", 
                           participants=sorted_participants,
                           archived=race_context.archived,
                           RaceScheduleType=RaceScheduleType, 
                           patrol_names=race_context.patrol_names)


@app.route("/add_participant", methods=["POST"])
@require_role('OWNER')
@require_race_context
def add_participant(race_context: RaceContext):
    if request.method == "POST":
        first_name = request.form.get("first_name") # Get first name
        last_name = request.form.get("last_name")  # Get last name
        try:
            patrol = request.form.get("patrol") # Get patrol
            if patrol in race_context.patrol_names:
                new_participant = _add_participant_to_race(first_name, last_name, patrol, race_context=race_context)  # Add participant
                save_data(context=race_context) # Save data after adding participant
                return redirect(url_for("edit_participant", participant_id=new_participant.participant_id))  # Redirect after successful addition
            else:
                flash(f"Invalid patrol: {patrol}", "error")
        except ValueError as e:
            logger.warning("Failed to add participant {first_name} {last_name} to patrol={patrol}: {e}")
            flash(f"Failed to add participant {first_name} {last_name} to patrol={patrol}: {e}", "error")

    return redirect(url_for("index"))


@app.route("/edit_participant/<participant_id>", methods=["GET", "POST"])
@require_role('OWNER')
@require_race_context
def edit_participant(participant_id, race_context: RaceContext):
    participant = next((p for p in race_context.participants if p.participant_id == participant_id), None) 
    if not participant:
        logger.warning('edit_participant: participant not found id=%s', participant_id)
        return respond_error("Participant not found", 404, "index")

    if request.method == "POST":
        try:
            car_weight_oz = float(request.form.get("car_weight_oz"))
            car_name = request.form.get("car_name")
            first_name = request.form.get("first_name")
            last_name = request.form.get("last_name")

            participant.car_weight_oz = car_weight_oz
            participant.car_name = car_name
            participant.first_name = first_name
            participant.last_name = last_name

            save_data(context=race_context) # Save data after editing participant
            logger.info('Edited participant id=%s weight=%s name=%s by uid=%s', participant_id, car_weight_oz, car_name, session.get('fb_uid') or session.get('role'))
            return redirect(url_for("index"))  # Redirect back to the participant list
        except ValueError:
            logger.warning('edit_participant invalid weight input for participant_id=%s value=%s', participant_id, request.form.get('car_weight_oz'))
            return respond_error ("Invalid weight input. Please enter a number.", 400, "edit_participant", {"participant_id": participant_id})

    return render_template("edit_participant.html", participant=participant, patrol_names = race_context.patrol_names)

@app.route("/delete_participant/<participant_id>", methods=["POST"])
@require_role('OWNER')
@require_race_context
def delete_participant(participant_id, race_context: RaceContext):
    participant = next((p for p in race_context.participants if p.participant_id == participant_id), None)
    if not participant:
        logger.warning('delete_participant: participant not found id=%s', participant_id)
        return respond_error("Participant not found", 404, "index")

    if participant_id in race_context.designs:
        del race_context.designs[participant_id]
    race_context.participants.remove(participant)  # Remove the participant

    save_data(context=race_context) # Save data after removing participant
    logger.info('Deleted participant id=%s name=%s by uid=%s', participant_id, participant.first_name + ' ' + participant.last_name, session.get('fb_uid') or session.get('role'))
    return redirect(url_for("index"))  # Redirect back to the participant list

@app.route("/participant_times/<participant_id>")
@require_race_context
def participant_times(participant_id, race_context: RaceContext):
 
    participant = next((p for p in race_context.participants if p.participant_id == participant_id), None)

    header, race_data = get_participant_race_data(participant_id, race_context=race_context)

    if not participant:
        logger.warning('participant_times: participant not found id=%s', participant_id)
        return respond_error("Participant not found", 404, "index")
    return render_template("participant_times.html", 
                           participant=participant, header=header, race_data=race_data, 
                           patrol_names=race_context.patrol_names)

@require_race_context
def _add_participant_to_race(first_name,last_name, patrol, race_context: RaceContext):

    if patrol in race_context.patrol_names.values():
        for k,v in race_context.patrol_names.items():
            if v == patrol:
                patrol = k
                break

    patrol = str(patrol)

    if patrol in race_context.patrol_names:

        new_p = Participant(first_name,last_name, patrol)

        existing_car_numbers = [p.car_number for p in race_context.participants if p.patrol == patrol]

        if existing_car_numbers:
            next_car_number = max(existing_car_numbers) + 1  # Find the maximum and add 1
        else:
            next_car_number = 1  # Start at 1 if no existing numbers

        new_p.car_number = next_car_number
        new_p.car_name = f"{race_context.patrol_names.get(patrol)[:1]}{next_car_number:02}"

        race_context.participants.append(new_p)

        race_context.designs[new_p.participant_id] = Design(new_p)

        logger.debug('Added participant id=%s name=%s %s patrol=%s car_number=%s', new_p.participant_id, new_p.first_name, new_p.last_name, new_p.patrol, new_p.car_number)
        return new_p
    else:
        logger.warning(f"Attempted to add participant {id}/{first_name} {last_name} with invalid patrol: {patrol}")
        raise ValueError(f"Invalid Patrol: {patrol}, valid options are: {', '.join(list(race_context.patrol_names.values()))}")

@require_race_context
def clear_races(race_context: RaceContext):

    race_context.races.clear()
    # If races have been rescheduled, then all data needs to be reset
    for p in race_context.participants:
        p.times = []
        p.average_time = 0
        p.best_time = float('inf')
        p.best_time_race_number = None

@require_race_context
def schedule_initial_races(schedule_type: RaceScheduleType, race_context: RaceContext):

    race_number = 1
    for patrol in dict(sorted(race_context.patrol_names.items())):
        patrol_racers = [p for p in race_context.participants if p.patrol == patrol]

        if patrol_racers:
            race_groups = group_racers(patrol_racers)
            if schedule_type == RaceScheduleType.PAIRED:
                assign_paired_lanes(race_groups, Rounds.FIRST, race_number, race_context=race_context)
            else:
                assign_all_lanes(race_groups, Rounds.FIRST, race_number, race_context=race_context)
            race_number += len(race_groups)

            race_context.initial_races_completed[patrol] = False # Initialize to False at the start
            race_context.semi_final_races_completed[patrol] = False 
        else:
            race_context.initial_races_completed[patrol] = True # No races for this patrol
            race_context.semi_final_races_completed[patrol] = True # No races for this patrol


@require_race_context
def schedule_semi_final_races(patrol, schedule_type: RaceScheduleType,race_context: RaceContext):

    if patrol == "Exhibition":
        return  # Don't schedule semi-finals for the exhibition patrol
    if not race_context.initial_races_completed[patrol]:
        return  # Don't schedule semi's if we havent finished the races

    top_racers,_ = get_top_racers(Rounds.FIRST, patrol, NUM_LANES, race_context=race_context)
    name_sorted_top_racers = sorted(top_racers, key=lambda racer: racer.car_name)

    if top_racers:
        race_groups = [name_sorted_top_racers]  # Create a single group of top racers
        if schedule_type == RaceScheduleType.PAIRED:
            assign_paired_lanes(race_groups, Rounds.SEMI, len(race_context.races) + 1, race_context=race_context)
        else:
            assign_all_lanes(race_groups, Rounds.SEMI, len(race_context.races) + 1, race_context=race_context)

@require_race_context
def schedule_final_races(schedule_type: RaceScheduleType, race_context: RaceContext):
    if not all(race_context.semi_final_races_completed.values()):
        return

    # 1. Get Top Racers from Semi-Finals:
    top_racers = []
    for patrol in dict(sorted(race_context.patrol_names.items())):
        if patrol != "Exhibition" and race_context.semi_final_races_completed.get(patrol, False):  # Check if semi-finals are complete
            top_racer,_ = get_top_racers(Rounds.SEMI, patrol, 1, race_context=race_context) # Only want to get top 1
            if top_racer:
                top_racers.append(top_racer[0])

    # 2. Create Final Race (using all lanes assignment):
    if top_racers: # Only if there are top racers
        race_groups = [top_racers]
        if schedule_type == RaceScheduleType.PAIRED:
            assign_paired_lanes(race_groups, Rounds.FINAL, len(race_context.races) + 1, race_context=race_context)
        else:
            assign_all_lanes(race_groups, Rounds.FINAL, len(race_context.races) + 1, race_context=race_context)

@require_race_context
def get_top_racers(round: Rounds, patrol = None, racer_count=NUM_LANES, race_context: RaceContext = None):
    top_racers = []
    overall_racer_averages = {}
    if patrol is not None and patrol != "":
        filtered_races = [r for r in race_context.races if r.round == round and r.patrol == patrol]
    else:
        filtered_races = [r for r in race_context.races if r.round == round and r.patrol != "Exhibition"]
    if filtered_races: # Only if semi-final races exist for this patrol
        all_racer_averages = {}
        for race in filtered_races:
            race_averages = calculate_race_averages(race)
            for racer, avg_time in race_averages.items():
                if racer not in all_racer_averages:
                  all_racer_averages[racer] = []
                all_racer_averages[racer].append(avg_time)

        # Calculate overall average time and sort
        for racer, avg_times in all_racer_averages.items():
            overall_racer_averages[racer] = sum(avg_times) / len(avg_times) if avg_times else float('inf')

        sorted_racers = sorted(overall_racer_averages.items(), key=lambda item: item[1])
        top_racers = [racer for racer, avg_time in sorted_racers[:racer_count] 
                      if avg_time != float('inf')]
    return top_racers, overall_racer_averages


def group_racers(racers):
    groups = []
    num_racers = len(racers)
    if num_racers <= NUM_LANES:
        groups = [racers + [None] * (NUM_LANES - num_racers)]
    else:
        # Get our total number of races
        num_races = (num_racers + NUM_LANES - 1) // NUM_LANES
        # Minimum number of cars per race
        base_racers_per_race = num_racers // num_races
        remainder = num_racers % num_races
        distribution = [base_racers_per_race + 1] * remainder + \
                [base_racers_per_race] * (num_races - remainder)
        racer_idx = 0
        for d in distribution:
            race = racers[racer_idx:racer_idx+d] + [None] * (NUM_LANES-d)
            racer_idx += d
            groups.append(race)
    return groups

@require_race_context
def assign_paired_lanes(groups, round: Rounds, race_number_start, race_context: RaceContext):
    lanes_half_a = [i for i in range(0,int(NUM_LANES/2))]
    lanes_half_b = [i for i in range(int(NUM_LANES/2),NUM_LANES)]
    swapped_lanes = list(reversed(lanes_half_a)) + list(reversed(lanes_half_b))
    # Create combinations and assign lanes:
    for grp in groups:
        race = Race(grp[0].patrol if grp[0] else None, race_number_start)
        race_number_start += 1
        race.round = round
        # Create two heats, swapping lanes on the second heat
        for heat_num in [1, 2]:
            heat = Heat(heat_num)
            # Then assign lanes normally for each half for the first heat
            # Then reassign them in half-reverse for the second heat
            # i.e. 0,1,2,3 then 1,0,3,2 for 4 lanes
            if heat_num == 1:
                for lane in range(NUM_LANES):
                    if lane < len(grp):
                        heat.lanes[lane+1] = grp[lane] if grp[lane] else None
            else:
                for lane in range(NUM_LANES):
                    if lane < len(grp):
                        heat.lanes[swapped_lanes[lane]+1]=grp[lane] if grp[lane] else None
            race.heats.append(heat)
        race_context.races.append(race)


@require_race_context
def assign_all_lanes(race_group, round: Rounds, race_number_start, race_context: RaceContext):
    for grp in race_group: # Iterate through race groups (only one in the finals)
        race = Race(grp[0].patrol if grp[0] else None, race_number_start) # Assign patrol and race number
        race_number_start += 1
        race.round = round
        if len(grp) < NUM_LANES:
            for i in range(len(grp), NUM_LANES):
                grp.append(None) # Pad with None if fewer racers than lanes
        num_cars = len(grp)
        # Assigning Lanes:
        # Goal is to have every car run in every lane    
        # If more cars run than lanes, then a different
        # car will sit out each race.
        for heat_idx in range(num_cars):
            heat = Heat(heat_idx + 1) # Heat numbers start at 1
            for lane in range (NUM_LANES):
                heat.lanes[lane+1] = grp[(heat_idx + lane) % num_cars] if grp[(heat_idx + lane) % num_cars] else None # Lane numbers start at 1
            race.heats.append(heat)
        race_context.races.append(race)


@app.route("/enter_times/<int:race_number>/<int:heat_number>", methods=["GET", "POST"])
@require_role('OWNER')
@require_race_context
def enter_times(race_number, heat_number, race_context:  RaceContext):
    race = next((r for r in race_context.races if r.race_number == race_number), None)
    if not race:
        logger.warning('enter_times: Race not found race_number=%s', race_number)
        return respond_error("Race not found", 404, "index")
    heat = next((h for h in race.heats if h.heat_number == heat_number), None)
    if not heat:
        logger.warning('enter_times: Heat not found race_number=%s heat_number=%s', race_number, heat_number)
        return respond_error("Heat not found", 404, "index")

    if request.method == "POST":
        submit = request.form.get("submit", None)
        logger.info('enter_times POST start race=%s heat=%s by uid=%s', race_number, heat_number, session.get('fb_uid') or session.get('role'))
        entered_count = 0
        for lane in range(1, NUM_LANES + 1):
            time_key = f"time_race_{race_number}_heat_{heat_number}_lane_{lane}"
            time = request.form.get(time_key, None)
            if time is not None:
                try:
                    time = float(time)
                    heat.times[lane] = time
                    participant = heat.lanes.get(lane)
                    if participant:
                        participant.times.append(time)
                        calculate_race_statistics(participant, race_context=race_context)
                    entered_count += 1
                except ValueError:
                    orig = request.form.get(time_key)
                    return respond_error("Invalid time input", 400, "index")
        save_data(context=race_context) # Saving data after entering race times
        logger.info('enter_times POST complete race=%s heat=%s entered=%s', race_number, heat_number, entered_count)

        if heat_number == len(race.heats):
            next_race = next((r for r in race_context.races if r.race_number == race_number+1), None)
            if next_race is not None and next_race.patrol == race.patrol:
                next_race_number = next_race.race_number
                next_heat_number = 1
            else:
                next_race_number = None
                next_heat_number = None
        else:
            next_race_number = race.race_number
            next_heat_number = heat_number + 1
            if next((r for r in race_context.races if r.race_number == next_race_number), None) == None:
                next_race_number = None
                next_heat_number = None
                
        if submit == "Submit & Go to Next Race" and next_race_number is not None:
            # Go to the next race
            return redirect(url_for("enter_times", race_number=next_race_number, heat_number=next_heat_number)) # Add round parameter
        else:
            # Go to the list
            if race.round == Rounds.SEMI:
                return redirect(url_for("display_results", patrol=race.patrol)) # Go back unfiltered
            else:
                return redirect(url_for("schedule", patrol=race.patrol, round=race.round.value)) # Add round parameter

    return render_template("enter_times.html", race=race, heat=heat, NUM_LANES=NUM_LANES, archived=race_context.archived)

@require_race_context
def check_races_complete(patrol, round, race_context:RaceContext):
    filtered_races = [r for r in race_context.races if r.patrol == patrol and r.round == round]
    filtered_heats = [h for r in filtered_races for h in r.heats]
    total_assigned_lanes = len([l for h in filtered_heats for l in h.lanes if h.lanes[l] is not None])
    total_times = len([t for h in filtered_heats for t in h.times if h.times[t] is not None])

    return total_assigned_lanes == total_times if total_assigned_lanes > 0 else None

@require_race_context
def check_races_scheduled(patrol, round, race_context: RaceContext):
    filtered_races = [r for r in race_context.races if r.patrol == patrol and r.round == round]
    filtered_heats = [h for r in filtered_races for h in r.heats]
    total_assigned_lanes = len([l for h in filtered_heats for l in h.lanes if h.lanes[l] is not None])

    return total_assigned_lanes > 0

@require_race_context
def check_round_complete(round, race_context: RaceContext):
    for p in race_context.patrol_names:
        if p == "Exhibition":
            race_context.semi_final_races_completed[p] = True
        else:
            complete = check_races_complete(p, round, race_context=race_context)
            if complete is not None:
                if round == Rounds.FIRST:
                    race_context.initial_races_completed[p] = complete
                elif round == Rounds.SEMI:
                    race_context.semi_final_races_completed[p] = complete

@app.route("/schedule_initial")
@require_role('OWNER')
@require_race_context
def schedule_initial(race_context: RaceContext):
    schedule_type = RaceScheduleType(int(request.args.get("schedule_type", RaceScheduleType.PAIRED.value)))
    logger.info('schedule_initial requested by uid=%s', session.get('fb_uid') or session.get('role'))
    clear_races(race_context=race_context)
    schedule_initial_races(schedule_type, race_context=race_context)
    save_data(context=race_context) # Saving data after scheduling initial round
    logger.info('schedule_initial completed; schedule_type=%s, total races=%s', schedule_type.name, len(race_context.races))
    return redirect(url_for("schedule", round=Rounds.FIRST.value)) # Redirect to the main schedule page

@app.route("/schedule_semifinal/<patrol>")
@require_role('OWNER')
@require_race_context
def schedule_semifinal(patrol, race_context: RaceContext):
    schedule_type = RaceScheduleType(int(request.args.get("schedule_type", RaceScheduleType.PAIRED.value)))
    logger.info('schedule_semifinal requested for patrol=%s by uid=%s', patrol, session.get('fb_uid') or session.get('role'))
    schedule_semi_final_races(patrol, schedule_type, race_context=race_context)
    save_data(context=race_context) # Saving data after scheduling semi-final round
    logger.info('schedule_semifinal completed for patrol=%s, schedule_type=%s, total races=%s', patrol, schedule_type.name, len(race_context.races))
    return redirect(url_for("schedule", patrol=patrol, round=Rounds.SEMI.value)) # Add round parameter

@app.route("/schedule_final")
@require_role('OWNER')
@require_race_context
def schedule_final(race_context: RaceContext):
    schedule_type = RaceScheduleType(int(request.args.get("schedule_type", RaceScheduleType.ALL_LANES.value)))
    logger.info('schedule_final requested by uid=%s', session.get('fb_uid') or session.get('role'))
    schedule_final_races(schedule_type,race_context=race_context)
    save_data(context=race_context) # Saving data after scheduling final round
    logger.info('schedule_final completed; schedule_type=%s, total races=%s', schedule_type.name, len(race_context.races))
    return redirect(url_for("schedule", round=Rounds.FINAL.value)) # Add round parameter

@app.route("/schedule", methods=["GET"])
@require_race_context
def schedule(race_context: RaceContext):
    selected_patrol = request.args.get("patrol", "")
    selected_round_str = request.args.get("round", str(Rounds.FIRST.value)) # Get round from query params
    try:
        selected_round = Rounds(int(selected_round_str))  # Convert to Rounds enum
    except ValueError:
        selected_round = Rounds.FIRST  # Default to first round if invalid round value
    if selected_round == Rounds.FINAL:
        selected_patrol = ""

    selected_round_name = ""
    if selected_round == Rounds.FIRST:
        selected_round_name = "First Round"
    elif selected_round == Rounds.SEMI:
        selected_round_name = "Semi-Finals"
    elif selected_round == Rounds.FINAL:
        selected_round_name = "Finals"
    elif selected_round == Rounds.NONE:
        selected_round_name = "All Rounds"

    check_round_complete(Rounds.FIRST, race_context=race_context)
    check_round_complete(Rounds.SEMI, race_context=race_context)
    save_data(context=race_context)

    if race_context.semi_final_races_completed:
        all_semi_final_races_completed = all([v for k,v in race_context.semi_final_races_completed.items() if k != "Exhibition"])
    else:
        all_semi_final_races_completed = False
    semi_final_races_scheduled = {p: check_races_scheduled(p, Rounds.SEMI, race_context=race_context) for p in race_context.patrol_names}

    return render_template("schedule.html", 
                           races=race_context.races, 
                           patrol_names=race_context.patrol_names,
                           selected_patrol=selected_patrol, 
                           selected_round=selected_round,
                           selected_round_value=selected_round.value,
                           selected_round_name = selected_round_name, 
                           Rounds=Rounds, 
                           RaceScheduleType=RaceScheduleType,
                           NUM_LANES=NUM_LANES,
                           initial_races_completed=race_context.initial_races_completed,
                           semi_final_races_completed=race_context.semi_final_races_completed,
                           all_semi_final_races_completed=all_semi_final_races_completed,
                           semi_final_races_scheduled=semi_final_races_scheduled,
                           archived=race_context.archived)

def calculate_race_averages(race: Race):
    racer_race_times = {}  # Store total times for racers in the race
    racer_heat_counts = {} # Store number of heats a racer participated in

    for heat in race.heats:
        for lane, racer in heat.lanes.items():
            if racer and heat.times.get(lane) is not None:
                if racer not in racer_race_times:
                    racer_race_times[racer] = 0
                    racer_heat_counts[racer] = 0
                racer_race_times[racer] += heat.times[lane]
                racer_heat_counts[racer] += 1
    
    racer_averages = {}
    for racer, total_time in racer_race_times.items():
        racer_averages[racer] = total_time / racer_heat_counts[racer] if racer_heat_counts[racer] > 0 else float('inf') # Handle cases where a racer might not have any times recorded

    return racer_averages

def calculate_race_statistics(participant: Participant, race_context: RaceContext):
    p = participant
    if p.times:
        p.best_time = min(p.times)
        p.average_time = sum(p.times) / len(p.times)
        p.best_time_race_number = get_best_time_race_number(p, race_context=race_context)
    else:
        p.best_time = float('inf')
        p.average_time = 0
        p.best_time_race_number = None

def get_best_time_race_number(participant: Participant, race_context: RaceContext):
    if participant.times:
        if participant.best_time > 0:
            for race in race_context.races:
                for heat in race.heats:
                    for lane,p in heat.lanes.items():
                        if p == participant and lane in heat.times and \
                           heat.times[lane] == participant.best_time:
                               return f"{race.race_number}-{heat.heat_number}" 
    return None

@app.route("/judge_design", methods=["GET", "POST"])
@require_role('JUDGE')
@require_race_context
def judge_design(race_context: RaceContext):

    race_id = request.args.get('race') or request.form.get("race") or session.get('current_race_id') or CURRENT_RACE_ID
    
    # Determine judge identity
    judge_id = None
    judge_name = "Judge"
    
    if session.get('judge_name') and session.get('judge_name') in race_context.judges:
        # Token-based judge
        judge_name = session.get('judge_name')
        judge_id = race_context.judges[judge_name]['id']
    elif session.get('fb_uid'):
        # Firebase user
        judge_id = session.get('fb_uid')
        judge_name = session.get('fb_email') or "JUDGE"
    elif session.get('role') in [Role.OWNER.name, Role.ADMIN.name]:
        # Owner/Admin token login
        judge_id = session.get('role') # 'OWNER' or 'ADMIN'
        judge_name = session.get('role').title()

    if not judge_id:
        flash("Unable to identify judge account.", "danger")
        return redirect(url_for('index'))

    racers = {}
    for patrol in race_context.patrol_names:
        racers[patrol] = [p for p in race_context.participants if p.patrol == patrol]

    if not race_context.judging_active:
        flash("Judging is currently closed.", "warning")

    if request.method == "POST" and race_context.judging_active:
        for p in race_context.participants:
            rank = request.form.get(f"rank_{p.participant_id}", 0)
            if rank:
                race_context.designs[p.participant_id].scores[judge_id] = int(rank)
        save_data(race_id=race_id, context=race_context)
        return redirect(url_for("design_results"))

    return render_template("judge_design.html", racers=racers, patrol_names=race_context.patrol_names,
                            judge_id=judge_id, designs=race_context.designs, judge_name=judge_name, 
                            judging_active=race_context.judging_active, archived=race_context.archived)

@app.route("/close_judging")
@require_race_context
def close_judging(race_context: RaceContext):
    race_context.judging_active = False
    save_data(context=race_context)
    return redirect(url_for("design_results"))

@app.route("/open_judging")
@require_race_context
def open_judging(race_context: RaceContext):
    race_context.judging_active = True
    save_data(context=race_context)
    return redirect(url_for("design_results"))


def score_design(design: Design):
    total_score = 0
    first = 0
    second = 0
    third = 0
    for judge_rank in design.scores.values():
        if judge_rank == 1:
            total_score += 3
            first += 1
        elif judge_rank == 2:
            total_score += 2
            second += 1
        elif judge_rank == 3:
            total_score += 1
            third += 1

    return (total_score,first,second,third)

@app.route("/design_results")
@require_race_context
def design_results(race_context: RaceContext):

    sorted_scores_by_patrol = {}

    for patrol in race_context.patrol_names:
        patrol_scores = []
        for p in race_context.participants:
            if p.patrol == patrol:
                design_score = score_design(race_context.designs[p.participant_id])
                patrol_scores.append((p, design_score[0], design_score[1],
                                      design_score[2], design_score[3]))

        # Sort the scores by total score, #first votes, #second votes, #third votes
        patrol_scores.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4]))
        sorted_scores_by_patrol[patrol] = patrol_scores[:3] #Take the top 3

    judges_data = race_context.judges
    for j_name, jinfo in judges_data.items():
        jinfo['voted'] = any([des.scores[jinfo['id']] for des in race_context.designs.values() if jinfo['id'] in des.scores.keys()])

    voting_status = "Not Started"
    voting_modifier = "is-danger"
    voted_data = [jinfo['voted'] for jinfo in judges_data.values()]
    if any(voted_data):
        voting_status = "In Progress"
        voting_modifier = "is-warning"
    if all(voted_data):
        voting_status = "Complete"
        voting_modifier = "is-success"

        
   

    return render_template("design_results.html", 
                           patrol_names=race_context.patrol_names, 
                           sorted_scores_by_patrol=sorted_scores_by_patrol,
                           judging_active=race_context.judging_active,
                           voting_status=voting_status, 
                           voting_modifier=voting_modifier,
                           archived=race_context.archived)


@app.route("/display_results")
@require_race_context
def display_results(race_context: RaceContext):
    patrol = request.args.get("patrol", None)
    valid_participants = [p for p in race_context.participants if p.times]
    sorted_participants = sorted(valid_participants, key=lambda x: (x.average_time, x.best_time))
    for p in sorted_participants:
        calculate_race_statistics(p, race_context=race_context)
    return render_template("results.html", 
                           participants=sorted_participants, 
                           patrol_names=race_context.patrol_names,
                           patrol=patrol)

@app.route("/download_results")
@require_race_context
def download_results(race_context: RaceContext):
    output = StringIO()
    writer = csv.writer(output)

    # Write header row
    writer.writerow(["Name", "Patrol", "Car Number", "Car Name", "Average Time", "Top Time", "Top Time Race #"])

    for p in race_context.participants:  # Assuming participants is your list of racers
        writer.writerow([p.first_name + " " + p.last_name, race_context.patrol_names.get(p.patrol), p.car_number, p.car_name, p.average_time, p.best_time, p.best_time_race_number])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=race_results.csv"},
    )

@require_race_context
def get_participant_race_data(participant_id, race_context: RaceContext):
    header = ["Round", "Race Number", "Heat Number", "Lane", "Time"]

    race_data = [ (race.round.name, race_index+1, heat.heat_number, lane, heat.times[lane]) \
            for race_index, race in enumerate(race_context.races) \
            for heat in race.heats \
            for lane, p in heat.lanes.items() \
            if p and p.participant_id == participant_id and lane in heat.times]

    return (header, race_data)


@app.route("/download_racer_data/<participant_id>")
@require_race_context
def download_racer_data(participant_id, race_context: RaceContext):
    participant = next((p for p in race_context.participants if p.participant_id == participant_id), None)
    if not participant:
        logger.warning('download_racer_data: participant not found id=%s', participant_id)
        return "Participant not found", 404

    output = StringIO()
    writer = csv.writer(output)

    header, race_data = get_participant_race_data(participant_id, race_context=race_context)

    # Write header row
    writer.writerow(header)  # Customize headers as needed
    for row in race_data:
        writer.writerow(row)
    
    fn = f"{race_context.race_id}_{participant.first_name}_{participant.last_name}_{participant.car_name}_race_data.csv"

    safe_fn = re.sub("[^a-zA-Z0-9_.-]", "_", fn)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={safe_fn}"},
    )

@app.route("/upload_roster", methods=["GET", "POST"])
@require_role('OWNER')
def upload_roster():
    role = session.get('role', Role.VIEWER.name)
    error_message = ""

    if Role[role] == Role.OWNER:
        if request.method == "POST":
            if 'file' not in request.files:
                flash('Must provide file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('Must provide file')
                return redirect(request.url)

            if file and (file.filename.endswith('.csv') or file.filename.endswith(".txt")): # Check file extension
                try:
                    # Load from memory
                    csv_file = file.stream.read().decode('utf-8')
                    race_context = load_roster_from_memory(csv_file) # See function below
                    save_data(context=race_context) # Save data after loading roster
                    logger.info('Roster uploaded successfully by uid=%s filename=%s', session.get('fb_uid') or session.get('role'), file.filename)
                    flash('Roster uploaded successfully!')
                    return redirect(url_for('index')) # Redirect to your main page

                except Exception as e:
                    flash(f'Error uploading roster: {str(e)}') # Display error message
                    traceback.print_exc()

            else:
                flash('Invalid file type. Please upload a CSV file.')

            return redirect(request.url) # Redirect back to the upload page
    else:
        error_message="Only the owner can upload racer data."

    return render_template("upload_roster.html", error_message="")

@require_race_context
def load_roster_from_memory(csv_string, race_context: RaceContext):
    csvfile = StringIO(csv_string)
    reader = csv.DictReader(csvfile)
    num_loaded = 0
    for row in reader:
        num_loaded += 1
        p = _add_participant_to_race(row["First Name"], row["Last Name"], row["Patrol"], race_context=race_context)
        logger.debug('Loaded roster row added participant id=%s name=%s %s patrol=%s', p.participant_id, p.first_name, p.last_name, p.patrol)
        if "car_weight_oz" in row and row['car_weight_oz'].strip() != "":
            try:
                if float(row["car_weight_oz"]) >= 0:
                    p.car_weight_oz = float(row["car_weight_oz"]) # Convert to float
                else: 
                    logger.warning("Negative car_weight_oz ignored: %s", row.get('car_weight_oz'))  # Log the error
                    raise ValueError("Negative weight")
            except ValueError:
                logger.warning("Invalid car_weight_oz: %s", row.get('car_weight_oz'))  # Log the error
    logger.info(f"Loaded {num_loaded} participants from roster upload.")
    return race_context

@app.route("/download_roster_template")
@require_race_context
def download_roster_template(race_context: RaceContext):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["First Name", "Last Name", "Patrol", "car_weight_oz"]) # Header row
    for patrol in race_context.patrol_names.values():
        writer.writerow(["", "", patrol, ""])  # Empty rows for each patrol
    for p in race_context.participants:
        writer.writerow([p.first_name, p.last_name, race_context.patrol_names[p.patrol], p.car_weight_oz])
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=roster_template.csv"},
    )

@app.route("/api/participants")
@require_race_context
def api_participants(race_context: RaceContext):
    return jsonify([p.toJSON() for p in race_context.participants])

@app.route("/api/patrol_names")
@require_race_context
def api_patrol_names(race_context: RaceContext):
    return jsonify(race_context.patrol_names)

@app.route("/api/races")
@require_race_context
def api_races(race_context: RaceContext):
    return jsonify([r.toJSON() for r in race_context.races])

@app.route("/api/designs")
@require_race_context
def api_designs(race_context: RaceContext):
    return jsonify({p: d.toJSON() for p, d in race_context.designs.items()})

@app.errorhandler(404)
def page_not_found(e):
    logger.warning('404 Not Found: %s %s', request.method, request.path)
    return render_template('404.html'), 404


@app.errorhandler(500)
def handle_500(e):
    logger.exception('Unhandled exception during request %s %s', request.method, request.path)
    return render_template('404.html'), 500


if __name__ == "__main__":
    # Honor FLASK_DEBUG environment variable when running with the builtin server.
    # In production use Gunicorn: `gunicorn RaceManager.app:app` (the block below won't run).
    debug_mode = os.environ.get('FLASK_DEBUG', os.environ.get('DEBUG', '0')).lower() in ('1', 'true', 'yes')
    app.run(host='0.0.0.0', debug=debug_mode)

