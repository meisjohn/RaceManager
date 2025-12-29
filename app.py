from glob import glob
import logging
import sys
from typing import Dict, List
from flask import Flask, render_template, request, redirect, url_for, \
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

# Basic startup log (CURRENT_RACE_ID is set later)
logger.info("Starting RaceManager app; DEFAULT_RACE not yet resolved; LOG_LEVEL=%s", LOG_LEVEL)


@app.before_request
def ensure_request_id():
    # propagate incoming request id or generate one
    rid = request.headers.get('X-Request-ID') or uuid.uuid4().hex
    try:
        g.request_id = rid
    except Exception:
        pass

# Optional Firestore support. Enable by setting USE_FIRESTORE=1 and
# providing Cloud Run service account or GOOGLE_APPLICATION_CREDENTIALS.
try:
    from google.cloud import firestore as _gcf
    FIRESTORE_AVAILABLE = True
except Exception:
    _gcf = None
    FIRESTORE_AVAILABLE = False

CURRENT_RACE_ID = os.environ.get("DEFAULT_RACE", "demo")
CONFIG_DIR = os.environ.get("CONFIG_DIR", ".")
USE_FIRESTORE = os.environ.get("USE_FIRESTORE", "0").lower() in ("1", "true", "yes")


# Config File templates and constants
DATA_FILE_TEMPLATE = "race_data_{race_id}.json"
RACE_MEMBERS_FILE_TEMPLATE = "race_members_{race_id}.json"
PATROL_CONFIG_FILE = "patrol_config.json"
AUTH_CONFIG_FILE = "auth_config.json"

NUM_LANES = os.getenv("NUM_LANES", 4)


# In-memory per-race cache (default TTL seconds). Use Redis later for cross-instance caching.
CACHE_TTL = int(os.environ.get('CACHE_TTL', '30'))
_race_cache = {}  # race_id -> {'data': ..., 'ts': float}
_race_cache_locks = {}
_race_cache_master_lock = threading.Lock()

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
            return None
    age = time.time() - entry.get('ts', 0)
    #logger.debug(f"Loaded data: {entry.get('data')}")
    logger.debug('Cache hit for race %s (age=%.1fs)', race_id, age)
    return entry.get('data')

def set_race_cached(race_id, data):
    try:
        with _get_race_lock(race_id):
            #logger.debug(f"data: {data}")
            _race_cache[race_id] = {'data': data, 'ts': time.time()}
            logger.info('Set in-memory cache for race %s', race_id)
    except Exception:
        logger.exception('Failed to set in-memory cache for race %s', race_id)

def invalidate_race_cache(race_id):
    try:
        with _get_race_lock(race_id):
            _race_cache.pop(race_id, None)
            logger.info('Invalidated in-memory cache for race %s', race_id)
    except Exception:
        logger.exception('Failed to invalidate cache for race %s', race_id)

def data_filename_for_race(race_id):
    return os.path.join(CONFIG_DIR, DATA_FILE_TEMPLATE.format(race_id=str(race_id)))


def read_race_doc(race_id):
    """Return the race-level document/data dict from Firestore or local file."""
    try:
        if USE_FIRESTORE and FIRESTORE_AVAILABLE:
            db = _gcf.Client()
            doc = db.collection('races').document(str(race_id)).get()
            return doc.to_dict() if doc.exists else {}
        else:
            fn = data_filename_for_race(race_id)
            try:
                with open(fn, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
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
            db = _gcf.Client()
            if merge:
                db.collection('races').document(str(race_id)).set(data, merge=True)
            else:
                db.collection('races').document(str(race_id)).set(data)
            return True
        else:
            fn = data_filename_for_race(race_id)
            existing = {}
            if merge and os.path.exists(fn):
                try:
                    with open(fn, 'r') as f:
                        existing = json.load(f)
                except Exception:
                    existing = {}
            if merge:
                for k, v in data.items():
                    existing[k] = v
                to_write = existing
            else:
                to_write = data
            with open(fn+".tmp", 'w') as f:
                json.dump(to_write, f, default=str)
            os.replace(fn+".tmp", fn) # atomic replace
            return True
    except Exception:
        logger.exception('Error writing race doc %s', race_id)
        return False

# Initialize Firebase Admin SDK for auth (used to verify ID tokens and set custom claims).
FIREBASE_ADMIN_AVAILABLE = False
try:
    # If a service account path provided in env, use it; otherwise rely on application default
    cred_path = os.environ.get('FIREBASE_CREDENTIALS')
    if cred_path:
        cred = fb_credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    else:
        # This will work on Cloud Run with Workload Identity or ADC
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
    FIREBASE_ADMIN_AVAILABLE = True
except Exception as e:
    logger.warning("Firebase Admin init failed or not configured: %s", e, exc_info=True)
    FIREBASE_ADMIN_AVAILABLE = False

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
            logger.debug('flash() failed when responding with HTML error')
        if redirect_endpoint:
            try:
                target = url_for(redirect_endpoint, **(redirect_kwargs or {}))
            except Exception:
                target = request.referrer or url_for('index')
        else:
            target = request.referrer or url_for('index')
        return redirect(target)
    return jsonify({'error': message}), status

def require_top_admin(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Session-based top-admin allowed (include legacy Owner role for local bootstrap)
        if session.get('isTopAdmin'):
            return func(*args, **kwargs)
        # Legacy Owner role acts as bootstrap admin for local-only testing
        role = session.get('role')
        if role and role == Role.OWNER.name:
            return func(*args, **kwargs)
        user = getattr(request, 'fb_user', None)
        if user and user.get('isTopAdmin'):
            return func(*args, **kwargs)
        logger.warning('Top-level admin required but not present for request to %s', request.path)
        return respond_error('Top-level admin required', 403, 'login')
    return wrapper


def require_role(min_role):
    roles_order = {'viewer': 1, 'editor': 2, 'admin': 3}
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # session owner bypass
            role = session.get('role')
            if role and Role[role] == Role.OWNER:
                return func(*args, **kwargs)

            # check firebase token and custom claims
            auth_header = request.headers.get('Authorization', '')
            m = re.match(r"Bearer\s+(.+)", auth_header)
            if FIREBASE_ADMIN_AVAILABLE and m:
                try:
                    decoded = fb_auth.verify_id_token(m.group(1))
                    # top-level admin bypass
                    if decoded.get('isTopAdmin'):
                        return func(*args, **kwargs)
                    uid = decoded.get('uid')
                except Exception as e:
                    logger.warning('Token verify in require_role failed: %s', e, exc_info=True)
                    return respond_error('Invalid token', 401, 'login')
            else:
                uid = None

            # determine race id
            race_id = request.args.get('race') or CURRENT_RACE_ID

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

            if member_role and roles_order.get(member_role, 0) >= roles_order.get(min_role, 0):
                return func(*args, **kwargs)
            logger.warning('Insufficient role for request %s: required=%s member_role=%s uid=%s', request.path, min_role, member_role, uid)
            return respond_error('Insufficient role', 403, 'login')
        return wrapper
    return decorator

def members_filename_for_race(race_id):
    return os.path.join(CONFIG_DIR, RACE_MEMBERS_FILE_TEMPLATE.format(race_id=str(race_id)))

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

# Load patrol names from JSON config file


def patrol_config_filename():
    return os.path.join(CONFIG_DIR, PATROL_CONFIG_FILE)

def auth_config_filename():
    return os.path.join(CONFIG_DIR, AUTH_CONFIG_FILE)

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

# DATA_FILE is per-race. Use `data_filename_for_race(race_id)` or Firestore when enabled.

class Role(Enum):
    PUBLIC = 0
    JUDGE = 1
    OWNER = 2

class Rounds(Enum):
    NONE = 0
    FIRST = 1
    SEMI = 2
    FINAL = 3

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
                "lanes": {lane: p.participant_id if p else None for lane, p in self.lanes.items()},
                "times": self.times
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
    def __init__(self, race_id, name, participants, races, designs, judges,
                 initial_races_completed, semi_final_races_completed,
                 judging_active, patrol_names = default_patrol_names):
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

def require_race_context(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'race_context' not in kwargs:
            if 'race_id' in kwargs:
                race_id = kwargs['race_id']
            else:
                race_id = session.get('current_race_id') or request.args.get('race') or request.form.get("race") or CURRENT_RACE_ID
            context = load_data(race_id=race_id)
            kwargs['race_context'] = context
        return func(*args, **kwargs)
    return wrapper

def create_race(name):

    # If race_id not provided, derive from name
    if not name:
        raise RuntimeError('Must specify name whenc creating a race')
    # create a simple URL-friendly id: keep alphanum, dash, underscore; replace other chars/spaces with underscore
    rid = re.sub(r"[^A-Za-z0-9_-]+", '_', name)
    rid = re.sub(r'_+', '_', rid).strip('_')
    race_id = rid[:120] or name.replace(' ', '_')
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
    global CURRENT_RACE_ID

    if race_id is None:
        race_id = CURRENT_RACE_ID
    CURRENT_RACE_ID = str(race_id)
    if name is None:
        try:
            name = session.get('current_race_name', str(race_id))
        except RuntimeError:
            name = str(race_id)

    # Try in-memory cache first
    cached = get_race_cached(CURRENT_RACE_ID)
    if cached is not None:
        logger.debug('Loaded race %s from in-memory cache', CURRENT_RACE_ID)
        data = cached
    else:
        data = read_race_doc(CURRENT_RACE_ID)

        if not data:
            logger.warning("Creating new race in load_data")
            data = create_race(name=name)
            resave=True

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
        patrol_names=data.get('patrol_names', default_patrol_names))

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

    for p_data in data.get('participants', []):
        p = Participant(p_data.get('first_name'), p_data.get('last_name'), p_data.get('patrol'))
        p.__dict__.update(p_data)
        context.participants.append(p)

    for r_data in data.get('races', []):
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

    try:
        session['current_race_id'] = context.race_id
        session['current_race_name'] = context.name
    except Exception:
        logger.debug('Unable to set session current_race_id after firestore create')

    if resave:
        save_data(CURRENT_RACE_ID, context=context)

    return context

def save_data(race_id=None, context: RaceContext=None):
    """Save current in-memory state to Firestore (if enabled) or to per-race JSON file."""
    if race_id is None:
        if context:
            race_id = context.race_id
        else:
            race_id = CURRENT_RACE_ID

    if context:
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
    except Exception:
        logger.debug('Failed to save race data %s', race_id)
    try:
        set_race_cached(race_id, payload)
    except Exception:
        logger.debug('Failed to set cache after local save for %s', race_id)

# Ensure a race is loaded at startup and allow switching via ?race=ID
@app.before_request
def ensure_race_loaded():
    # Log basic request context and selected race param
    logger.debug('Incoming request: %s %s args=%s uid=%s', request.method, request.path, dict(request.args), session.get('fb_uid'))
    # If a race parameter is provided in the request, load that race's data
    race_param = request.args.get('race')
    global CURRENT_RACE_ID
    try:
        if race_param and race_param != CURRENT_RACE_ID:
            logger.info('Request to switch race from %s to %s based on request param', CURRENT_RACE_ID, race_param)
            # Only load the requested race if it already exists or the actor is authorized to create it.
            allowed_to_create = False
            # Determine actor privileges (session or firebase)
            session_role = session.get('role')
            session_is_top = session.get('isTopAdmin')
            session_uid = session.get('fb_uid')
            fb_user = getattr(request, 'fb_user', None)
            if fb_user and fb_user.get('isTopAdmin'):
                session_is_top = True

            # If using Firestore, check whether the race document exists
            race_exists = False
            if USE_FIRESTORE and FIRESTORE_AVAILABLE:
                try:
                    db = _gcf.Client()
                    doc = db.collection('races').document(str(race_param)).get()
                    if doc.exists:
                        race_exists = True
                except Exception:
                    logger.exception('Error checking firestore for race %s', race_param)
            else:
                # local file based
                filename = data_filename_for_race(race_param)
                if os.path.exists(filename):
                    race_exists = True

            # Allow creation only for top-level admins or the legacy OWNER role
            if session_is_top or session_role == Role.OWNER.name:
                allowed_to_create = True

            if race_exists:
                logger.info('Switching race to existing id %s', race_param)
                race_context = load_data(race_param)
                try:
                    session['current_race_id'] = str(CURRENT_RACE_ID)
                    session['current_race_name'] = str(race_context.name)
                except Exception:
                    logger.debug('Unable to set session current_race_id')
            else:
                if allowed_to_create:
                    logger.info(f'Authorized actor ({session_uid}/{fb_user}/{session_role}) creating new race id {race_param}; creating data.')
                    race_context = load_data(race_param)
                    try:
                        session['current_race_id'] = str(CURRENT_RACE_ID)
                        session['current_race_name'] = str(race_context.name)
                    except Exception:
                        logger.debug('Unable to set session current_race_id')
                else:
                    msg = f"Requested race '{race_param}' does not exist and you are not authorized to create it."
                    logger.warning('Requested race %s does not exist and actor not authorized to create it; ignoring param', race_param)
                    try:
                        flash(msg, 'warning')
                    except Exception:
                        logger.debug('flash unavailable in this context when ignoring race param %s', race_param)
                    # Do not change CURRENT_RACE_ID; leave previous data loaded.
    except Exception:
        logger.exception('Error while attempting to switch race to %s', race_param)
        traceback.print_exc()
    # If a firebase session exists, populate request.fb_user for downstream checks
    try:
        if not hasattr(request, 'fb_user') and session.get('fb_uid'):
            request.fb_user = {'uid': session.get('fb_uid'), 'isTopAdmin': session.get('isTopAdmin', False), 'email': session.get('fb_email')}
            logger.debug('Populated request.fb_user from session for uid=%s isTop=%s', request.fb_user.get('uid'), request.fb_user.get('isTopAdmin'))
    except Exception:
        logger.exception('Failed to populate fb_user from session')

# Load default race at startup
try:
    load_data(CURRENT_RACE_ID)
except Exception:
    logger.exception("Encountered exception loading saved data")
    traceback.print_exc()


# Inject current race info into all templates
@app.context_processor
def inject_current_race():
    rid = str(session.get('current_race_id') or CURRENT_RACE_ID)
    name = rid
    try:
        if USE_FIRESTORE and FIRESTORE_AVAILABLE:
            try:
                db = _gcf.Client()
                doc = db.collection('races').document(rid).get()
                if doc.exists:
                    name = doc.to_dict().get('name', rid)
                    race_qr = doc.to_dict().get('race_qr', None)
            except Exception:
                logger.debug('Failed to read race name from Firestore for %s', rid)
        else:
            fn = data_filename_for_race(rid)
            try:
                with open(fn, 'r') as f:
                    data = json.load(f)
                    name = data.get('name', rid)
                    race_qr = data.get('race_qr', None)
            except Exception:
                # file may not exist or not contain a name; ignore
                pass
    except Exception:
        logger.exception('Error determining current race name for %s', rid)
    if race_qr is None:
        url = request.url_root.rstrip('/') + url_for('index', race=rid)
        race_qr = generate_qr(url, f'race-{rid}')
    return {'current_race_id': rid, 'current_race_name': name, 'current_race_qr': race_qr}

def get_races_list():
    races_list = []
    if USE_FIRESTORE and FIRESTORE_AVAILABLE:
        try:
            db = _gcf.Client()
            for doc in db.collection('races').list_documents():
                d = doc.get().to_dict() or {}
                races_list.append({'id': doc.id, 'name': d.get('name', '')})
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
                    except Exception:
                        name = rid
                races_list.append({'id': rid, 'name': name})
    return races_list

# --- Admin UI and management endpoints ---
@app.route('/admin/ui')
@require_top_admin
def admin_ui():
    races_list = get_races_list() 
    # Determine data mode for UI hint
    if USE_FIRESTORE:
        if FIRESTORE_AVAILABLE:
            data_mode = 'Firestore'
        else:
            data_mode = 'Firestore (unavailable — using local files)'
    else:
        data_mode = 'Local files'

    return render_template('admin_races.html', races=races_list, data_mode=data_mode, firebase_admin_available=FIREBASE_ADMIN_AVAILABLE)


@app.route('/admin/set_owner_token', methods=['POST'])
@require_top_admin
def admin_set_owner_token():
    # Generate and persist a per-race owner token and optionally a QR image
    race_id = request.form.get('race') or session.get('current_race_id') or CURRENT_RACE_ID
    token_val = uuid.uuid4().hex
    if race_id == 'demo':
        token_val = 'demo_token' # sample token for the demo race
    logger.info('Admin generating owner token for race %s', race_id)
    # persist owner token
    if not write_race_doc(race_id, {'owner_token': token_val}, merge=True):
        return respond_error('Failed to set owner token', 500, 'admin_ui')

    # generate QR that links to the login page with the token shown (owner can use page to submit)
    qr_fn = None
    try:
        url = request.url_root.rstrip('/') + url_for('owner_login', owner_token=token_val, race=race_id)
        qr_fn = generate_qr(url, f'owner-{race_id}')
        # store QR filename alongside token
        write_race_doc(race_id, {'owner_qr': qr_fn}, merge=True)
    except Exception:
        logger.exception('Failed to generate owner QR for %s', race_id)
        qr_fn = None

    try:
        flash('Owner token generated for race ' + str(race_id), 'info')
    except Exception:
        logger.debug('flash unavailable after creating owner token')
    # redirect back to the members page for the race so UI shows the new token/QR
    return redirect(url_for('admin_race_members', race_id=race_id))


@app.route('/admin/create_race', methods=['POST'])
@require_top_admin
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

    actor = getattr(request, 'fb_user', None) or {'uid': session.get('fb_uid'), 'email': session.get('fb_email'), 'role': session.get('role')}
    logger.info('admin_create_race requested by %s for race_id=%s name=%s', actor.get('uid') or actor.get('role'), race_id, name)
 
    try:
        write_race_doc(race_id=race_id, data=payload, merge=True)
    except Exception as e:
        logger.debug('Failed to write race data %s', race_id)
        logger.exception('create race error: %s', e)
        return respond_error("Error creating race", 500, 'admin_ui', flash_category='error')
    try:
        set_race_cached(race_id, payload)
    except Exception:
        logger.debug('Failed to set cache after create for %s', race_id)
        logger.exception('create race error: %s', e)
        return respond_error("Error creating race", 500, 'admin_ui', flash_category='error')
    return redirect(url_for('admin_ui'))

@app.route('/admin/races/<race_id>/members', methods=['GET', 'POST'])
def admin_race_members(race_id):
    # allow top admin or race admin (race admin stored in members)
    user = getattr(request, 'fb_user', None)
    logger.info('admin_race_members access for race %s by %s', race_id, (user.get('uid') if user else session.get('role')))
    # allow session-based owner or firebase session
    session_role = session.get('role')
    session_uid = session.get('fb_uid')
    if not user and not session_role and not session_uid:
        logger.warning('admin_race_members: authentication required for race %s request %s', race_id, request.path)
        return respond_error('Authentication required — please sign in', 401, 'login')

    # Determine if this actor is top-level admin
    is_top = False
    if session.get('isTopAdmin') or session_role == Role.OWNER.name:
        is_top = True
    if user and user.get('isTopAdmin'):
        is_top = True

    # check if user is race admin when not top
    if not is_top:
        # load member role from firestore or local members file
        uid = None
        if user:
            uid = user.get('uid')
        if not uid:
            uid = session_uid
        if not uid:
            logger.warning('admin_race_members: missing uid for race %s', race_id)
            return respond_error('Authentication required — please sign in', 401, 'login')
        if USE_FIRESTORE and FIRESTORE_AVAILABLE:
            try:
                db = _gcf.Client()
                mdoc = db.collection('races').document(race_id).collection('members').document(uid).get()
                mdata = mdoc.to_dict() if mdoc.exists else {}
                if mdata.get('role') != 'admin':
                    logger.warning('admin_race_members: user %s not authorized for race %s (role=%s)', uid, race_id, mdata.get('role'))
                    return respond_error('Not authorized for this race', 403, 'admin_ui')
            except Exception as e:
                logger.exception('admin_race_members: Error checking membership for uid=%s race=%s: %s', uid, race_id, e)
                return jsonify({'error': 'Error checking membership'}), 500
        else:
            local_members = read_members_local(race_id)
            role = local_members.get(uid, {}).get('role')
            if role != 'admin':
                logger.warning('admin_race_members: user %s not authorized for race %s (local role=%s)', uid, race_id, role)
                return respond_error('Not authorized for this race', 403, 'admin_ui')

    if request.method == 'POST':
        # normalize JSON body if provided (silent to avoid 415 on form posts)
        json_body = request.get_json(silent=True) or {}
        # --- Handle judge management actions (regen/remove) from admin UI ---
        regen_judge = request.form.get('regen_judge') or json_body.get('regen_judge') or \
            request.form.get('judge_name') or json_body.get('judge_name')
        remove_judge = request.form.get('remove_judge') or json_body.get('remove_judge')
        if regen_judge:
            # only allow top admins or race owner to regen or create judge tokens
            if not (is_top or session_role == Role.OWNER.name):
                return respond_error('Not authorized to regenerate judge tokens', 403, 'admin_race_members', {'race_id': race_id})
            try:
                data = read_race_doc(race_id)
                judges_data = data.get('judges', {})

                # generate new token and QR for the judge
                token_val = uuid.uuid4().hex
                judges_data.setdefault(regen_judge, {})['token'] = token_val
                judges_data.setdefault(regen_judge, {})['id'] = judges_data.get(regen_judge, {}).get('id') or uuid.uuid4().hex
                url = request.url_root.rstrip('/') + url_for('judge_login', race=race_id, judge_token=token_val, judge_name=regen_judge)
                try:
                    qr=judges_data[regen_judge].get('qr', None)
                    if qr:
                        try:
                            os.remove(os.path.join(app.static_folder, 'qr', qr))
                        except Exception:
                            logger.debug('Failed to remove QR file %s for removed judge %s', qr, remove_judge)
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
            if not (is_top or session_role == Role.OWNER.name):
                return respond_error('Not authorized to remove judges', 403, 'admin_race_members', {'race_id': race_id})
            try:
                data = read_race_doc(race_id)
                judges_data = data.get('judges', {})
                if remove_judge in judges_data:
                    qr=judges_data[remove_judge].get('qr', None)
                    if qr:
                        try:
                            os.remove(os.path.join(app.static_folder, 'qr', qr))
                        except Exception:
                            logger.debug('Failed to remove QR file %s for removed judge %s', qr, remove_judge)
                    judges_data.pop(remove_judge, None)
                if not write_race_doc(race_id, {'judges': judges_data}, merge=True):
                    raise RuntimeError('Failed to persist judges data')
                flash(f'Removed judge {remove_judge}', 'info')
                return redirect(url_for('admin_race_members', race_id=race_id))
            except Exception:
                logger.exception('Error removing judge %s', remove_judge)
                return respond_error('Failed to remove judge', 500, 'admin_race_members', {'race_id': race_id})

        # add or update member
        uid = request.form.get('uid') or json_body.get('uid')
        role = request.form.get('role') or json_body.get('role')
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
        if USE_FIRESTORE and FIRESTORE_AVAILABLE:
            try:
                db = _gcf.Client()
                db.collection('races').document(race_id).collection('members').document(uid).set({'role': role, 'updatedAt': _gcf.SERVER_TIMESTAMP}, merge=True)
                return redirect(url_for('admin_race_members', race_id=race_id))
            except Exception as e:
                logger.exception('assign member error: %s', e)
                return 'error', 500
        else:
            m = read_members_local(race_id)
            existing = m.get(uid, {})
            existing['role'] = role
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
    rd = read_race_doc(race_id)
    judges_data = rd.get('judges', {})
    owner_token = rd.get('owner_token')
    owner_qr = rd.get('owner_qr')    # filename saved by admin_set_owner_token

    return render_template('admin_members.html', race_id=race_id, members=members, judges=judges_data, owner_token=owner_token, owner_qr=owner_qr)


@app.route('/admin/set_top_admin', methods=['POST'])
@require_top_admin
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
    if not id_token:
        logger.warning('session_login called without idToken')
        return 'idToken required', 400
    if not FIREBASE_ADMIN_AVAILABLE:
        logger.error('session_login attempted but Firebase Admin SDK not available')
        return 'Server not configured for token verification', 500
    try:
        decoded = fb_auth.verify_id_token(id_token)
    except Exception as e:
        logger.warning('session_login verify failed: %s', e, exc_info=True)
        return 'Invalid token', 401

    # set session values
    session['fb_uid'] = decoded.get('uid')
    session['isTopAdmin'] = decoded.get('isTopAdmin', False)
    # optional: set a friendly username/email
    session['fb_email'] = decoded.get('email')

    logger.info('session_login successful for uid=%s isTopAdmin=%s email=%s', session.get('fb_uid'), session.get('isTopAdmin'), session.get('fb_email'))

    # If using Firestore/local member files, preload membership roles for convenience
    race_id = request.args.get('race') or CURRENT_RACE_ID
    members = {}
    uid = decoded.get('uid')
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
    session['members'] = members

    return jsonify({'ok': True})


@app.route('/logout')
def logout():
    # Clear both legacy session role and firebase session keys
    logger.info('Logout requested; clearing session for role=%s fb_uid=%s', session.get('role'), session.get('fb_uid'))
    session.pop('role', None)
    session.pop('judge_name', None)
    session.pop('fb_uid', None)
    session.pop('isTopAdmin', None)
    session.pop('fb_email', None)
    session.pop('members', None)
    return redirect(url_for('index'))

@app.route("/owner_login", methods=["GET", "POST"])
def owner_login():
    token = request.form.get("owner_token") or request.args.get("owner_token") or None

    # Determine current race for owner token lookup
    race_id = session.get('current_race_id') or request.args.get('race') or request.form.get("race") or CURRENT_RACE_ID
    # First prefer race-local owner token stored in race data
    owner_token_ok = False
    try:
        rd = read_race_doc(race_id)
        owner_token = rd.get('owner_token')
        if owner_token and token == owner_token:
            owner_token_ok = True
    except Exception:
        logger.exception('Error checking owner token for race %s', race_id)

    # Fallback to global config token for backward compatibility
    if not owner_token_ok and token and token == auth_config.get('owner_token'):
        owner_token_ok = True

    if owner_token_ok:
        session["role"] = Role.OWNER.name
        logger.info('Owner login succeeded for race=%s; role set to OWNER', race_id)
        flash("Logged in as Owner", "success")
        return redirect(url_for("index"))
    logger.warning('Owner login failed with provided token for race=%s', race_id)
    return render_template("login.html", error_message="Invalid Owner token"), 401

def generate_qr(url,id):

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

@app.route("/judge_login", methods=["GET", "POST"])
def judge_login():
    token = request.form.get("judge_token") or request.args.get("judge_token") or None
    race_id = session.get('current_race_id') or request.args.get('race') or request.form.get("race") or CURRENT_RACE_ID
    # First, if a token was provided, see if it matches an existing per-race judge token

    judge_token_ok = False
    try:
        rd = read_race_doc(race_id)
        judges_data = rd.get('judges', {})
        judge_names = [judge_name for judge_name in judges_data if judges_data[judge_name].get('token') == token]
        judge_token = judges_data[judge_names[0]].get('token') if judge_names else None

        if judge_token and token == judge_token:
            judge_token_ok = True
    except Exception:
        logger.exception('Error checking judge token for race %s', race_id)

    # Fallback to global config token for backward compatibility
    if not judge_token_ok and token and token == auth_config.get('judge_token'):
        judge_token_ok = True

    if judge_token_ok:
        session["role"] = Role.JUDGE.name
        session["judge_name"] = judge_names[0]
        logger.info('judge login succeeded for race=%s; role set to judge, judge name=%s', race_id, judge_names[0])
        flash(f"{judge_names[0]} logged in as Judge", "success")
        return redirect(url_for("judge_design"))
    logger.warning('judge login failed with provided token for race=%s', race_id)
    return render_template("login.html", error_message="Invalid Judge token"), 401

@app.route("/login")
def login():
    owner_token = request.args.get("owner_token", None)
    judge_token = request.args.get("judge_token", None)
    return render_template("login.html", owner_token=owner_token, judge_token=judge_token, races_list=get_races_list() )


@app.route("/race_links")
def race_links():
    return render_template("race_links.html", races_list=get_races_list())


@app.route("/", methods=["GET", "POST"])
@require_race_context
def index(race_context: RaceContext):
    if request.method == "POST":
        first_name = request.form.get("first_name") # Get first name
        last_name = request.form.get("last_name")  # Get last name
        try:
            patrol = request.form.get("patrol") # Get patrol
            if patrol in race_context.patrol_names:
                add_participant(first_name, last_name, patrol, race_context=race_context)  # Add participant
                save_data(context=race_context) # Save data after adding participant
                return redirect(url_for("index"))  # Redirect after successful addition
            else:
                flash(f"Invalid patrol: {patrol}", "error")
        except ValueError as e:
            logger.warning('index: failed to add participant %s %s patrol=%s: %s', first_name, last_name, patrol, e)
            flash(str(e), "error")

    sorted_participants = sorted(race_context.participants, key=lambda p: (p.patrol, p.car_number))

    return render_template("index.html", 
                           participants=sorted_participants, 
                           patrol_names=race_context.patrol_names)

@app.route("/edit_participant/<participant_id>", methods=["GET", "POST"])
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
    if not participant:
        logger.warning('participant_times: participant not found id=%s', participant_id)
        return respond_error("Participant not found", 404, "index")
    return render_template("participant_times.html", participant=participant, patrol_names=race_context.patrol_names)

@require_race_context
def add_participant(first_name,last_name, patrol, race_context: RaceContext):

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
def schedule_initial_races(race_context: RaceContext):

    race_number = 1
    for patrol in race_context.patrol_names:
        patrol_racers = [p for p in race_context.participants if p.patrol == patrol]

        if patrol_racers:
            race_groups = group_racers(patrol_racers)
            assign_paired_lanes(race_groups, Rounds.FIRST, race_number, race_context=race_context)
            race_number += len(race_groups)

            race_context.initial_races_completed[patrol] = False # Initialize to False at the start
            race_context.semi_final_races_completed[patrol] = False 
        else:
            race_context.initial_races_completed[patrol] = True # No races for this patrol
            race_context.semi_final_races_completed[patrol] = True # No races for this patrol


@require_race_context
def schedule_semi_final_races(patrol, race_context: RaceContext):

    if patrol == "Exhibition":
        return  # Don't schedule semi-finals for the exhibition patrol
    if not race_context.initial_races_completed[patrol]:
        return  # Don't schedule semi's if we havent finished the races

    top_racers,_ = get_top_racers(Rounds.FIRST, patrol, NUM_LANES, race_context=race_context)
    name_sorted_top_racers = sorted(top_racers, key=lambda racer: racer.car_name)

    if top_racers:
        race_groups = [name_sorted_top_racers]  # Create a single group of top racers
        assign_paired_lanes(race_groups, Rounds.SEMI, len(race_context.races) + 1, race_context=race_context) 

@require_race_context
def schedule_final_races(race_context: RaceContext):
    if not all(race_context.semi_final_races_completed.values()):
        return

    # 1. Get Top Racers from Semi-Finals:
    top_racers = []
    for patrol in race_context.patrol_names:
        if patrol != "Exhibition" and race_context.semi_final_races_completed.get(patrol, False):  # Check if semi-finals are complete
            top_racer,_ = get_top_racers(Rounds.SEMI, patrol, 1, race_context=race_context) # Only want to get top 1
            if top_racer:
                top_racers.append(top_racer[0])

    # 2. Create Final Race (using all lanes assignment):
    if top_racers: # Only if there are top racers
        race_groups = [top_racers]
        assign_all_lanes(race_groups, Rounds.FINAL, len(race_context.races) + 1, race_context=race_context)  # Use assign_all_lanes

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


@app.after_request
def log_response(response):
    try:
        status = response.status_code
    except Exception:
        return response
    if status >= 500:
        logger.error('Response %s %s returned %s', request.method, request.path, status)
    elif status >= 400:
        logger.warning('Response %s %s returned %s', request.method, request.path, status)
    return response

@app.route("/enter_times/<int:race_number>/<int:heat_number>", methods=["GET", "POST"])
@require_role('admin')
@require_race_context
def enter_times(race_number, heat_number, race_context:  RaceContext):
    role = session.get('role', Role.PUBLIC.name)
    race = next((r for r in race_context.races if r.race_number == race_number), None)
    if not race:
        logger.warning('enter_times: Race not found race_number=%s', race_number)
        return respond_error("Race not found", 404, "index")
    heat = next((h for h in race.heats if h.heat_number == heat_number), None)
    if not heat:
        logger.warning('enter_times: Heat not found race_number=%s heat_number=%s', race_number, heat_number)
        return respond_error("Heat not found", 404, "index")
    if Role[role] != Role.OWNER:
        flash("This Time Entry page is only intended for the Owner, please log in.", "error")
        return redirect(url_for("index"))
    else:

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
                return redirect(url_for("schedule", patrol=race.patrol, round=race.round.value)) # Add round parameter

    return render_template("enter_times.html", race=race, heat=heat, NUM_LANES=NUM_LANES)

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
            complete = check_races_complete(p, round)
            if complete is not None:
                if round == Rounds.FIRST:
                    race_context.initial_races_completed[p] = complete
                elif round == Rounds.SEMI:
                    race_context.semi_final_races_completed[p] = complete
                save_data(context=race_context)

@app.route("/schedule_initial")
@require_role('admin')
@require_race_context
def schedule_initial(race_context: RaceContext):
    logger.info('schedule_initial requested by uid=%s', session.get('fb_uid') or session.get('role'))
    clear_races(race_context=race_context)
    schedule_initial_races(race_context=race_context)
    save_data(context=race_context) # Saving data after scheduling initial round
    logger.info('schedule_initial completed; total races=%s', len(race_context.races))
    return redirect(url_for("schedule", round=Rounds.FIRST.value)) # Redirect to the main schedule page

@app.route("/schedule_semifinal/<patrol>")
@require_role('admin')
@require_race_context
def schedule_semifinal(patrol, race_context: RaceContext):
    logger.info('schedule_semifinal requested for patrol=%s by uid=%s', patrol, session.get('fb_uid') or session.get('role'))
    schedule_semi_final_races(patrol, race_context=race_context)
    save_data(context=race_context) # Saving data after scheduling semi-final round
    logger.info('schedule_semifinal completed for patrol=%s; total races=%s', patrol, len(race_context.races))
    return redirect(url_for("schedule", round=Rounds.SEMI.value)) # Add round parameter

@app.route("/schedule_final")
@require_role('admin')
@require_race_context
def schedule_final(race_context: RaceContext):
    logger.info('schedule_final requested by uid=%s', session.get('fb_uid') or session.get('role'))
    schedule_final_races(race_context=race_context)
    save_data(context=race_context) # Saving data after scheduling final round
    logger.info('schedule_final completed; total races=%s', len(race_context.races))
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

    check_round_complete(Rounds.FIRST, race_context=race_context)
    check_round_complete(Rounds.SEMI, race_context=race_context)

    top_racers, overall_racer_averages = get_top_racers(selected_round, selected_patrol, NUM_LANES, race_context=race_context)

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
                           NUM_LANES=NUM_LANES,
                           top_racers=list(enumerate(top_racers)),
                           overall_racer_averages=overall_racer_averages,
                           initial_races_completed=race_context.initial_races_completed,
                           semi_final_races_completed=race_context.semi_final_races_completed,
                           all_semi_final_races_completed=all_semi_final_races_completed,
                           semi_final_races_scheduled=semi_final_races_scheduled )

def calculate_race_averages(race):
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

def calculate_race_statistics(participant, race_context: RaceContext):
    p = participant
    if p.times:
        p.best_time = min(p.times)
        p.average_time = sum(p.times) / len(p.times)
        p.best_time_race_number = get_best_time_race_number(p, race_context=race_context)
    else:
        p.best_time = float('inf')
        p.average_time = 0
        p.best_time_race_number = None

def get_best_time_race_number(participant, race_context: RaceContext):
    best_time_race_number = None
    if participant.times:
        if participant.best_time > 0:
            for race in race_context.races:
                for heat in race.heats:
                    for lane,p in heat.lanes.items():
                        if p == participant and lane in heat.times and \
                           heat.times[lane] == participant.best_time:
                               return race.race_number
    return None

@app.route("/judge_design", methods=["GET", "POST"])
@require_race_context
def judge_design(race_context: RaceContext):

    race_id = session.get('current_race_id') or request.args.get('race') or request.form.get("race") or CURRENT_RACE_ID
    role = session.get('role', Role.PUBLIC.name)
    judge_name = session.get('judge_name', None)

    judges_data=race_context.judges
    judge_id = judges_data[judge_name]['id'] if judge_name else None

    racers = {}
    for patrol in race_context.patrol_names:
        racers[patrol] = [p for p in race_context.participants if p.patrol == patrol]

    good = True
    if not race_context.judging_active:
        good = False
        flash("Judging is currently closed.", "warning")

    if good and Role[role] != Role.JUDGE:
        good = False
        flash("This page is only for authorized Judges, please log in.", "danger")

    if good and request.method == "POST":

        for p in race_context.participants:
            rank = request.form.get(f"rank_{p.participant_id}", 0)
            race_context.designs[p.participant_id].scores[judge_id] = int(rank)
        save_data(race_id=race_id, context=race_context)
        return redirect(url_for("design_results"))

    return render_template("judge_design.html", racers=racers, patrol_names=race_context.patrol_names,
                            judge_id=judge_id, designs=race_context.designs, judge_name=judge_name, 
                            judging_active=race_context.judging_active)

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


def score_design(design):
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
        

    return render_template("design_results.html", 
                           patrol_names=race_context.patrol_names, 
                           sorted_scores_by_patrol=sorted_scores_by_patrol,
                           judging_active=race_context.judging_active)


@app.route("/display_results")
@require_race_context
def display_results(race_context: RaceContext):
    valid_participants = [p for p in race_context.participants if p.times]
    sorted_participants = sorted(valid_participants, key=lambda x: (x.average_time, x.best_time))
    for p in sorted_participants:
        calculate_race_statistics(p, race_context=race_context)
    return render_template("results.html", 
                           participants=sorted_participants, 
                           patrol_names=race_context.patrol_names)

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


@app.route("/download_racer_data/<participant_id>")
@require_race_context
def download_racer_data(participant_id, race_context: RaceContext):
    participant = next((p for p in race_context.participants if p.participant_id == participant_id), None)
    if not participant:
        logger.warning('download_racer_data: participant not found id=%s', participant_id)
        return "Participant not found", 404

    output = StringIO()
    writer = csv.writer(output)

    # Write header row
    writer.writerow(["Race #", "Time"])  # Customize headers as needed

    for i, time in enumerate(participant.times):
        race_number = next((race.race_number for race in race_context.races for heat in race.heats for lane, p in heat.lanes.items() if p == participant and i < len(heat.times) and heat.times[lane] == time), None)
        writer.writerow([race_number, time])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=racer_{participant_id}_data.csv"},
    )

@app.route("/upload_roster", methods=["GET", "POST"])
@require_role('admin')
def upload_roster():
    role = session.get('role', Role.PUBLIC.name)
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
        p = add_participant(row["First Name"], row["Last Name"], row["Patrol"], race_context=race_context)
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
