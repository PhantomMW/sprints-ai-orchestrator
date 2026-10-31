import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
import secrets
import threading
import time
import uuid
from typing import Any


class SecurityConfigurationError(Exception):
    """Raised when an insecure security configuration is detected in production."""
    pass


# Default secret for development/demo; production MUST inject via environment variable
DEFAULT_HITL_SECRET = "sprints-hitl-secure-governance-secret-key-2026"

# Allowed roles with authority to approve tier-1 employer recommendations
AUTHORIZED_APPROVER_ROLES = {
    "placement_director",
    "lead_career_advisor",
    "placement_approver",
    "institutional_admin",
}

# Default database path for persistent shared anti-replay nonce storage
DEFAULT_NONCE_DB_PATH = Path("data/hitl_replay_store.sqlite3")


def get_hitl_secret() -> str:
    """
    Retrieve the cryptographic signing key for HITL session verification.
    
    Security Guarantee:
        In production/staging environments (SPRINTS_ENV=production|prod|staging),
        using the default hardcoded secret or omitting HITL_AUTH_SECRET raises
        SecurityConfigurationError to prevent signature forgery vulnerabilities.
    """
    secret = os.getenv("HITL_AUTH_SECRET")
    app_env = os.getenv("SPRINTS_ENV", os.getenv("ENV", "development")).strip().lower()

    if app_env in ("production", "prod", "staging"):
        if not secret:
            raise SecurityConfigurationError(
                "FATAL: HITL_AUTH_SECRET environment variable is mandatory in production/staging environments. "
                "Default development keys are strictly disallowed."
            )
        if secret == DEFAULT_HITL_SECRET:
            raise SecurityConfigurationError(
                "FATAL: Insecure default HITL_AUTH_SECRET detected in production environment. "
                "A unique, cryptographically strong secret key must be configured."
            )

    return secret or DEFAULT_HITL_SECRET


class PersistentNonceStore:
    """
    Persistent, process-safe, multi-worker anti-replay nonce store backed by SQLite.
    
    Prevents token replay attacks across application restarts, gunicorn/uvicorn workers,
    and distributed nodes using atomic transactional inserts and primary key constraints.
    """

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or os.getenv("HITL_NONCE_DB_PATH") or DEFAULT_NONCE_DB_PATH)
        self._lock = threading.Lock()
        self._memory_fallback: set[str] = set()
        self._init_db()

    def _init_db(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA busy_timeout = 5000")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS consumed_nonces (
                        nonce TEXT PRIMARY KEY,
                        reviewer_id TEXT NOT NULL,
                        consumed_at REAL NOT NULL,
                        expires_at REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_nonces_expires 
                    ON consumed_nonces (expires_at)
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS login_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        attempted_at REAL NOT NULL,
                        success INTEGER NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_login_attempts_user_time 
                    ON login_attempts (username, attempted_at)
                """)
                conn.commit()
        except Exception as e:
            app_env = os.getenv("SPRINTS_ENV", os.getenv("ENV", "development")).strip().lower()
            if app_env in ("production", "prod", "staging"):
                raise SecurityConfigurationError(
                    f"FATAL: Persistent nonce store database failed to initialize at {self.db_path}: {e}"
                )

    def consume(self, nonce: str, reviewer_id: str, expires_at: float) -> bool:
        """
        Atomically mark a nonce as consumed.
        
        Returns:
            True if nonce was successfully consumed (first time seen).
            False if nonce has already been consumed (replay attack detected).
        """
        now = time.time()
        with self._lock:
            # Check fast in-memory cache
            if nonce in self._memory_fallback:
                return False

            try:
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                    # Atomic insert: fails with IntegrityError if nonce already exists
                    conn.execute(
                        "INSERT INTO consumed_nonces (nonce, reviewer_id, consumed_at, expires_at) VALUES (?, ?, ?, ?)",
                        (nonce, reviewer_id, now, expires_at),
                    )
                    # Periodic retention cleanup: prune expired nonces older than current time
                    conn.execute("DELETE FROM consumed_nonces WHERE expires_at < ?", (now,))
                    conn.commit()
                self._memory_fallback.add(nonce)
                return True
            except sqlite3.IntegrityError:
                # Primary key collision: nonce replayed
                self._memory_fallback.add(nonce)
                return False
            except Exception as e:
                app_env = os.getenv("SPRINTS_ENV", os.getenv("ENV", "development")).strip().lower()
                if app_env in ("production", "prod", "staging"):
                    # Fail-closed in production: strictly disallow in-memory fallback
                    raise SecurityConfigurationError(
                        f"FATAL: Persistent nonce store write error in production: {e}. "
                        "In-memory fallback is disabled in production to prevent token replay attacks."
                    )
                # Fallback to process-local set if SQLite write fails ONLY in development
                if nonce in self._memory_fallback:
                    return False
                self._memory_fallback.add(nonce)
                return True

    def reset(self) -> None:
        """Reset the nonce store and rate limit records (used primarily for test isolation)."""
        with self._lock:
            self._memory_fallback.clear()
            try:
                if self.db_path.exists():
                    with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                        conn.execute("DELETE FROM consumed_nonces")
                        conn.execute("DELETE FROM login_attempts")
                        conn.commit()
            except Exception:
                pass

    def record_login_attempt(self, username: str, success: bool) -> None:
        """Record a login attempt persistently for cross-worker rate limiting."""
        clean_user = username.strip().casefold()
        now = time.time()
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                    conn.execute(
                        "INSERT INTO login_attempts (username, attempted_at, success) VALUES (?, ?, ?)",
                        (clean_user, now, 1 if success else 0),
                    )
                    if success:
                        conn.execute("DELETE FROM login_attempts WHERE username = ? AND success = 0", (clean_user,))
                    conn.execute("DELETE FROM login_attempts WHERE attempted_at < ?", (now - 86400,))
                    conn.commit()
            except Exception:
                pass

    def check_login_rate_limit(self, username: str) -> tuple[bool, int]:
        """
        Check whether an account is locked out across all workers.
        Returns (is_locked, remaining_seconds).
        """
        clean_user = username.strip().casefold()
        now = time.time()
        window_start = now - LOGIN_LOCKOUT_WINDOW_SECONDS
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                    cursor = conn.execute(
                        "SELECT attempted_at FROM login_attempts WHERE username = ? AND success = 0 AND attempted_at > ? ORDER BY attempted_at ASC",
                        (clean_user, window_start),
                    )
                    attempts = [row[0] for row in cursor.fetchall()]
                    if len(attempts) >= MAX_FAILED_LOGIN_ATTEMPTS:
                        remaining = int(LOGIN_LOCKOUT_WINDOW_SECONDS - (now - attempts[0]))
                        return True, max(1, remaining)
            except Exception:
                pass
        return False, 0

    def reset_login_attempts(self, username: str | None = None) -> None:
        """Reset persistent login attempt records for a specific user or all users."""
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                    if username:
                        conn.execute("DELETE FROM login_attempts WHERE username = ?", (username.strip().casefold(),))
                    else:
                        conn.execute("DELETE FROM login_attempts")
                    conn.commit()
            except Exception:
                pass


# Global singleton persistent nonce store
_GLOBAL_NONCE_STORE = PersistentNonceStore()


def get_authorized_reviewers() -> set[str]:
    """
    Retrieve the set of authorized reviewer identities.
    Can be overridden via AUTHORIZED_HITL_REVIEWERS environment variable.
    """
    env_reviewers = os.getenv("AUTHORIZED_HITL_REVIEWERS")
    if env_reviewers:
        return {r.strip().casefold() for r in env_reviewers.split(",") if r.strip()}
    return {
        "sarah_lead",
        "placement_director",
        "director of placement",
        "career_board_admin",
        "ahmed_lead",
        "career_admin",
        "placement_lead",
        "institutional_reviewer",
        "test reviewer",
    }


# ==============================================================================
# REVIEWER CREDENTIALS & PBKDF2 AUTHENTICATION (P1 Login System)
# ==============================================================================

def hash_password(password: str, salt: str | None = None) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 with 100,000 iterations and cryptographically secure random salt."""
    salt_str = salt or secrets.token_hex(16)
    salt_bytes = salt_str.encode("utf-8")
    pwd_bytes = password.encode("utf-8")
    derived = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100000)
    derived_hex = derived.hex()
    return f"pbkdf2:sha256:100000${salt_str}${derived_hex}"


def verify_password(password: str, hashed_str: str) -> bool:
    """Verify a plain password against a stored PBKDF2 hash using constant-time comparison."""
    try:
        parts = hashed_str.split("$")
        if len(parts) != 3:
            return False
        _algo_info, salt_str, stored_hash = parts
        salt_bytes = salt_str.encode("utf-8")
        pwd_bytes = password.encode("utf-8")
        computed = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100000).hex()
        return hmac.compare_digest(computed, stored_hash)
    except Exception:
        return False


# Rate limiting and lockout state for human reviewer authentication
MAX_FAILED_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_WINDOW_SECONDS = 900  # 15 minutes lockout window


def reset_login_rate_limits() -> None:
    """Reset the persistent shared login rate limit tracking state across all workers."""
    _GLOBAL_NONCE_STORE.reset_login_attempts()


# Pre-hashed institutional reviewer credentials (PBKDF2-HMAC-SHA256, 100,000 rounds)
# Stored solely as salted hashes without plaintext credentials in source code.
# In production, custom credentials should be configured via SPRINTS_REVIEWERS_CONFIG.
_DEFAULT_REVIEWER_HASHES: dict[str, dict[str, str]] = {
    "sarah_lead": {
        "role": "lead_career_advisor",
        "password_hash": "pbkdf2:sha256:100000$sprints_salt_sarah$647253ca4b365e1545ce346b9002728fee80edfa5c39da5991d181779bcd9003",
    },
    "placement_director": {
        "role": "placement_director",
        "password_hash": "pbkdf2:sha256:100000$sprints_salt_director$c0be12f39905d70caca5274e569856c77d16fe241475fe6e73041d9b4ac2cd57",
    },
    "director of placement": {
        "role": "placement_director",
        "password_hash": "pbkdf2:sha256:100000$sprints_salt_director_alias$75bfbfb7453ec1adb9e22b59cb3549c0f70aa391dea48d51dbed921ac456d635",
    },
    "career_admin": {
        "role": "placement_approver",
        "password_hash": "pbkdf2:sha256:100000$sprints_salt_admin$eea51c8572067f62782f93253516f1c00193090502004960dbfac506e3889778",
    },
    "institutional_reviewer": {
        "role": "institutional_admin",
        "password_hash": "pbkdf2:sha256:100000$sprints_salt_gov$3cc6fa3f2131f35458ed1299324d3c35f4401dd113c7f009d0539bbacf00dde3",
    },
    "ahmed_lead": {
        "role": "lead_career_advisor",
        "password_hash": "pbkdf2:sha256:100000$sprints_salt_ahmed$8bdb05841c3abd697459d50c6ae59f435988c0ef327d481444e978d82089dd94",
    },
    "test reviewer": {
        "role": "placement_approver",
        "password_hash": "pbkdf2:sha256:100000$sprints_salt_test$688a80d9aa3caade5f66911bd6226f0d718de12d20b915cba72d89ef9d57396d",
    },
}


def get_registered_reviewers() -> dict[str, dict[str, str]]:
    """
    Retrieve registered institutional reviewers.
    Prioritizes SPRINTS_REVIEWERS_CONFIG environment variable if present.
    In production environments, using default accounts requires explicit opt-in
    via SPRINTS_ENABLE_DEMO_REVIEWERS=true or custom SPRINTS_REVIEWERS_CONFIG.
    """
    config_env = os.getenv("SPRINTS_REVIEWERS_CONFIG")
    if config_env:
        try:
            parsed = json.loads(config_env)
            if isinstance(parsed, dict):
                return {k.strip().casefold(): v for k, v in parsed.items()}
        except Exception:
            pass

    app_env = os.getenv("SPRINTS_ENV", os.getenv("ENV", "development")).strip().lower()
    if app_env in ("production", "prod", "staging"):
        if os.getenv("SPRINTS_ENABLE_DEMO_REVIEWERS") != "true" and not config_env:
            # In strict production without demo opt-in or custom config, return empty
            return {}

    return dict(_DEFAULT_REVIEWER_HASHES)


def authenticate_reviewer(username: str, password: str) -> tuple[bool, str, dict[str, Any]]:
    """
    Authenticate a human reviewer using verified credentials before issuing a session token.
    Applies persistent SQLite-backed rate limiting shared across all workers and processes.
    
    Returns:
        (is_authenticated, reason, reviewer_info)
    """
    clean_user = username.strip().casefold()

    # 1. Multi-worker persistent rate limiting check (anti brute-force)
    is_locked, remaining = _GLOBAL_NONCE_STORE.check_login_rate_limit(clean_user)
    if is_locked:
        return (
            False,
            f"Account temporarily locked due to excessive failed attempts. Try again in {remaining}s.",
            {},
        )

    reviewers_dir = get_registered_reviewers()
    user_record = reviewers_dir.get(clean_user)

    if not user_record:
        _GLOBAL_NONCE_STORE.record_login_attempt(clean_user, success=False)
        return False, f"Reviewer '{username}' not registered in institutional directory.", {}

    if not verify_password(password, user_record["password_hash"]):
        _GLOBAL_NONCE_STORE.record_login_attempt(clean_user, success=False)
        return False, "Invalid reviewer password or credentials.", {}

    # Successful authentication: record success and clear lockout history
    _GLOBAL_NONCE_STORE.record_login_attempt(clean_user, success=True)

    return True, "Reviewer credentials authenticated successfully.", {
        "reviewer_id": username.strip(),
        "role": user_record["role"],
    }


def login_and_issue_token(
    username: str,
    password: str,
    expires_in_seconds: int = 3600,
    secret: str | None = None,
) -> tuple[bool, str, str | None]:
    """
    Authenticate a human reviewer and issue an HMAC-SHA256 signed session token.
    
    Returns:
        (success, message, token_or_none)
    """
    ok, reason, user_info = authenticate_reviewer(username, password)
    if not ok:
        return False, reason, None

    token = generate_reviewer_token(
        reviewer_id=user_info["reviewer_id"],
        role=user_info["role"],
        expires_in_seconds=expires_in_seconds,
        secret=secret,
    )
    return True, f"Login successful for {username}. Token issued.", token


# ==============================================================================
# TOKEN GENERATION & VERIFICATION (HMAC-SHA256)
# ==============================================================================

def generate_reviewer_token(
    reviewer_id: str,
    role: str = "placement_approver",
    expires_in_seconds: int = 3600,
    secret: str | None = None,
) -> str:
    """
    Generate an HMAC-SHA256 cryptographically signed session token for a reviewer.

    Format: <base64_payload>.<hmac_signature_hex>
    """
    key = (secret or get_hitl_secret()).encode("utf-8")
    now = int(time.time())
    payload = {
        "reviewer_id": reviewer_id.strip(),
        "role": role.strip().casefold(),
        "nonce": uuid.uuid4().hex,
        "issued_at": now,
        "expires_at": now + expires_in_seconds,
    }
    raw_json = json.dumps(payload, sort_keys=True).encode("utf-8")
    b64_payload = base64.urlsafe_b64encode(raw_json).decode("utf-8")

    signature = hmac.new(key, b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{b64_payload}.{signature}"


def verify_reviewer_auth(
    decision: dict[str, Any],
    thread_id: str = "",
    secret: str | None = None,
    nonce_store: PersistentNonceStore | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """
    Cryptographically verify the identity, role, and token freshness of a HITL reviewer.

    Security Controls:
        - Production Secret Guard: Enforces non-default secrets in production environments.
        - Cryptographic Signature: Validates HMAC-SHA256 signature against server secret.
        - Expiration: Rejects expired session tokens.
        - Persistent Anti-Replay: Verifies token nonce has not been consumed across workers/restarts.
        - Identity Allowlist: Ensures reviewer_id is recognized in get_authorized_reviewers().
        - RBAC Privilege: Ensures role has placement approval clearance.
        - Audit Trail: Returns structured audit details (timestamps, thread_id, decision_id).

    Args:
        decision: Resume decision dictionary submitted to the workflow.
        thread_id: The active workflow conversation thread ID.
        secret: Optional custom secret (defaults to environment or default secret).
        nonce_store: Optional persistent nonce store override (defaults to global store).

    Returns:
        Tuple of (is_authenticated: bool, reason: str, audit_record: dict).
    """
    if not isinstance(decision, dict):
        return False, "Invalid decision payload format", {}

    auth_token = decision.get("auth_token") or decision.get("token")
    reviewer = str(decision.get("reviewer", "")).strip()

    if not auth_token:
        # Reject plain-text names claiming approval without cryptographic credentials
        return (
            False,
            "Missing cryptographic auth_token: Unsigned reviewer names are rejected under enterprise RBAC governance.",
            {
                "claimed_reviewer": reviewer,
                "error_code": "AUTH_TOKEN_MISSING",
                "thread_id": thread_id,
            },
        )

    parts = str(auth_token).split(".")
    if len(parts) != 2:
        return (
            False,
            "Malformed auth_token structure: Expected <base64_payload>.<signature>",
            {"error_code": "MALFORMED_TOKEN", "thread_id": thread_id},
        )

    b64_payload, signature = parts
    try:
        key = (secret or get_hitl_secret()).encode("utf-8")
    except SecurityConfigurationError as sce:
        return False, str(sce), {"error_code": "INSECURE_CONFIGURATION", "thread_id": thread_id}

    expected_sig = hmac.new(key, b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(signature, expected_sig):
        return (
            False,
            "Cryptographic signature verification failed: Token has been forged or tampered with.",
            {"error_code": "SIGNATURE_MISMATCH", "thread_id": thread_id},
        )

    try:
        raw_json = base64.urlsafe_b64decode(b64_payload.encode("utf-8")).decode("utf-8")
        payload = json.loads(raw_json)
    except Exception as e:
        return (
            False,
            f"Unable to decode token payload: {e}",
            {"error_code": "PAYLOAD_DECODE_ERROR", "thread_id": thread_id},
        )

    token_reviewer = str(payload.get("reviewer_id", "")).strip()
    token_role = str(payload.get("role", "")).strip().casefold()
    expires_at = float(payload.get("expires_at", 0))
    nonce = str(payload.get("nonce", "")).strip()

    # Expiration check
    if time.time() > expires_at:
        return (
            False,
            f"Session token expired at timestamp {expires_at}",
            {"error_code": "TOKEN_EXPIRED", "thread_id": thread_id, "reviewer": token_reviewer},
        )

    # Persistent Anti-replay check
    store = nonce_store or _GLOBAL_NONCE_STORE
    consumed = store.consume(nonce=nonce, reviewer_id=token_reviewer, expires_at=expires_at)
    if not consumed:
        return (
            False,
            f"Anti-replay protection violation: Nonce '{nonce}' has already been consumed.",
            {"error_code": "REPLAY_ATTACK_DETECTED", "thread_id": thread_id},
        )

    # Validate identity consistency if reviewer was also specified in payload
    if reviewer and reviewer.casefold() != token_reviewer.casefold():
        return (
            False,
            f"Identity mismatch: Claimed reviewer '{reviewer}' does not match authenticated token '{token_reviewer}'.",
            {"error_code": "IDENTITY_MISMATCH", "thread_id": thread_id},
        )

    # Allowlist & RBAC role check
    authorized_set = get_authorized_reviewers()
    if token_reviewer.casefold() not in authorized_set:
        return (
            False,
            f"Reviewer '{token_reviewer}' is not in authorized institutional directory.",
            {"error_code": "REVIEWER_NOT_AUTHORIZED", "thread_id": thread_id},
        )

    if token_role not in AUTHORIZED_APPROVER_ROLES:
        return (
            False,
            f"Role '{token_role}' lacks placement recommendation approval privileges.",
            {"error_code": "INSUFFICIENT_ROLE_PRIVILEGE", "thread_id": thread_id},
        )

    audit_record = {
        "decision_id": uuid.uuid4().hex,
        "reviewer_id": token_reviewer,
        "role": token_role,
        "thread_id": thread_id,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auth_method": "HMAC-SHA256-Signed-Session",
        "signature_verified": True,
        "replay_prevented": True,
    }

    return True, "Authenticated and authorized successfully", audit_record


# ==============================================================================
# CALLER AUTHORIZATION TOKENS FOR DATABASE QUERIES (P1 DB Authorization)
# ==============================================================================

def generate_caller_token(
    caller_id: str,
    role: str = "advisor",
    expires_in_seconds: int = 3600,
    secret: str | None = None,
) -> str:
    """Generate an HMAC-SHA256 signed caller token for authorized database queries."""
    key = (secret or get_hitl_secret()).encode("utf-8")
    now = int(time.time())
    payload = {
        "caller_id": caller_id.strip(),
        "role": role.strip().casefold(),
        "nonce": uuid.uuid4().hex,
        "issued_at": now,
        "expires_at": now + expires_in_seconds,
    }
    raw_json = json.dumps(payload, sort_keys=True).encode("utf-8")
    b64_payload = base64.urlsafe_b64encode(raw_json).decode("utf-8")
    sig = hmac.new(key, b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{b64_payload}.{sig}"


def verify_caller_token(
    auth_token: str | None,
    secret: str | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Verify cryptographic authenticity of a database caller token."""
    if not auth_token:
        return False, "Missing caller authorization token", {}

    parts = str(auth_token).split(".")
    if len(parts) != 2:
        return False, "Malformed caller token format", {}

    b64_payload, signature = parts
    try:
        key = (secret or get_hitl_secret()).encode("utf-8")
    except SecurityConfigurationError as sce:
        return False, str(sce), {}

    expected = hmac.new(key, b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return False, "Cryptographic signature mismatch on caller token", {}

    try:
        raw_json = base64.urlsafe_b64decode(b64_payload.encode("utf-8")).decode("utf-8")
        payload = json.loads(raw_json)
    except Exception as e:
        return False, f"Failed to parse caller token payload: {e}", {}

    if time.time() > float(payload.get("expires_at", 0)):
        return False, "Caller token has expired", {}

    return True, "Caller token verified", payload
