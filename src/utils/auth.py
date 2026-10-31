import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any

# Default secret for development/demo; production must inject via environment
DEFAULT_HITL_SECRET = "sprints-hitl-secure-governance-secret-key-2026"

# Allowed roles with authority to approve tier-1 employer recommendations
AUTHORIZED_APPROVER_ROLES = {
    "placement_director",
    "lead_career_advisor",
    "placement_approver",
    "institutional_admin",
}

# In-memory replay cache: tracks used token nonces to prevent replay attacks
_CONSUMED_NONCES: set[str] = set()


def get_hitl_secret() -> str:
    """Retrieve the cryptographic signing key for HITL session verification."""
    return os.getenv("HITL_AUTH_SECRET", DEFAULT_HITL_SECRET)


def get_authorized_reviewers() -> set[str]:
    """
    Retrieve the set of authorized reviewer identities.
    Can be overridden via AUTHORIZED_HITL_REVIEWERS environment variable.
    """
    env_reviewers = os.getenv("AUTHORIZED_HITL_REVIEWERS")
    if env_reviewers:
        return {r.strip().casefold() for r in env_reviewers.split(",") if r.strip()}
    return {"sarah_lead", "placement_director", "director of placement", "career_board_admin", "ahmed_lead", "career_admin", "placement_lead", "institutional_reviewer", "test reviewer"}


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
) -> tuple[bool, str, dict[str, Any]]:
    """
    Cryptographically verify the identity, role, and token freshness of a HITL reviewer.

    Security Controls:
        - Cryptographic Signature: Validates HMAC-SHA256 signature against server secret.
        - Expiration: Rejects expired session tokens.
        - Anti-Replay: Verifies token nonce has not been consumed previously.
        - Identity Allowlist: Ensures reviewer_id is recognized in get_authorized_reviewers().
        - RBAC Privilege: Ensures role has placement approval clearance.
        - Audit Trail: Returns structured audit details (timestamps, thread_id, decision_id).

    Args:
        decision: Resume decision dictionary submitted to the workflow.
        thread_id: The active workflow conversation thread ID.
        secret: Optional custom secret (defaults to environment or default secret).

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
    key = (secret or get_hitl_secret()).encode("utf-8")
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
    expires_at = payload.get("expires_at", 0)
    nonce = payload.get("nonce", "")

    # Expiration check
    if time.time() > expires_at:
        return (
            False,
            f"Session token expired at timestamp {expires_at}",
            {"error_code": "TOKEN_EXPIRED", "thread_id": thread_id, "reviewer": token_reviewer},
        )

    # Anti-replay check
    if nonce in _CONSUMED_NONCES:
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

    # Consume nonce to prevent replay
    _CONSUMED_NONCES.add(nonce)

    audit_record = {
        "decision_id": uuid.uuid4().hex,
        "reviewer_id": token_reviewer,
        "role": token_role,
        "thread_id": thread_id,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auth_method": "HMAC-SHA256-Signed-Session",
        "signature_verified": True,
    }

    return True, "Authenticated and authorized successfully", audit_record
