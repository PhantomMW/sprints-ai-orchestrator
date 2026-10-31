import pytest
from src.utils.auth import (
    SecurityConfigurationError,
    authenticate_reviewer,
    generate_caller_token,
    generate_reviewer_token,
    get_hitl_secret,
    hash_password,
    login_and_issue_token,
    verify_caller_token,
    verify_password,
    verify_reviewer_auth,
    PersistentNonceStore,
)


def test_auth_password_hashing_and_verification():
    """Verify PBKDF2-HMAC-SHA256 password hashing with constant-time verification."""
    password = "SuperSecretPassword2026!"
    hashed = hash_password(password)
    assert hashed.startswith("pbkdf2:sha256:100000$")
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword", hashed) is False
    assert verify_password("", hashed) is False


def test_auth_reviewer_login_success_and_failure():
    """Verify reviewer login credentials check and token issuance."""
    # Valid login
    ok, msg, token = login_and_issue_token("sarah_lead", "SprintsAdvisor2026!")
    assert ok is True
    assert token is not None
    assert "." in token

    # Invalid password
    bad_ok, bad_msg, bad_token = login_and_issue_token("sarah_lead", "IncorrectPass")
    assert bad_ok is False
    assert bad_token is None
    assert "Invalid" in bad_msg

    # Unregistered reviewer
    unreg_ok, unreg_msg, unreg_token = login_and_issue_token("unknown_user", "AnyPassword")
    assert unreg_ok is False
    assert unreg_token is None


def test_auth_persistent_nonce_store(tmp_path):
    """Verify that PersistentNonceStore detects replay attacks across restarts and instances."""
    db_file = tmp_path / "replay_test.sqlite3"
    store1 = PersistentNonceStore(db_file)
    store1.reset()

    # First consume succeeds
    assert store1.consume("nonce-test-1", "reviewer_a", 9999999999.0) is True

    # Immediate replay on same store fails
    assert store1.consume("nonce-test-1", "reviewer_a", 9999999999.0) is False

    # Replay on separate new store instance reading same SQLite DB fails
    store2 = PersistentNonceStore(db_file)
    assert store2.consume("nonce-test-1", "reviewer_a", 9999999999.0) is False


def test_auth_production_secret_guard(monkeypatch):
    """Verify that production environments strictly disallow missing or default secrets."""
    monkeypatch.setenv("SPRINTS_ENV", "production")
    monkeypatch.delenv("HITL_AUTH_SECRET", raising=False)

    with pytest.raises(SecurityConfigurationError) as excinfo:
        get_hitl_secret()
    assert "mandatory in production" in str(excinfo.value)

    # Insecure default secret in production also raises error
    from src.utils.auth import DEFAULT_HITL_SECRET
    monkeypatch.setenv("HITL_AUTH_SECRET", DEFAULT_HITL_SECRET)
    with pytest.raises(SecurityConfigurationError) as excinfo2:
        get_hitl_secret()
    assert "Insecure default" in str(excinfo2.value)


def test_auth_caller_token_for_database():
    """Verify that caller authorization tokens are correctly generated and verified."""
    token = generate_caller_token("advisor_agent", "advisor")
    valid, reason, payload = verify_caller_token(token)
    assert valid is True
    assert payload["caller_id"] == "advisor_agent"
    assert payload["role"] == "advisor"

    # Expired token
    expired_token = generate_caller_token("advisor_agent", "advisor", expires_in_seconds=-10)
    exp_valid, exp_reason, _ = verify_caller_token(expired_token)
    assert exp_valid is False
    assert "expired" in exp_reason.lower()

def test_auth_reviewer_login_rate_limiting():
    """Verify that excessive failed login attempts trigger lockout rate limiting."""
    from src.utils.auth import reset_login_rate_limits, MAX_FAILED_LOGIN_ATTEMPTS
    reset_login_rate_limits()

    # Attempt 5 incorrect logins
    for i in range(MAX_FAILED_LOGIN_ATTEMPTS):
        ok, msg, token = login_and_issue_token("sarah_lead", f"WrongPassword{i}")
        assert ok is False
        assert token is None

    # 6th attempt should be blocked by rate limiting lockout
    ok6, msg6, token6 = login_and_issue_token("sarah_lead", "WrongPassword6")
    assert ok6 is False
    assert "temporarily locked" in msg6.lower()

    # Reset for subsequent tests
    reset_login_rate_limits()


def test_nonce_concurrent_race_condition(tmp_path):
    """Verify that multiple concurrent threads attempting to consume the same nonce are handled atomically."""
    import concurrent.futures
    db_file = tmp_path / "concurrent_nonce_test.sqlite3"
    store = PersistentNonceStore(db_file)
    store.reset()

    target_nonce = "concurrent-race-test-nonce-123"
    results = []

    def try_consume(thread_idx):
        # Each thread uses a fresh store instance pointing to the same SQLite DB
        thread_store = PersistentNonceStore(db_file)
        return thread_store.consume(target_nonce, f"reviewer_{thread_idx}", 9999999999.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(try_consume, i) for i in range(15)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # Exactly 1 thread must succeed, and all 14 others must be rejected
    assert results.count(True) == 1
    assert results.count(False) == 14


def test_nonce_store_fail_closed_in_production(tmp_path, monkeypatch):
    """Verify that in production environments, PersistentNonceStore fails closed on DB errors."""
    monkeypatch.setenv("SPRINTS_ENV", "production")
    # Point to an impossible/invalid DB path that cannot be written
    invalid_path = tmp_path / "non_existent_dir_cannot_create" / "invalid.sqlite3"
    
    store = PersistentNonceStore(invalid_path)
    # Simulate DB failure during consume by corrupting connection
    store.db_path = tmp_path / "does_not_exist_sub" / "readonly.sqlite3"
    
    with pytest.raises(SecurityConfigurationError) as excinfo:
        store.consume("test-nonce-fail-closed", "reviewer_fail", 9999999999.0)
    assert "in-memory fallback is disabled in production" in str(excinfo.value).lower()
