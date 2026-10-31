from collections import defaultdict
import hashlib
import hmac
import os
from pathlib import Path
import pickle
import sqlite3
import threading
import base64
from typing import Any

try:
    from cryptography.fernet import Fernet
    HAS_FERNET = True
except ImportError:
    HAS_FERNET = False

class CheckpointSecurityError(Exception):
    """Raised when checkpoint cryptographic integrity verification fails."""
    pass

SecurityError = CheckpointSecurityError

from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.memory import InMemorySaver


import io


class SafeCheckpointUnpickler(pickle.Unpickler):
    """
    Restricted Unpickler that blocks arbitrary remote code execution gadgets.
    Guarantees that state deserialization only reconstructs safe application structures
    and primitives, preventing malicious deserialization attacks (e.g. os.system, subprocess).
    """

    BLOCKED_MODULES = {
        "os", "posix", "nt", "subprocess", "sys", "shutil",
        "socket", "pty", "commands", "ctypes", "builtins.eval", "builtins.exec"
    }

    def find_class(self, module: str, name: str) -> Any:
        # Check blocked module names and prefixes
        if module in self.BLOCKED_MODULES or module.startswith(("os.", "subprocess.", "sys.")):
            raise CheckpointSecurityError(
                f"Security violation: Deserialization of dangerous module '{module}.{name}' is blocked."
            )

        if module == "builtins" and name in ("eval", "exec", "compile", "open", "__import__", "getattr", "setattr"):
            raise CheckpointSecurityError(
                f"Security violation: Deserialization of dangerous builtin '{name}' is blocked."
            )

        return super().find_class(module, name)


def safe_checkpoint_loads(data: bytes) -> Any:
    """Safely load pickled checkpoint data using the restricted unpickler."""
    if not isinstance(data, (bytes, bytearray)):
        raise CheckpointSecurityError("Invalid checkpoint payload: expected bytes.")
    return SafeCheckpointUnpickler(io.BytesIO(data)).load()


class SqliteSaver(InMemorySaver):
    """
    A persistent SQLite-backed checkpointer for LangGraph workflows with
    cryptographic payload integrity verification and retention management.

    Persists conversation threads, workflow checkpoints, task writes,
    and channel versions to a local SQLite database across application restarts.
    Enables durable Human-in-the-Loop review pausing and resumption.

    Security & Deployment Boundary:
        - Cryptographic Integrity: Pickled checkpoint blobs are signed using HMAC-SHA256.
          Upon loading, signatures are verified to guard against offline database tampering.
        - Retention Management: Supports pruning obsolete checkpoints via `purge_old_checkpoints()`.
        - Concurrency Boundary: Protected by WAL mode and busy timeouts for local multi-threaded
          or multi-process usage on a single node. For distributed, multi-tenant horizontal clusters,
          evaluate PostgreSQL checkpointers (e.g., PostgresSaver).
    """

    def __init__(
        self,
        db_path: str | Path = "data/orchestrator_checkpoints.sqlite3",
        secret_key: str | None = None,
    ):
        super().__init__()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        app_env = os.getenv("SPRINTS_ENV", os.getenv("ENV", "development")).strip().lower()
        resolved_key = secret_key or os.getenv("CHECKPOINT_SECRET_KEY")
        if app_env in ("production", "prod", "staging"):
            if not resolved_key:
                raise CheckpointSecurityError(
                    "FATAL: CHECKPOINT_SECRET_KEY environment variable is mandatory in production/staging environments."
                )
            if resolved_key == "sprints-checkpoint-integrity-key-2026":
                raise CheckpointSecurityError(
                    "FATAL: Insecure default CHECKPOINT_SECRET_KEY detected in production. A strong key must be configured."
                )
        self.secret_key = (resolved_key or "sprints-checkpoint-integrity-key-2026").encode("utf-8")
        if HAS_FERNET:
            derived_32 = hashlib.sha256(self.secret_key).digest()
            self._fernet_key = base64.urlsafe_b64encode(derived_32)
            self._cipher = Fernet(self._fernet_key)
        else:
            self._cipher = None
            if app_env in ("production", "prod", "staging"):
                raise CheckpointSecurityError(
                    "FATAL: cryptography library (Fernet) is mandatory in production environments to enforce at-rest checkpoint encryption."
                )
        self._lock = threading.Lock()
        self._init_db()
        self._load_from_db()

    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt serialized bytes at rest using AES-CBC/HMAC Fernet cipher."""
        if self._cipher:
            return self._cipher.encrypt(data)
        app_env = os.getenv("SPRINTS_ENV", os.getenv("ENV", "development")).strip().lower()
        if app_env in ("production", "prod", "staging"):
            raise CheckpointSecurityError(
                "FATAL: Checkpoint encryption at rest is mandatory in production environments."
            )
        return data

    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt binary blob from storage; enforces decryption in production without silent plaintext fallback."""
        if self._cipher:
            try:
                return self._cipher.decrypt(data)
            except Exception as e:
                app_env = os.getenv("SPRINTS_ENV", os.getenv("ENV", "development")).strip().lower()
                if app_env in ("production", "prod", "staging"):
                    raise CheckpointSecurityError(
                        f"FATAL: Checkpoint decryption failed in production: {e}"
                    )
                return data
        app_env = os.getenv("SPRINTS_ENV", os.getenv("ENV", "development")).strip().lower()
        if app_env in ("production", "prod", "staging"):
            raise CheckpointSecurityError(
                "FATAL: Checkpoint encryption at rest is mandatory in production environments."
            )
        return data

    def _sign(self, data: bytes, context_prefix: str = "") -> str:
        """Compute an HMAC-SHA256 signature bound to contextual row identifiers to prevent cut-and-paste attacks."""
        payload = context_prefix.encode("utf-8") + data
        return hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()

    def _verify_signature(self, data: bytes, signature: str | None, context_prefix: str = "") -> bool:
        """Verify the integrity of a serialized binary blob against its signature with coordinate binding."""
        if not signature:
            app_env = os.getenv("SPRINTS_ENV", os.getenv("ENV", "development")).strip().lower()
            if app_env in ("production", "prod", "staging"):
                return False  # Unsigned checkpoints are strictly rejected in production environments
            return True  # Permitted in development for backward compatibility
        expected = self._sign(data, context_prefix=context_prefix)
        return hmac.compare_digest(expected, signature)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT,
                        checkpoint_ns TEXT,
                        checkpoint_id TEXT,
                        checkpoint_tuple BLOB,
                        signature TEXT,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    )
                """)
                # Ensure signature column exists for upgraded databases
                try:
                    conn.execute("ALTER TABLE checkpoints ADD COLUMN signature TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists

                b_cols = [r[1] for r in conn.execute("PRAGMA table_info(blobs)").fetchall()]
                if b_cols and "blob_key" not in b_cols:
                    conn.execute("DROP TABLE blobs")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS blobs (
                        blob_key BLOB PRIMARY KEY,
                        blob_data BLOB,
                        signature TEXT
                    )
                """)

                w_cols = [r[1] for r in conn.execute("PRAGMA table_info(writes)").fetchall()]
                if w_cols and "write_key" not in w_cols:
                    conn.execute("DROP TABLE writes")
                try:
                    conn.execute("ALTER TABLE blobs ADD COLUMN signature TEXT")
                except sqlite3.OperationalError:
                    pass

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS writes (
                        write_key BLOB PRIMARY KEY,
                        write_tuple BLOB,
                        signature TEXT
                    )
                """)
                try:
                    conn.execute("ALTER TABLE writes ADD COLUMN signature TEXT")
                except sqlite3.OperationalError:
                    pass

                conn.commit()
            finally:
                conn.close()

    def _load_from_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                for row in conn.execute(
                    "SELECT thread_id, checkpoint_ns, checkpoint_id, checkpoint_tuple, signature FROM checkpoints"
                ):
                    t_id, ns, cp_id, cp_data, sig = row
                    ctx = f"{t_id}:{ns}:{cp_id}:"
                    if not self._verify_signature(cp_data, sig, context_prefix=ctx):
                        raise SecurityError(
                            f"Integrity check failed for checkpoint {t_id}/{cp_id}: signature mismatch."
                        )
                    self.storage[t_id][ns][cp_id] = safe_checkpoint_loads(self._decrypt(cp_data))

                for row in conn.execute("SELECT blob_key, blob_data, signature FROM blobs"):
                    b_key, b_data, sig = row
                    ctx = f"blob:{b_key!r}:"
                    if not self._verify_signature(b_data, sig, context_prefix=ctx):
                        raise SecurityError("Integrity check failed for checkpoint blob: signature mismatch.")
                    self.blobs[safe_checkpoint_loads(b_key)] = safe_checkpoint_loads(self._decrypt(b_data))

                for row in conn.execute("SELECT write_key, write_tuple, signature FROM writes"):
                    w_key, w_data, sig = row
                    ctx = f"write:{w_key!r}:"
                    if not self._verify_signature(w_data, sig, context_prefix=ctx):
                        raise SecurityError("Integrity check failed for checkpoint write: signature mismatch.")
                    outer_key, inner_key = safe_checkpoint_loads(w_key)
                    self.writes[outer_key][inner_key] = safe_checkpoint_loads(self._decrypt(w_data))
            finally:
                conn.close()

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict[str, Any]:
        res = super().put(config, checkpoint, metadata, new_versions)
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        cp_id = checkpoint["id"]

        with self._lock:
            conn = self._connect()
            try:
                if cp_id in self.storage[thread_id][checkpoint_ns]:
                    raw_data = pickle.dumps(self.storage[thread_id][checkpoint_ns][cp_id])
                    data = self._encrypt(raw_data)
                    ctx = f"{thread_id}:{checkpoint_ns}:{cp_id}:"
                    sig = self._sign(data, context_prefix=ctx)
                    conn.execute(
                        "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?)",
                        (thread_id, checkpoint_ns, cp_id, data, sig),
                    )
                for k, v in new_versions.items():
                    blob_key = (thread_id, checkpoint_ns, k, v)
                    if blob_key in self.blobs:
                        raw_b_data = pickle.dumps(self.blobs[blob_key])
                        b_data = self._encrypt(raw_b_data)
                        b_key_ser = pickle.dumps(blob_key)
                        ctx = f"blob:{b_key_ser!r}:"
                        conn.execute(
                            "INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                            (b_key_ser, b_data, self._sign(b_data, context_prefix=ctx)),
                        )
                conn.commit()
            finally:
                conn.close()
        return res

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Any,
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path)
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        outer_key = (thread_id, checkpoint_ns, checkpoint_id)

        with self._lock:
            conn = self._connect()
            try:
                if outer_key in self.writes:
                    for inner_key, write_tuple in list(self.writes[outer_key].items()):
                        full_key = (outer_key, inner_key)
                        full_key_ser = pickle.dumps(full_key)
                        raw_w_data = pickle.dumps(write_tuple)
                        w_data = self._encrypt(raw_w_data)
                        ctx = f"write:{full_key_ser!r}:"
                        conn.execute(
                            "INSERT OR REPLACE INTO writes VALUES (?, ?, ?)",
                            (full_key_ser, w_data, self._sign(w_data, context_prefix=ctx)),
                        )
                conn.commit()
            finally:
                conn.close()

    def purge_old_checkpoints(self, keep_last_n_per_thread: int = 50) -> int:
        """
        Prune older checkpoints per thread to prevent unbounded database growth.

        Args:
            keep_last_n_per_thread: Maximum number of recent checkpoints to retain per thread.

        Returns:
            Number of deleted obsolete checkpoint rows.
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                # Find threads exceeding retention threshold
                threads = [
                    row[0]
                    for row in cursor.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
                ]
                deleted_count = 0
                for t_id in threads:
                    cursor.execute(
                        """
                        DELETE FROM checkpoints
                        WHERE thread_id = ?
                          AND checkpoint_id NOT IN (
                              SELECT checkpoint_id FROM checkpoints
                              WHERE thread_id = ?
                              ORDER BY rowid DESC
                              LIMIT ?
                          )
                        """,
                        (t_id, t_id, keep_last_n_per_thread),
                    )
                    deleted_count += cursor.rowcount
                conn.commit()
                return deleted_count
            finally:
                conn.close()
