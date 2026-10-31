from collections import defaultdict
from pathlib import Path
import pickle
import sqlite3
import threading
from typing import Any

from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.memory import InMemorySaver


class SqliteSaver(InMemorySaver):
    """
    A persistent SQLite-backed checkpointer for LangGraph workflows.

    Persists conversation threads, workflow checkpoints, task writes,
    and channel versions to a local SQLite database across application restarts.
    Enables durable Human-in-the-Loop review pausing and resumption.

    Security & Deployment Boundary:
        - Checkpoints are serialized using standard pickle within a local trusted database.
        - Database files must not be loaded from untrusted external sources.
        - For distributed, multi-tenant production clusters, evaluate PostgreSQL checkpointers
          or signed envelope serialization (e.g. HMAC / JSON).
    """

    def __init__(self, db_path: str | Path = "data/orchestrator_checkpoints.sqlite3"):
        super().__init__()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        self._load_from_db()

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
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS blobs (
                        blob_key BLOB PRIMARY KEY,
                        blob_data BLOB
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS writes (
                        write_key BLOB PRIMARY KEY,
                        write_tuple BLOB
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    def _load_from_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                for row in conn.execute("SELECT thread_id, checkpoint_ns, checkpoint_id, checkpoint_tuple FROM checkpoints"):
                    t_id, ns, cp_id, cp_data = row
                    self.storage[t_id][ns][cp_id] = pickle.loads(cp_data)

                for row in conn.execute("SELECT blob_key, blob_data FROM blobs"):
                    b_key, b_data = row
                    self.blobs[pickle.loads(b_key)] = pickle.loads(b_data)

                for row in conn.execute("SELECT write_key, write_tuple FROM writes"):
                    w_key, w_data = row
                    outer_key, inner_key = pickle.loads(w_key)
                    self.writes[outer_key][inner_key] = pickle.loads(w_data)
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
                    data = pickle.dumps(self.storage[thread_id][checkpoint_ns][cp_id])
                    conn.execute(
                        "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?)",
                        (thread_id, checkpoint_ns, cp_id, data)
                    )
                for k, v in new_versions.items():
                    blob_key = (thread_id, checkpoint_ns, k, v)
                    if blob_key in self.blobs:
                        conn.execute(
                            "INSERT OR REPLACE INTO blobs VALUES (?, ?)",
                            (pickle.dumps(blob_key), pickle.dumps(self.blobs[blob_key]))
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
                        conn.execute(
                            "INSERT OR REPLACE INTO writes VALUES (?, ?)",
                            (pickle.dumps(full_key), pickle.dumps(write_tuple))
                        )
                conn.commit()
            finally:
                conn.close()
