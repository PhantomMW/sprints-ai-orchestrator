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
    """

    def __init__(self, db_path: str | Path = "data/orchestrator_checkpoints.sqlite3"):
        super().__init__()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        self._load_from_db()

    def _init_db(self) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
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
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    channel TEXT,
                    version TEXT,
                    blob_data BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    checkpoint_id TEXT,
                    task_id TEXT,
                    idx INTEGER,
                    write_tuple BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                )
            """)

    def _load_from_db(self) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            for row in conn.execute("SELECT thread_id, checkpoint_ns, checkpoint_id, checkpoint_tuple FROM checkpoints"):
                t_id, ns, cp_id, cp_data = row
                self.storage[t_id][ns][cp_id] = pickle.loads(cp_data)

            for row in conn.execute("SELECT thread_id, checkpoint_ns, channel, version, blob_data FROM blobs"):
                t_id, ns, chan, ver, b_data = row
                self.blobs[(t_id, ns, chan, ver)] = pickle.loads(b_data)

            for row in conn.execute("SELECT thread_id, checkpoint_ns, checkpoint_id, task_id, idx, write_tuple FROM writes"):
                t_id, ns, cp_id, task_id, idx, w_data = row
                self.writes[(t_id, ns, cp_id)][(task_id, idx)] = pickle.loads(w_data)

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

        with self._lock, sqlite3.connect(self.db_path) as conn:
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
                        "INSERT OR REPLACE INTO blobs VALUES (?, ?, ?, ?, ?)",
                        (thread_id, checkpoint_ns, k, str(v), pickle.dumps(self.blobs[blob_key]))
                    )
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

        with self._lock, sqlite3.connect(self.db_path) as conn:
            if outer_key in self.writes:
                for (t_id, idx), write_tuple in self.writes[outer_key].items():
                    conn.execute(
                        "INSERT OR REPLACE INTO writes VALUES (?, ?, ?, ?, ?, ?)",
                        (thread_id, checkpoint_ns, checkpoint_id, t_id, idx, pickle.dumps(write_tuple))
                    )
