"""Thread-safe session data recorder for the LLM proxy."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from .models import CompletionRecord, SessionRecord, TrainingRoundTiming

logger = logging.getLogger(__name__)


class SessionClosedError(Exception):
    """Raised when recording for a session that has been completed/deleted."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session {session_id} is closed, rejecting new completions")


class SessionAlreadyExistsError(Exception):
    """Raised when attempting to create a session with a duplicate ID."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session {session_id} already exists")


class SessionRecorder:
    """Manages session records for the LLM proxy.

    Thread-safe: multiple proxy request handlers may record completions
    concurrently for different sessions.

    When *persist_dir* is set, every turn is appended to a JSONL file
    on disk as it is recorded.  On startup, existing JSONL files in
    the directory are loaded back into memory so sessions survive
    proxy restarts.
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._closed_sessions: set[str] = set()
        self._lock = threading.Lock()
        self._persist_dir = persist_dir
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._load_from_disk()

    def _session_path(self, session_id: str) -> str:
        return os.path.join(self._persist_dir, f"{session_id}.jsonl")

    def _load_from_disk(self) -> None:
        """Reload sessions from JSONL files on startup."""
        loaded = 0
        for fname in os.listdir(self._persist_dir):
            if not fname.endswith(".jsonl"):
                continue
            session_id = fname[:-6]  # strip .jsonl
            fpath = os.path.join(self._persist_dir, fname)
            try:
                session = SessionRecord(session_id=session_id)
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        if entry.get("type") == "session_create":
                            session.created_at = entry.get("created_at", session.created_at)
                            session.completed = entry.get("completed", False)
                        elif entry.get("type") == "turn":
                            record = CompletionRecord(**entry["data"])
                            session.turns.append(record)
                        elif entry.get("type") == "completed":
                            session.completed = True
                self._sessions[session_id] = session
                loaded += 1
            except Exception as e:
                logger.warning("Failed to load session %s from disk: %s", session_id, e)
        if loaded:
            logger.info("Loaded %d sessions from %s", loaded, self._persist_dir)

    def _append_to_disk(self, session_id: str, entry: dict) -> None:
        """Append a single JSON line to the session's JSONL file."""
        if not self._persist_dir:
            return
        try:
            fpath = self._session_path(session_id)
            with open(fpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to persist turn for session %s: %s", session_id, e)

    def create_session(self, session_id: str) -> None:
        """Create a new session for recording.

        Raises:
            SessionAlreadyExistsError: If a session with the same ID already
                exists (either active or previously closed).
        """
        with self._lock:
            if session_id in self._sessions or session_id in self._closed_sessions:
                raise SessionAlreadyExistsError(session_id)
            session = SessionRecord(session_id=session_id)
            self._sessions[session_id] = session
        self._append_to_disk(session_id, {
            "type": "session_create",
            "created_at": session.created_at,
        })

    def reset_session(self, session_id: str) -> bool:
        """Reset an active session by clearing all recorded turns.

        Used before retrying an agent run to avoid double-recording.
        Only works on sessions that are still active (not closed).

        Returns:
            ``True`` if the session was found and reset, ``False`` otherwise.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.turns.clear()
        logger.info("Session %s reset (turns cleared)", session_id)
        return True

    def record_completion(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        completion_text: str,
        token_ids: list[int] | None = None,
        logprobs: list[float] | None = None,
        finish_reason: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        timing: dict[str, float] | None = None,
        node_id: str | None = None,
        gpu_id: str | None = None,
        worker_id: str | None = None,
        rank_info: dict[str, str] | None = None,
    ) -> None:
        """Record a single LLM completion for a session.

        For multi-turn conversations, only the *new* messages (the delta
        since the previous turn) are stored in ``request_messages`` to
        avoid duplicating the conversation history.  The first turn stores
        the full messages array; subsequent turns store only the messages
        that were appended since the previous request.
        """
        with self._lock:
            if session_id in self._closed_sessions:
                raise SessionClosedError(session_id)
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning(
                    "Session %s not found, auto-creating for recording", session_id
                )
                session = SessionRecord(session_id=session_id)
                self._sessions[session_id] = session

            # Compute delta: store only new messages not present in the
            # previous turn's request to avoid O(n²) duplication in
            # multi-turn conversations.
            #
            # Turn 0 stores full messages; subsequent turns store deltas.
            # The sum of all stored message-list lengths equals the total
            # message count at the most recent turn, which is used to
            # compute the slice point for the next delta.
            if session.turns:
                prev_full_count = sum(
                    len(t.request_messages) for t in session.turns
                )
                if len(messages) > prev_full_count:
                    stored_messages = messages[prev_full_count:]
                elif len(messages) == prev_full_count:
                    # No new messages (e.g. agent retried with identical history)
                    stored_messages = []
                else:
                    # Agent modified or shortened the history — store full
                    # messages as a fallback to avoid data loss.
                    logger.debug(
                        "Session %s: message count decreased from %d to %d, "
                        "storing full messages",
                        session_id, prev_full_count, len(messages),
                    )
                    stored_messages = messages
            else:
                # First turn: store full messages
                stored_messages = messages

            record = CompletionRecord(
                request_messages=stored_messages,
                completion_text=completion_text,
                completion_token_ids=token_ids or [],
                completion_logprobs=logprobs or [],
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                timing=timing,
                node_id=node_id,
                gpu_id=gpu_id,
                worker_id=worker_id,
                rank_info=rank_info,
            )
            session.turns.append(record)
        self._append_to_disk(session_id, {
            "type": "turn",
            "data": record.model_dump(),
        })

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve session data. Returns None if not found."""
        with self._lock:
            return self._sessions.get(session_id)

    def is_session_closed(self, session_id: str) -> bool:
        """Check whether a session has been completed or deleted."""
        with self._lock:
            return session_id in self._closed_sessions

    def mark_completed(self, session_id: str) -> None:
        """Mark a session as completed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.completed = True
            self._closed_sessions.add(session_id)
        self._append_to_disk(session_id, {"type": "completed"})

    def delete_session(self, session_id: str) -> None:
        """Remove a session and free its memory."""
        with self._lock:
            self._sessions.pop(session_id, None)
            self._closed_sessions.add(session_id)
        if self._persist_dir:
            fpath = self._session_path(session_id)
            try:
                os.remove(fpath)
            except FileNotFoundError:
                pass

    def pop_session(self, session_id: str) -> SessionRecord | None:
        """Remove and return a session atomically.

        Also deletes the persisted JSONL file (if any), mirroring
        :meth:`delete_session`.
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
            self._closed_sessions.add(session_id)
        if self._persist_dir:
            fpath = self._session_path(session_id)
            try:
                os.remove(fpath)
            except FileNotFoundError:
                pass
        return session

    def dump_session_to_file(
        self,
        session_id: str,
        dump_dir: str,
        *,
        remove: bool = False,
    ) -> str | None:
        """Serialize a session to ``{dump_dir}/{session_id}.json``.

        If *remove* is ``True``, the session is atomically removed from
        memory after a successful dump.

        Returns the file path written, or ``None`` if the session was
        not found.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            data = session.model_dump()
            if remove:
                del self._sessions[session_id]

        os.makedirs(dump_dir, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(data.get("created_at", 0)))
        filename = f"{ts}_{session_id}.json"
        filepath = os.path.join(dump_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Session %s dumped to %s (%d turns)", session_id, filepath, len(data.get("turns", [])))
        return filepath

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        with self._lock:
            return list(self._sessions.keys())

    def list_completed_sessions(self, dump_dir: str) -> list[dict]:
        """List completed sessions from dump_dir.

        Scans ``dump_dir`` for JSON files with name format
        ``{timestamp}_{session_id}.json``, reads each to extract summary
        info.  If multiple dumps exist for the same session_id, only the
        latest (by filename sort order) is returned.

        Returns:
            A list of dicts with keys: session_id, turns, completed.
        """
        if not dump_dir or not os.path.isdir(dump_dir):
            return []

        # Collect files grouped by session_id, keeping only the latest
        # filename (timestamps sort lexicographically).
        session_files: dict[str, str] = {}  # session_id -> latest filename
        try:
            filenames = os.listdir(dump_dir)
        except OSError as e:
            logger.warning("Failed to list dump_dir %s: %s", dump_dir, e)
            return []

        for fname in sorted(filenames):
            if not fname.endswith(".json"):
                continue
            # Format: {timestamp}_{session_id}.json
            stem = fname[:-5]  # strip .json
            parts = stem.split("_", 1)
            if len(parts) != 2:
                continue
            _ts, session_id = parts
            # Later files (sorted) overwrite earlier ones for same session_id
            session_files[session_id] = fname

        results: list[dict] = []
        for session_id, fname in session_files.items():
            fpath = os.path.join(dump_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                turns = len(data.get("turns", []))
                completed = data.get("completed", True)
                results.append({
                    "session_id": session_id,
                    "turns": turns,
                    "completed": completed,
                })
            except Exception as e:
                logger.warning(
                    "Failed to read completed session file %s: %s", fpath, e
                )
        return results

    def load_completed_session(
        self, session_id: str, dump_dir: str
    ) -> SessionRecord | None:
        """Load a completed session's full record from dump_dir.

        Searches ``dump_dir`` for JSON files named
        ``{timestamp}_{session_id}.json``.  If multiple dumps exist for
        the same session_id (different timestamps), the latest one is
        returned.

        Returns:
            A :class:`SessionRecord` deserialized from the file, or
            ``None`` if no matching file is found.
        """
        if not dump_dir or not os.path.isdir(dump_dir):
            return None

        try:
            filenames = os.listdir(dump_dir)
        except OSError as e:
            logger.warning("Failed to list dump_dir %s: %s", dump_dir, e)
            return None

        # Collect all matching files for this session_id.
        # File format: {timestamp}_{session_id}.json where timestamp is
        # YYYYMMDDTHHMMSS (no underscores), so splitting on the first
        # underscore cleanly separates timestamp from session_id.
        matching: list[str] = []
        for fname in filenames:
            if not fname.endswith(".json"):
                continue
            stem = fname[:-5]  # strip .json
            parts = stem.split("_", 1)
            if len(parts) != 2:
                continue
            _ts, sid = parts
            if sid == session_id:
                matching.append(fname)

        if not matching:
            return None

        # Sort lexicographically — timestamps sort correctly, so the
        # last element is the latest dump.
        matching.sort()
        latest_fname = matching[-1]
        fpath = os.path.join(dump_dir, latest_fname)

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return SessionRecord.model_validate(data)
        except Exception as e:
            logger.warning(
                "Failed to load completed session %s from %s: %s",
                session_id, fpath, e,
            )
            return None

    # ------------------------------------------------------------------
    # Training round timing
    # ------------------------------------------------------------------

    def record_training_timing(self, timing: TrainingRoundTiming, dump_dir: str) -> str:
        """Record training round timing to a JSON file in dump_dir.

        Returns the file path.
        """
        os.makedirs(dump_dir, exist_ok=True)
        filename = f"training_timing_{timing.epoch}_{timing.global_step}.json"
        filepath = os.path.join(dump_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(timing.model_dump_json(indent=2))
        return filepath

    def list_training_timings(self, dump_dir: str) -> list[dict]:
        """List all training timing records from dump_dir."""
        if not os.path.isdir(dump_dir):
            return []
        items = []
        for fn in os.listdir(dump_dir):
            if fn.startswith("training_timing_") and fn.endswith(".json"):
                filepath = os.path.join(dump_dir, fn)
                try:
                    with open(filepath, encoding="utf-8") as f:
                        data = json.load(f)
                    items.append(data)
                except Exception:
                    pass
        items.sort(key=lambda x: (x.get("epoch", 0), x.get("global_step", 0)))
        return items

    def load_training_timing(self, epoch: int, global_step: int, dump_dir: str) -> TrainingRoundTiming | None:
        """Load a specific training timing record."""
        filename = f"training_timing_{epoch}_{global_step}.json"
        filepath = os.path.join(dump_dir, filename)
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            return TrainingRoundTiming.model_validate(data)
        except Exception:
            return None
