# analytics/state_cache.py
"""
State Cache - Authoritative Source of Truth for Activity Monitor State.
Enterprise-grade implementation where RAM is the authority, DB is for persistence only.

Architecture:
- StateCache (RAM) is THE source of truth for current state
- Database is written to ONLY on state transitions (not every frame)
- Background thread handles DB persistence asynchronously
- Queries read from cache, never wait for DB

This prevents:
- Database overwhelm from high-frequency updates
- Latency from DB I/O on critical path
- Cache-DB inconsistency issues
"""

import time
import logging
import threading
import queue
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

log = logging.getLogger("state_cache")


class StateType(Enum):
    """Types of state tracked."""
    POSTURE = "posture"
    ACTIVITY = "activity"
    ALERT = "alert"
    SUPPORT_SURFACE = "support_surface"


@dataclass
class PersonState:
    """Complete state for a single person."""
    person_id: str
    posture: str = "UNKNOWN"
    posture_confidence: float = 0.0
    posture_subtype: Optional[str] = None
    activity: str = "UNKNOWN"
    activity_confidence: float = 0.0
    support_surface_id: Optional[str] = None
    support_surface_type: Optional[str] = None
    last_updated: float = field(default_factory=time.time)
    state_start_time: float = field(default_factory=time.time)
    frame_count: int = 0

    def update_posture(self, posture: str, confidence: float, subtype: Optional[str] = None):
        """Update posture state."""
        state_changed = self.posture != posture
        self.posture = posture
        self.posture_confidence = confidence
        self.posture_subtype = subtype
        self.last_updated = time.time()
        self.frame_count += 1
        if state_changed:
            self.state_start_time = time.time()
        return state_changed

    def update_activity(self, activity: str, confidence: float):
        """Update activity state."""
        state_changed = self.activity != activity
        self.activity = activity
        self.activity_confidence = confidence
        self.last_updated = time.time()
        if state_changed:
            self.state_start_time = time.time()
        return state_changed

    def update_support_surface(self, surface_id: Optional[str], surface_type: Optional[str]):
        """Update support surface info."""
        self.support_surface_id = surface_id
        self.support_surface_type = surface_type
        self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "person_id": self.person_id,
            "posture": self.posture,
            "posture_confidence": round(self.posture_confidence, 2),
            "posture_subtype": self.posture_subtype,
            "activity": self.activity,
            "activity_confidence": round(self.activity_confidence, 2),
            "support_surface_id": self.support_surface_id,
            "support_surface_type": self.support_surface_type,
            "time_in_state": round(time.time() - self.state_start_time, 1),
            "last_updated": self.last_updated,
        }


@dataclass
class StateTransition:
    """Record of a state transition for DB persistence."""
    timestamp: float
    person_id: str
    state_type: StateType
    old_state: str
    new_state: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateCache:
    """
    Authoritative state cache - THE source of truth.

    Features:
    - In-memory state for all tracked persons
    - Async DB persistence only on state transitions
    - Thread-safe operations
    - Configurable retention and cleanup

    Usage:
        cache = StateCache()
        cache.start()

        # Update state (high frequency - every frame)
        cache.update_posture("person_1", "SITTING", 0.85, "leaning_back")

        # Query state (always from RAM, instant)
        state = cache.get_person_state("person_1")

        cache.stop()
    """

    def __init__(
        self,
        max_persons: int = 50,
        stale_timeout: float = 30.0,
        db_writer: Optional[Callable[[StateTransition], None]] = None,
    ):
        """
        Args:
            max_persons: Maximum tracked persons
            stale_timeout: Time after which inactive persons are cleaned up
            db_writer: Optional callback for DB persistence
        """
        self.max_persons = max_persons
        self.stale_timeout = stale_timeout
        self._db_writer = db_writer

        # Person states (OrderedDict for LRU-like behavior)
        self._states: OrderedDict[str, PersonState] = OrderedDict()
        self._lock = threading.RLock()

        # Transition queue for async DB writes
        self._transition_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._running = False
        self._writer_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            "updates": 0,
            "transitions": 0,
            "db_writes": 0,
            "db_errors": 0,
        }

        log.info("StateCache initialized (max_persons=%d, stale_timeout=%.1fs)",
                max_persons, stale_timeout)

    def start(self):
        """Start background writer thread."""
        if self._running:
            return
        self._running = True
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        log.info("StateCache writer thread started")

    def stop(self):
        """Stop background writer and flush pending transitions."""
        self._running = False
        if self._writer_thread:
            # Signal thread to exit
            self._transition_queue.put(None)
            self._writer_thread.join(timeout=5.0)
        # Flush any remaining transitions
        self._flush_transitions()
        log.info("StateCache stopped (stats: %s)", self._stats)

    def _writer_loop(self):
        """Background thread for async DB writes."""
        while self._running:
            try:
                transition = self._transition_queue.get(timeout=1.0)
                if transition is None:
                    break
                self._write_transition(transition)
            except queue.Empty:
                continue
            except Exception as e:
                log.exception("Writer thread error: %s", e)

    def _write_transition(self, transition: StateTransition):
        """Write transition to database."""
        if self._db_writer is None:
            return

        try:
            self._db_writer(transition)
            self._stats["db_writes"] += 1
        except Exception as e:
            self._stats["db_errors"] += 1
            log.warning("DB write failed for transition: %s", e)

    def _flush_transitions(self):
        """Flush all pending transitions to DB."""
        while not self._transition_queue.empty():
            try:
                transition = self._transition_queue.get_nowait()
                if transition is not None:
                    self._write_transition(transition)
            except queue.Empty:
                break

    def _get_or_create_person(self, person_id: str) -> PersonState:
        """Get or create person state."""
        if person_id not in self._states:
            if len(self._states) >= self.max_persons:
                # Remove oldest (first item)
                oldest_id = next(iter(self._states))
                del self._states[oldest_id]
                log.debug("Evicted stale person: %s", oldest_id)

            self._states[person_id] = PersonState(person_id=person_id)

        # Move to end (most recently used)
        self._states.move_to_end(person_id)
        return self._states[person_id]

    def update_posture(
        self,
        person_id: str,
        posture: str,
        confidence: float,
        subtype: Optional[str] = None,
        support_surface_id: Optional[str] = None,
        support_surface_type: Optional[str] = None,
    ) -> bool:
        """
        Update posture state for a person.

        Args:
            person_id: Person identifier
            posture: Posture state (STANDING, SITTING, LYING, etc.)
            confidence: Confidence score 0-1
            subtype: Optional subtype (e.g., "leaning_back")
            support_surface_id: Optional support surface ID
            support_surface_type: Optional support surface type

        Returns:
            True if state changed (transition occurred)
        """
        with self._lock:
            person = self._get_or_create_person(person_id)
            old_posture = person.posture

            state_changed = person.update_posture(posture, confidence, subtype)
            person.update_support_surface(support_surface_id, support_surface_type)

            self._stats["updates"] += 1

            if state_changed:
                self._stats["transitions"] += 1
                # Queue transition for async DB write
                transition = StateTransition(
                    timestamp=time.time(),
                    person_id=person_id,
                    state_type=StateType.POSTURE,
                    old_state=old_posture,
                    new_state=posture,
                    confidence=confidence,
                    metadata={
                        "subtype": subtype,
                        "support_surface_id": support_surface_id,
                        "support_surface_type": support_surface_type,
                    }
                )
                try:
                    self._transition_queue.put_nowait(transition)
                except queue.Full:
                    log.warning("Transition queue full, dropping oldest")

            return state_changed

    def update_activity(
        self,
        person_id: str,
        activity: str,
        confidence: float,
    ) -> bool:
        """
        Update activity state for a person.

        Returns:
            True if state changed
        """
        with self._lock:
            person = self._get_or_create_person(person_id)
            old_activity = person.activity

            state_changed = person.update_activity(activity, confidence)
            self._stats["updates"] += 1

            if state_changed:
                self._stats["transitions"] += 1
                transition = StateTransition(
                    timestamp=time.time(),
                    person_id=person_id,
                    state_type=StateType.ACTIVITY,
                    old_state=old_activity,
                    new_state=activity,
                    confidence=confidence,
                )
                try:
                    self._transition_queue.put_nowait(transition)
                except queue.Full:
                    pass

            return state_changed

    def get_person_state(self, person_id: str) -> Optional[PersonState]:
        """
        Get current state for a person.
        Always reads from RAM - instant, never waits for DB.
        """
        with self._lock:
            return self._states.get(person_id)

    def get_all_states(self) -> Dict[str, PersonState]:
        """Get all person states (copy to avoid holding lock)."""
        with self._lock:
            return dict(self._states)

    def get_posture(self, person_id: str) -> Optional[str]:
        """Quick access to posture state."""
        state = self.get_person_state(person_id)
        return state.posture if state else None

    def get_activity(self, person_id: str) -> Optional[str]:
        """Quick access to activity state."""
        state = self.get_person_state(person_id)
        return state.activity if state else None

    def remove_person(self, person_id: str):
        """Remove person from tracking."""
        with self._lock:
            self._states.pop(person_id, None)

    def cleanup_stale(self):
        """Remove stale persons not updated recently."""
        with self._lock:
            now = time.time()
            stale = [
                pid for pid, state in self._states.items()
                if now - state.last_updated > self.stale_timeout
            ]
            for pid in stale:
                del self._states[pid]
                log.debug("Cleaned up stale person: %s", pid)
            return len(stale)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                **self._stats,
                "tracked_persons": len(self._states),
                "queue_size": self._transition_queue.qsize(),
            }

    def get_display_state(self, person_id: str) -> Dict[str, Any]:
        """
        Get state formatted for display overlay.

        Returns dict ready for UI rendering.
        """
        state = self.get_person_state(person_id)
        if state is None:
            return {
                "posture": "UNKNOWN",
                "posture_confidence": 0.0,
                "activity": "UNKNOWN",
                "support": None,
            }

        return {
            "posture": state.posture,
            "posture_confidence": state.posture_confidence,
            "posture_subtype": state.posture_subtype,
            "activity": state.activity,
            "activity_confidence": state.activity_confidence,
            "support": state.support_surface_id,
            "time_in_state": round(time.time() - state.state_start_time, 1),
        }


# Singleton instance
_state_cache: Optional[StateCache] = None
_cache_lock = threading.Lock()


def get_state_cache(**kwargs) -> StateCache:
    """Get or create global state cache."""
    global _state_cache
    if _state_cache is None:
        with _cache_lock:
            if _state_cache is None:
                _state_cache = StateCache(**kwargs)
                _state_cache.start()
    return _state_cache


def close_state_cache():
    """Close global state cache."""
    global _state_cache
    if _state_cache is not None:
        with _cache_lock:
            if _state_cache is not None:
                _state_cache.stop()
                _state_cache = None
