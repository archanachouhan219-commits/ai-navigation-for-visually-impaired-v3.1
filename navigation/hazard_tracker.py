"""Hazard state tracking and suppression engine for conservative navigation."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

from voice.text_to_speech import TTSPriority

logger = logging.getLogger(__name__)


@dataclass
class HazardState:
    """Represents a discrete, comparable navigation hazard state."""
    hazard_id: str           # Stable key: e.g. "danger:all_blocked", "warning:steer_left", "warning:stop", "caution:unverified", "info:clear"
    level: str               # "danger", "warning", "caution", "info"
    direction: Optional[str] # "left", "right", None
    severity: int            # 4=danger, 3=warning, 2=caution, 1=info
    is_imminent: bool        # True for collision danger or immediate stop
    text: str                # Recommended spoken text


class HazardTracker:
    """Tracks and throttles navigation hazard announcements.

    Implements:
    - State-based hazard comparison (identifying new, changed, cleared, or unchanged hazards).
    - Immediate announcement for new hazards or significant severity/direction changes.
    - Strict suppression of repeated announcements while a hazard remains unchanged.
    - Multi-frame confirmation to prevent single-frame flickering from causing spurious announcements.
    - Announcement of hazard clearance transitions.
    - Long-interval reminders (e.g. 30s) if a hazard persists unchanged.
    """

    def __init__(
        self,
        repeat_cooldown_seconds: float = 30.0,
        change_cooldown_seconds: float = 2.0,
        min_persistence_frames: int = 2,
    ):
        self.repeat_cooldown_seconds = float(repeat_cooldown_seconds)
        self.change_cooldown_seconds = float(change_cooldown_seconds)
        self.min_persistence_frames = max(1, int(min_persistence_frames))

        self.last_announced_state: Optional[HazardState] = None
        self.last_announced_at: float = 0.0

        # Confirmation tracking
        self.pending_hazard_id: Optional[str] = None
        self.pending_streak: int = 0

    def reset(self) -> None:
        """Reset state tracking."""
        self.last_announced_state = None
        self.last_announced_at = 0.0
        self.pending_hazard_id = None
        self.pending_streak = 0

    def create_state(self, decision: Dict[str, Any]) -> HazardState:
        """Construct a structured HazardState from a NavigationEngine decision."""
        level = decision.get("level", "info")
        direction = decision.get("direction")
        text = decision.get("text", "")

        if level == "danger":
            hazard_id = "danger:all_blocked"
            severity = 4
            is_imminent = True
        elif level == "warning":
            if direction in ("left", "right"):
                hazard_id = f"warning:steer_{direction}"
                is_imminent = False
            else:
                hazard_id = "warning:stop"
                is_imminent = True
            severity = 3
        elif level == "caution":
            hazard_id = f"caution:{direction or 'unverified'}"
            severity = 2
            is_imminent = False
        else:
            hazard_id = "info:clear"
            severity = 1
            is_imminent = False

        return HazardState(
            hazard_id=hazard_id,
            level=level,
            direction=direction,
            severity=severity,
            is_imminent=is_imminent,
            text=text,
        )

    def evaluate(
        self,
        decision: Dict[str, Any],
        now: float,
    ) -> Tuple[bool, Optional[TTSPriority], str]:
        """Evaluate whether a navigation decision should be spoken.

        Returns:
            (should_speak, priority, reason)
        """
        state = self.create_state(decision)

        # 1. Multi-frame persistence confirmation for transitions
        if self.last_announced_state is None or state.hazard_id != self.last_announced_state.hazard_id:
            # Imminent collision danger needs zero persistence delay
            if state.is_imminent:
                confirmed = True
                self.pending_hazard_id = None
                self.pending_streak = 0
            else:
                if state.hazard_id == self.pending_hazard_id:
                    self.pending_streak += 1
                else:
                    self.pending_hazard_id = state.hazard_id
                    self.pending_streak = 1
                confirmed = (self.pending_streak >= self.min_persistence_frames)
        else:
            confirmed = True
            self.pending_hazard_id = None
            self.pending_streak = 0

        if not confirmed:
            return False, None, "unconfirmed_transition"

        # 2. Evaluate against last announced state
        # Case A: First ever announcement
        if self.last_announced_state is None:
            priority = TTSPriority.CRITICAL if state.is_imminent else (
                TTSPriority.HIGH if state.severity == 3 else (
                    TTSPriority.NORMAL if state.severity == 2 else TTSPriority.LOW
                )
            )
            self._commit(state, now)
            return True, priority, "initial_announcement"

        last = self.last_announced_state
        time_since_last = now - self.last_announced_at

        # Case B: Unchanged Hazard State (same hazard_id)
        if state.hazard_id == last.hazard_id:
            if time_since_last >= self.repeat_cooldown_seconds:
                # Periodic reminder for persistent obstacle
                # Note: Reminders are queued as NORMAL, never spamming CRITICAL
                priority = TTSPriority.NORMAL if state.severity <= 3 else TTSPriority.CRITICAL
                self._commit(state, now)
                return True, priority, "periodic_reminder"
            else:
                # SUPPRESSED: unchanged hazard
                return False, None, "suppressed_unchanged"

        # Case C: Meaningful Change in Hazard State (different hazard_id)
        severity_escalated = state.severity > last.severity
        direction_changed = (
            state.direction != last.direction
            and state.direction is not None
            and last.direction is not None
        )
        hazard_cleared = (last.severity >= 3 and state.severity <= 2)

        # Imminent danger escalations bypass change cooldown
        if state.is_imminent and severity_escalated:
            self._commit(state, now)
            return True, TTSPriority.CRITICAL, "imminent_escalation"

        # Other changes respect minimum change cooldown to prevent rapid switching
        if time_since_last < self.change_cooldown_seconds:
            return False, None, "cooldown_active"

        if severity_escalated or direction_changed:
            priority = TTSPriority.CRITICAL if state.is_imminent else TTSPriority.HIGH
            self._commit(state, now)
            reason = "severity_escalated" if severity_escalated else "direction_changed"
            return True, priority, reason

        if hazard_cleared:
            self._commit(state, now)
            return True, TTSPriority.LOW, "hazard_cleared"

        # Mild transition (e.g. caution to info or info to caution)
        if state.severity == 2:
            priority = TTSPriority.NORMAL
        else:
            priority = TTSPriority.LOW

        self._commit(state, now)
        return True, priority, "state_transition"

    def _commit(self, state: HazardState, now: float) -> None:
        self.last_announced_state = state
        self.last_announced_at = now
        self.pending_hazard_id = None
        self.pending_streak = 0

