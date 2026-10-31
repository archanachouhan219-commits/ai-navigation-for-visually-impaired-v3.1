"""Abstract base interface for speech synthesis (TTS) backends.

Enables future evaluation of alternative TTS engines (e.g. ElevenLabs, Vapi, Kokoro)
without modifying navigation or pipeline logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseTTS(ABC):
    """Abstract interface for text-to-speech output engines."""

    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        """Returns True if the engine is currently uttering speech."""
        pass

    @abstractmethod
    def say(self, text: str, urgent: bool = False, priority: Optional[Any] = None) -> None:
        """Queue or immediately utter a spoken message.

        Args:
            text: The text to be spoken.
            urgent: If True, indicates a critical hazard/stop message that should
                    interrupt active non-urgent speech and clear stale queues.
            priority: Optional TTSPriority level. If None, defaults to CRITICAL when
                      urgent=True, else NORMAL.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up background worker threads and release audio resources."""
        pass

    def stop(self) -> None:
        """Immediately interrupt active speech and discard pending lower-priority utterances."""
        pass
