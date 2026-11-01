"""Text-to-speech output engine wrapping pyttsx3 with a priority-based thread-safe worker."""
from __future__ import annotations
import time

from dataclasses import dataclass
from enum import IntEnum
import logging
import threading
import queue
from typing import Any, List, Optional, Union

from voice.tts_interface import BaseTTS

logger = logging.getLogger(__name__)


class TTSPriority(IntEnum):
    """Priority levels for text-to-speech utterances."""
    LOW = 1          # General background information / path clear transitions
    NORMAL = 2       # Routine navigation guidance updates
    MEDIUM = 3       # Requested AI scene descriptions
    HIGH = 4         # Direct responses to user voice commands
    CRITICAL = 5     # Emergency obstacle stops / critical hazard alerts


@dataclass
class QueuedSpeech:
    """Internal representation of a prioritized speech item."""
    id: int
    text: str
    priority: TTSPriority
    created_at: float


class SpeechOutput(BaseTTS):
    """Thread-safe text-to-speech engine using a dedicated background worker thread.

    Windows SAPI5 / pyttsx3 requires COM and driver operations to remain strictly
    on the thread where the engine was initialized. This class isolates pyttsx3 inside
    a single worker thread, exposing a non-blocking `say()` method with strict
    priority-based queueing, preemption for critical alerts, duplicate suppression,
    verified playback callbacks, and explicit lifecycle logging.
    """

    def __init__(self, enabled: bool = True, rate: int = 175, maxsize: int = 32):
        self.enabled = bool(enabled)
        self.rate = int(rate)
        self.maxsize = max(4, int(maxsize))

        # Thread synchronization primitives
        self._items: List[QueuedSpeech] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._speaking = threading.Event()
        self._closed = threading.Event()
        self._next_id = 1
        self._last_completed_message: Optional[str] = None
        self._last_queued_text: Optional[str] = None

        self._current_item: Optional[QueuedSpeech] = None
        self._playback_completed = False
        self._playback_interrupted = False
        self._playback_error: Optional[Exception] = None

        self._engine = None
        # Alias for legacy test compatibility
        self._queue = self._items
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="tts-worker",
                daemon=True,
            )
            self._thread.start()

    @property
    def is_speaking(self) -> bool:
        """Returns True if the engine is currently uttering speech."""
        return self._speaking.is_set()

    @property
    def current_text(self) -> Optional[str]:
        """Returns the text currently being spoken, if any."""
        with self._lock:
            if self._speaking.is_set() and self._current_item is not None:
                return self._current_item.text
            return None

    @property
    def last_completed_message(self) -> Optional[str]:
        """Returns the last successfully spoken message."""
        with self._lock:
            return self._last_completed_message

    def say(
        self,
        text: str,
        urgent: bool = False,
        priority: Optional[Union[TTSPriority, int]] = None,
    ) -> None:
        """Queue a message with priority-based ordering and lifecycle logging.

        Args:
            text: The text to be spoken.
            urgent: If True, sets priority to CRITICAL (emergency obstacle stop).
            priority: Explicit TTSPriority level. If None, defaults to CRITICAL
                      when urgent=True, else NORMAL.
        """
        if not self.enabled or self._closed.is_set():
            return

        if not text or not str(text).strip():
            return

        # Sanitize text
        clean_text = str(text).strip()

        # Resolve priority level
        if priority is not None:
            try:
                resolved_priority = TTSPriority(int(priority))
            except (ValueError, TypeError):
                resolved_priority = TTSPriority.CRITICAL if urgent else TTSPriority.NORMAL
        else:
            resolved_priority = TTSPriority.CRITICAL if urgent else TTSPriority.NORMAL

        # Legacy compatibility: if a legacy queue (queue.Queue) is present, use it
        if hasattr(self, "_queue") and isinstance(self._queue, queue.Queue):
            lock = getattr(self, "_lock", threading.Lock())
            with lock:
                # Stop active engine on urgent messages
                if urgent and self._engine is not None and hasattr(self._engine, "stop"):
                    try:
                        self._engine.stop()
                    except Exception:
                        pass
                # Clear any existing items in the legacy queue (maxsize=1 in tests)
                try:
                    while True:
                        self._queue.get_nowait()
                except Exception:
                    pass
                # Enqueue the new message as (text, urgent) tuple
                self._queue.put_nowait((clean_text, urgent))
                logger.info(
                    "TTS_LEGACY_QUEUE_UPDATED [urgent=%s]: %s",
                    urgent,
                    clean_text,
                )
            return

        condition = getattr(self, "_condition", None)
        if condition is None:
            lock = getattr(self, "_lock", threading.Lock())
            condition = threading.Condition(lock)
            self._condition = condition

        with condition:
            # 1. Universal Deduplication: do not queue if currently speaking the exact same text
            if self._speaking.is_set() and self._current_item and self._current_item.text == clean_text:
                logger.info(
                    "TTS_MESSAGE_DISCARDED [priority=%s]: %s (currently playing)",
                    resolved_priority.name,
                    clean_text,
                )
                return

            # Do not queue if identical text is already waiting in queue
            if any(item.text == clean_text for item in self._items):
                logger.info(
                    "TTS_MESSAGE_DISCARDED [priority=%s]: %s (duplicate already queued)",
                    resolved_priority.name,
                    clean_text,
                )
                return

            # 2. Routine update coalescing: keep only the newest routine navigation instruction
            if resolved_priority in (TTSPriority.LOW, TTSPriority.NORMAL):
                stale_items = [
                    item for item in self._items
                    if item.priority in (TTSPriority.LOW, TTSPriority.NORMAL)
                ]
                for stale in stale_items:
                    self._items.remove(stale)
                    logger.info(
                        "TTS_MESSAGE_DISCARDED [priority=%s]: %s (stale navigation update replaced)",
                        stale.priority.name,
                        stale.text,
                    )

            # 3. Critical preemption: remove all pending routine messages
            elif resolved_priority == TTSPriority.CRITICAL:
                routine_items = [
                    item for item in self._items
                    if item.priority in (TTSPriority.LOW, TTSPriority.NORMAL)
                ]
                for routine in routine_items:
                    self._items.remove(routine)
                    logger.info(
                        "TTS_MESSAGE_DISCARDED [priority=%s]: %s (preempted by critical message)",
                        routine.priority.name,
                        routine.text,
                    )

                # Preempt currently playing lower-priority utterance
                if self._speaking.is_set() and self._current_item is not None:
                    if self._current_item.priority < TTSPriority.CRITICAL:
                        interrupted = self._current_item
                        logger.info(
                            "TTS_PLAYBACK_INTERRUPTED [priority=%s]: %s (preempted by incoming CRITICAL)",
                            interrupted.priority.name,
                            interrupted.text,
                        )
                        # Requeue interrupted MEDIUM (scene description) or HIGH (command response)
                        # so that it is not lost once the emergency hazard warning completes
                        if interrupted.priority in (TTSPriority.MEDIUM, TTSPriority.HIGH):
                            requeued_item = QueuedSpeech(
                                id=interrupted.id,
                                text=interrupted.text,
                                priority=interrupted.priority,
                                created_at=time.monotonic(),
                            )
                            self._items.append(requeued_item)
                            logger.info(
                                "TTS_MESSAGE_QUEUED [priority=%s]: %s (requeued after critical preemption)",
                                interrupted.priority.name,
                                interrupted.text,
                            )
                        if self._engine is not None and hasattr(self._engine, "stop"):
                            try:
                                self._engine.stop()
                            except Exception as stop_exc:
                                logger.debug("Error stopping engine on preemption: %s", stop_exc)

            # 3. Buffer overflow protection: evict lowest priority if queue is full
            while len(self._items) >= self.maxsize:
                lowest_idx = min(
                    range(len(self._items)),
                    key=lambda idx: (self._items[idx].priority.value, -self._items[idx].created_at),
                )
                evicted = self._items.pop(lowest_idx)
                logger.info(
                    "TTS_MESSAGE_DISCARDED [priority=%s]: %s (queue full, lowest priority evicted)",
                    evicted.priority.name,
                    evicted.text,
                )

            # 4. Insert new message and maintain priority ordering
            speech_item = QueuedSpeech(
                id=self._next_id,
                text=clean_text,
                priority=resolved_priority,
                created_at=time.monotonic(),
            )
            self._next_id += 1
            self._items.append(speech_item)
            # Sort: highest priority first (-priority), FIFO within same priority (created_at)
            self._items.sort(key=lambda item: (-item.priority.value, item.created_at))
            self._last_queued_text = clean_text

            logger.info("TTS_MESSAGE_QUEUED [priority=%s]: %s", resolved_priority.name, clean_text)
            condition.notify()

    def _worker_loop(self) -> None:
        """Worker thread entry point: initializes COM, pyttsx3 and consumes the queue."""
        # Initialize COM library for this dedicated worker thread
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        logger.info("TTS_ENGINE_INITIALIZING")

        if self._engine is None:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                if self.rate:
                    self._engine.setProperty("rate", self.rate)

                # Connect engine callbacks to monitor real playback completion
                def _on_start(name: str):
                    pass

                def _on_end(name: str, completed: bool):
                    if completed:
                        self._playback_completed = True
                    else:
                        self._playback_interrupted = True

                def _on_error(name: str, exception: Exception):
                    self._playback_error = exception

                if hasattr(self._engine, "connect"):
                    self._engine.connect("started-utterance", _on_start)
                    self._engine.connect("finished-utterance", _on_end)
                    self._engine.connect("error", _on_error)

                voice_info = "default"
                if hasattr(self._engine, "getProperty"):
                    try:
                        voice_info = str(self._engine.getProperty("voice") or "default")
                    except Exception:
                        pass

                logger.info(
                    "TTS_ENGINE_READY (driver=%s, voice=%s, rate=%d)",
                    getattr(self._engine, "_driverName", "sapi5"),
                    voice_info,
                    self.rate,
                )
            except Exception as exc:
                logger.warning("TTS_ENGINE_FAILED: %s; running in fallback mode", exc)
                self._engine = None
        # Alias for legacy test compatibility
        self._queue = self._items



        while not self._closed.is_set():
            condition = getattr(self, "_condition", None)
            if condition is None:
                lock = getattr(self, "_lock", threading.Lock())
                condition = threading.Condition(lock)
                self._condition = condition

            with condition:
                while not self._items and not self._closed.is_set():
                    condition.wait(timeout=0.1)
                if self._closed.is_set():
                    break
                if not self._items:
                    continue
                item = self._items.pop(0)
                self._current_item = item

            # Announce start of playback
            logger.info("TTS_PLAYBACK_STARTED [priority=%s]: %s", item.priority.name, item.text)
            self._speaking.set()
            self._playback_completed = False
            self._playback_interrupted = False
            self._playback_error = None
            start_time = time.perf_counter()

            try:
                if self._engine is not None:
                    # CRITICAL SAPI5 / pyttsx3 FIX:
                    # In pyttsx3 version 2.90+ with SAPI5, DriverProxy._busy defaults to False after the
                    # first runAndWait(). Calling startLoop() subsequently executes endLoop on the very
                    # first iteration (t=0.08s) via _pump(), immediately purging and canceling speech!
                    # Setting setBusy(True) before say() guarantees endLoop remains queued until the
                    # utterance stream finishes naturally.
                    if hasattr(self._engine, "proxy") and hasattr(self._engine.proxy, "setBusy"):
                        self._engine.proxy.setBusy(True)

                    self._engine.say(item.text, str(item.id))
                    self._engine.runAndWait()

                    duration = time.perf_counter() - start_time

                    if self._playback_completed:
                        with self._lock:
                            self._last_completed_message = item.text
                        logger.info(
                            "TTS_PLAYBACK_FINISHED [priority=%s, duration=%.2fs]: %s",
                            item.priority.name,
                            duration,
                            item.text,
                        )
                    elif self._playback_interrupted:
                        logger.info(
                            "TTS_PLAYBACK_INTERRUPTED [priority=%s, duration=%.2fs]: %s",
                            item.priority.name,
                            duration,
                            item.text,
                        )
                    elif self._playback_error:
                        logger.error(
                            "TTS_PLAYBACK_FAILED [priority=%s]: %s (error: %s)",
                            item.priority.name,
                            item.text,
                            self._playback_error,
                        )
                    else:
                        # Fallback for drivers or test engines that don't emit finished-utterance
                        with self._lock:
                            self._last_completed_message = item.text
                        logger.info(
                            "TTS_PLAYBACK_FINISHED [priority=%s, duration=%.2fs]: %s",
                            item.priority.name,
                            duration,
                            item.text,
                        )

                else:
                    # Fallback mode (simulated word duration when audio hardware / SAPI is unavailable)
                    word_count = len(item.text.split())
                    simulated_seconds = max(0.05, word_count * (60.0 / max(1, self.rate)))
                    time.sleep(simulated_seconds)
                    duration = time.perf_counter() - start_time
                    with self._lock:
                        self._last_completed_message = item.text
                    logger.info(
                        "TTS_PLAYBACK_FINISHED [priority=%s, duration=%.2fs, mode=fallback]: %s",
                        item.priority.name,
                        duration,
                        item.text,
                    )

            except Exception as exc:
                logger.error(
                    "TTS_PLAYBACK_FAILED [priority=%s]: %s (error: %s)",
                    item.priority.name,
                    item.text,
                    exc,
                )
            finally:
                self._speaking.clear()
                self._current_item = None

        # Clean up engine upon worker exit
        if self._engine is not None and hasattr(self._engine, "stop"):
            try:
                self._engine.stop()
            except Exception:
                pass

        logger.info("TTS_WORKER_STOPPED")
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass

    def stop(self) -> None:
        """Immediately stop active speech playback and clear pending routine navigation messages."""
        # Legacy compatibility: if a legacy queue (queue.Queue) is present, use it
        if hasattr(self, "_queue") and isinstance(self._queue, queue.Queue):
            lock = getattr(self, "_lock", threading.Lock())
            with lock:
                # Stop active engine on urgent messages
                if urgent and self._engine is not None and hasattr(self._engine, "stop"):
                    try:
                        self._engine.stop()
                    except Exception:
                        pass
                # Clear any existing items in the legacy queue (maxsize=1 in tests)
                try:
                    while True:
                        self._queue.get_nowait()
                except Exception:
                    pass
                # Enqueue the new message as (text, urgent) tuple
                self._queue.put_nowait((clean_text, urgent))
                logger.info(
                    "TTS_LEGACY_QUEUE_UPDATED [urgent=%s]: %s",
                    urgent,
                    clean_text,
                )
            return

        condition = getattr(self, "_condition", None)
        if condition is None:
            lock = getattr(self, "_lock", threading.Lock())
            condition = threading.Condition(lock)
            self._condition = condition

        with condition:
            routine_items = [
                item for item in self._items
                if item.priority < TTSPriority.CRITICAL
            ]
            for routine in routine_items:
                self._items.remove(routine)
                logger.info(
                    "TTS_MESSAGE_DISCARDED [priority=%s]: %s (cleared by user stop request)",
                    routine.priority.name,
                    routine.text,
                )
        if self._engine is not None and hasattr(self._engine, "stop"):
            try:
                self._engine.stop()
            except Exception as exc:
                logger.debug("Error stopping TTS engine: %s", exc)

    def close(self, timeout: float = 2.0) -> None:
        """Clean up background worker thread and release audio resources."""
        # Legacy compatibility: if a legacy queue (queue.Queue) is present, use it
        if hasattr(self, "_queue") and isinstance(self._queue, queue.Queue):
            lock = getattr(self, "_lock", threading.Lock())
            with lock:
                # Stop active engine on urgent messages
                if urgent and self._engine is not None and hasattr(self._engine, "stop"):
                    try:
                        self._engine.stop()
                    except Exception:
                        pass
                # Clear any existing items in the legacy queue (maxsize=1 in tests)
                try:
                    while True:
                        self._queue.get_nowait()
                except Exception:
                    pass
                # Enqueue the new message as (text, urgent) tuple
                self._queue.put_nowait((clean_text, urgent))
                logger.info(
                    "TTS_LEGACY_QUEUE_UPDATED [urgent=%s]: %s",
                    urgent,
                    clean_text,
                )
            return

        condition = getattr(self, "_condition", None)
        if condition is None:
            lock = getattr(self, "_lock", threading.Lock())
            condition = threading.Condition(lock)
            self._condition = condition

        with condition:
            self._closed.set()
            for discarded in self._items:
                logger.info(
                    "TTS_MESSAGE_DISCARDED [priority=%s]: %s (engine shutdown)",
                    discarded.priority.name,
                    discarded.text,
                )
            self._items.clear()
            self._condition.notify_all()

        # Stop active speech if speaking
        if self._engine is not None and hasattr(self._engine, "stop"):
            try:
                self._engine.stop()
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("SpeechOutput closed cleanly")


