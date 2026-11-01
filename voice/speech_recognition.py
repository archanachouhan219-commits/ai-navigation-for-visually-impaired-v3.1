"""Optional offline Vosk microphone listener with TTS echo suppression."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import queue
import re
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class VoiceListener:
    """Listens for microphone audio and decodes speech using Vosk.

    Implements:
    - Path resolution for acoustic models with clear diagnostics when missing.
    - Graceful degradation if sounddevice / PortAudio or microphone is unavailable.
    - Continuous audio capture with selective acoustic self-echo filtering.
    - Barge-in support so user can issue commands while TTS speech is playing.
    - Structured lifecycle logging (FRAME_RECEIVED, PARTIAL, FINAL, FAILED).
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        callback: Callable[[str], None],
        should_ignore: Optional[Callable[[], bool]] = None,
        get_active_speech: Optional[Callable[[], Optional[str]]] = None,
    ):
        self.raw_model_path = str(model_path)
        self.callback = callback
        self.should_ignore = should_ignore or (lambda: False)
        self.get_active_speech = get_active_speech
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.available = False
        self.resolved_model_path: Optional[Path] = None

        self._check_model_path()

    def _check_model_path(self) -> None:
        """Resolve and validate the Vosk model directory path."""
        p = Path(self.raw_model_path)
        if not p.is_absolute():
            # Check relative to cwd, then relative to project root
            cwd_path = Path.cwd() / p
            root_path = Path(__file__).resolve().parent.parent / p
            if cwd_path.is_dir():
                self.resolved_model_path = cwd_path
            elif root_path.is_dir():
                self.resolved_model_path = root_path
            else:
                self.resolved_model_path = root_path
        else:
            self.resolved_model_path = p

        if self.resolved_model_path.is_dir() and any(self.resolved_model_path.iterdir()):
            self.available = True
        else:
            logger.warning(
                "Vosk acoustic model directory not found at '%s' (resolved to '%s'). "
                "Offline voice recognition will be unavailable. To enable voice interaction, "
                "download an acoustic model (e.g. vosk-model-small-en-us-0.15 from "
                "https://alphacephei.com/vosk/models) and extract it to '%s'.",
                self.raw_model_path,
                self.resolved_model_path,
                self.resolved_model_path,
            )
            self.available = False

    def start(self) -> None:
        """Start the background voice listener thread if available."""
        if not self.available:
            logger.info("VoiceListener not started: acoustic model unavailable.")
            return

        if self.thread and self.thread.is_alive():
            logger.warning("VoiceListener thread is already active.")
            return

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="voice-listener", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            import sounddevice as sd
            from vosk import Model, KaldiRecognizer
        except ImportError as exc:
            logger.warning("Voice recognition dependencies missing: %s", exc)
            return

        # Query microphone availability
        try:
            input_device = sd.query_devices(kind="input")
            if not input_device:
                logger.warning("No microphone input device detected; voice recognition disabled.")
                return
        except Exception as exc:
            logger.warning("Microphone query failed: %s; voice recognition disabled.", exc)
            return

        try:
            model = Model(str(self.resolved_model_path))
            recognizer = KaldiRecognizer(model, 16000)
        except Exception as exc:
            logger.warning("Failed to initialize Vosk model from '%s': %s", self.resolved_model_path, exc)
            return

        audio_queue: queue.Queue = queue.Queue(maxsize=16)
        frame_counter = 0

        def audio_callback(indata, frames, time_info, status):
            nonlocal frame_counter
            frame_counter += 1
            if status:
                logger.debug("Audio callback status: %s", status)

            # Sampled frame reception log (every 50 frames, ~1.6s)
            if frame_counter % 50 == 0:
                logger.debug("VOICE_AUDIO_FRAME_RECEIVED (frames=%d, bytes=%d, sample #%d)", frames, len(indata), frame_counter)

            # If legacy should_ignore is active and no get_active_speech provided, drop
            if self.get_active_speech is None and self.should_ignore():
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break
                return

            try:
                audio_queue.put_nowait(bytes(indata))
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(bytes(indata))
                except queue.Full:
                    pass

        try:
            with sd.RawInputStream(
                samplerate=16000,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=audio_callback,
            ):
                logger.info("Offline voice listener started successfully (16 kHz mono)")

                while not self.stop_event.is_set():
                    try:
                        data = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    # If legacy should_ignore is explicitly active without active_speech callback:
                    if self.get_active_speech is None and self.should_ignore():
                        try:
                            recognizer.Reset()
                        except Exception:
                            pass
                        continue

                    if recognizer.AcceptWaveform(data):
                        try:
                            res = json.loads(recognizer.Result())
                            text = res.get("text", "").strip()
                            if text:
                                logger.info("VOICE_FINAL_RECOGNITION: %r", text)

                                # Self-echo detection against active TTS output
                                is_echo = False
                                if self.get_active_speech is not None:
                                    active_tts = self.get_active_speech()
                                    if active_tts:
                                        rec_words = set(re.findall(r"\w+", text.lower()))
                                        tts_words = set(re.findall(r"\w+", active_tts.lower()))
                                        # Specific barge-in command words are NEVER suppressed as echo
                                        barge_in_words = {
                                            "stop", "pause", "quiet", "cancel", "silence",
                                            "resume", "unmute", "repeat", "again",
                                        }
                                        if not (rec_words & barge_in_words):
                                            common = rec_words & tts_words
                                            if rec_words and (len(common) / len(rec_words) >= 0.6):
                                                is_echo = True
                                                logger.info(
                                                    "VOICE_ECHO_FILTERED: %r (matches active TTS output)",
                                                    text,
                                                )
                                elif self.should_ignore():
                                    is_echo = True

                                if not is_echo:
                                    self.callback(text)
                        except Exception as exc:
                            logger.warning("VOICE_RECOGNITION_FAILED: %s", exc)
                    else:
                        try:
                            partial_res = json.loads(recognizer.PartialResult())
                            partial = partial_res.get("partial", "").strip()
                            if partial:
                                logger.debug("VOICE_PARTIAL_RECOGNITION: %s", partial)
                        except Exception:
                            pass

        except Exception as exc:
            logger.warning("VOICE_RECOGNITION_FAILED: %s", exc)

    def stop(self) -> None:
        """Signal the listener thread to stop and wait for it to join."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
            self.thread = None
        logger.info("Voice listener stopped cleanly")
