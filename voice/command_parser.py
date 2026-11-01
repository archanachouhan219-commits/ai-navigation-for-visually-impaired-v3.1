"""Deterministic voice command parser; offline regex matching with no external LLM dependencies."""
from __future__ import annotations

import re
from typing import Any, Dict


class CommandParser:
    """Extracts navigation and accessibility intents from transcribed text utterances."""

    def parse(self, text: str) -> Dict[str, Any]:
        raw = (text or "").lower().replace("'", "")
        normalized = re.sub(r"[^a-z0-9 ]", " ", raw).strip()
        normalized = " ".join(normalized.split())
        if not normalized:
            return {"intent": "unknown"}

        tokens = set(normalized.split())

        # Exact or phrase stop/pause/quiet/cancel intents
        stop_phrases = (
            "stop guidance", "stop speaking", "be quiet", "stop navigation",
            "cancel find", "stop finding", "pause guidance", "cancel guidance",
            "quiet please", "silence",
        )
        stop_words = {"stop", "pause", "quiet", "cancel", "mute", "silence"}

        if tokens & stop_words or any(phrase in normalized for phrase in stop_phrases):
            # Special case: if user said 'unmute', that's resume, not stop
            if "unmute" not in tokens:
                action = "stop"
                if "cancel" in tokens:
                    action = "cancel"
                elif "pause" in tokens:
                    action = "pause"
                elif "quiet" in tokens or "silence" in tokens:
                    action = "quiet"
                return {"intent": "stop", "action": action}

        # Resume / unmute intents
        resume_phrases = (
            "resume guidance", "start guidance", "continue guidance",
            "resume navigation", "start navigation", "unmute guidance",
        )
        resume_words = {"resume", "continue", "unmute"}
        if tokens & resume_words or any(phrase in normalized for phrase in resume_phrases):
            return {"intent": "resume"}

        # Repeat last utterance
        repeat_phrases = (
            "say again", "repeat that", "what did you say", "repeat last",
            "say that again", "tell me again",
        )
        if "repeat" in tokens or any(phrase in normalized for phrase in repeat_phrases):
            return {"intent": "repeat"}

        # Scene description queries
        scene_phrases = (
            "path clear", "obstacle ahead", "in front of me", "what is ahead",
            "what is in front", "describe the scene", "describe scene",
            "what is around me", "what is around", "tell me what you see",
            "what do you see", "describe surroundings", "describe the surroundings",
            "describe my surroundings", "describe room", "describe the room",
            "what do i see", "whats ahead", "whats in front", "whats around",
            "can i walk", "look ahead",
        )
        if any(phrase in normalized for phrase in scene_phrases):
            return {"intent": "scene"}

        # Distance queries
        distance_phrases = (
            "how far", "distance to", "nearest object", "how close",
            "nearest obstacle", "how close is it", "how far is the obstacle",
            "how close is the obstacle", "distance",
        )
        if any(phrase in normalized for phrase in distance_phrases):
            return {"intent": "distance"}

        # Find target object queries
        match = re.search(
            r"(?:find|locate|where is|where are|look for|search for) (?:my |the |a |an )?(.+)",
            normalized,
        )
        if match:
            target = match.group(1).strip()
            if target:
                return {"intent": "find", "target": target}

        return {"intent": "unknown", "text": normalized}
