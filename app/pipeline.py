"""Application orchestration for camera, perception, conservative guidance, and voice."""
import logging
import queue
import time

import cv2

from camera_input.camera import CameraStream
from navigation.navigation_engine import NavigationEngine
from navigation.target_guidance import TargetGuidance
from navigation.hazard_tracker import HazardTracker
from perception.depth_estimator import DepthEstimator, create_depth_estimator
from perception.depth_worker import DepthWorker
from perception.detector import YOLODetector
from perception.tracker import SimpleTracker
from spatial.free_space import FreeSpaceAnalyzer
from spatial.world_model import WorldModel
from voice.command_parser import CommandParser
from voice.text_to_speech import SpeechOutput, TTSPriority


def run_app(cfg, camera_factory=None, detector=None, depth_estimator=None, speech=None,
            voice_listener_factory=None, scene_worker=None, max_frames=None, display=True):
    """Run the app. Dependency arguments are primarily for deterministic smoke tests."""
    logging.basicConfig(
        level=getattr(logging, str(cfg.get("app", {}).get("log_level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    camera_cfg = cfg["camera"]
    perception_cfg = cfg["perception"]
    voice_cfg = cfg["voice"]
    navigation_cfg = cfg["navigation"]

    if detector is None:
        detector = YOLODetector(
            perception_cfg.get("detector_model", "yolov8n.pt"),
            perception_cfg.get("confidence", 0.45),
            perception_cfg.get("image_size", 640),
            perception_cfg.get("device", "auto"),
            perception_cfg.get("detector_backend", "yolo"),
            perception_cfg.get("world_model", "yolov8s-worldv2.pt"),
            perception_cfg.get("world_classes", []),
        )

    estimator = None
    depth_worker = None
    if perception_cfg.get("enable_depth", True):
        try:
            estimator = depth_estimator or create_depth_estimator(cfg)
            depth_worker = DepthWorker(estimator)
        except Exception as exc:
            logging.warning("Depth disabled: %s", exc)

    sd_cfg = cfg.get("scene_description", {})
    if scene_worker is None and sd_cfg.get("enabled", True):
        try:
            from scene_description.scene_service import SceneService
            from scene_description.scene_worker import SceneWorker
            scene_service = SceneService.from_config(cfg)
            scene_worker = SceneWorker(
                service=scene_service,
                jpeg_quality=sd_cfg.get("jpeg_quality", 80),
            )
        except Exception as exc:
            logging.warning("Scene description disabled: %s", exc)

    tracker = SimpleTracker() if perception_cfg.get("enable_tracking", True) else None
    world = WorldModel()
    free_space = FreeSpaceAnalyzer(
        navigation_cfg.get("near_relative_depth_threshold", 0.72),
        navigation_cfg.get("large_box_area_threshold", 0.12),
        navigation_cfg.get("corridor_box_area_threshold", 0.025),
        navigation_cfg.get("corridor_y_start", 0.48),
        perception_cfg.get("metric_warning_distance_m", 1.5),
        perception_cfg.get("metric_critical_distance_m", 0.8),
    )
    navigation = NavigationEngine()
    speech = speech or SpeechOutput(
        bool(voice_cfg.get("enabled", True) and voice_cfg.get("tts_enabled", True)),
        voice_cfg.get("tts_rate", 175),
    )
    parser = CommandParser()
    target_guidance = TargetGuidance(navigation_cfg.get("target_guidance_interval_seconds", 2.0))
    hazard_tracker = HazardTracker()
    commands = queue.Queue()
    last_spoken = ""
    last_spoken_at = 0.0
    last_response = ""
    guidance_muted = False
    frame_number = 0
    latest_depth = None
    latest_depth_id = -1
    latest_depth_completed_at = 0.0
    depth_latency = 0.0
    max_depth_age_seconds = 0.75
    fps_started = time.perf_counter()
    fps_count = 0
    fps = 0.0
    previous_depth_error = None

    def on_voice(command_text):
        # The listener independently checks this gate at capture and recognition time.
        if getattr(speech, "is_speaking", False):
            return
        logging.info("Voice command: %s", command_text)
        commands.put(command_text)

    listener = None
    if voice_cfg.get("enabled") and voice_cfg.get("speech_recognition") == "vosk":
        try:
            if voice_listener_factory is None:
                from voice.speech_recognition import VoiceListener
                voice_listener_factory = VoiceListener
            listener = voice_listener_factory(
                voice_cfg.get("vosk_model_path", "models/vosk-model-small-en-us-0.15"),
                on_voice,
                should_ignore=lambda: bool(getattr(speech, "is_speaking", False)),
            )
            listener.start()
        except Exception as exc:
            logging.warning("Voice recognition disabled: %s", exc)

    camera_factory = camera_factory or CameraStream
    source = camera_cfg["source"]
    camera = camera_factory(
        source,
        camera_cfg["width"],
        camera_cfg["height"],
        camera_cfg.get("reconnect_attempts", 3),
        camera_cfg.get("reconnect_delay_seconds", 1.0),
    )

    try:
        with camera as stream:
            while max_frames is None or frame_number < max_frames:
                ok, frame = stream.read()
                if not ok or frame is None:
                    logging.error("Camera read failed. Press q to exit.")
                    if display:
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    time.sleep(0.05)
                    continue

                frame_number += 1
                fps_count += 1
                now_perf = time.perf_counter()
                if now_perf - fps_started >= 1.0:
                    fps = fps_count / (now_perf - fps_started)
                    fps_count = 0
                    fps_started = now_perf

                total_started = time.perf_counter()
                detections, detector_latency = detector.predict(frame)
                depth_every = max(1, int(perception_cfg.get("depth_every_n_frames", 3)))
                if depth_worker and (frame_number % depth_every == 0 or latest_depth is None):
                    depth_worker.submit(frame, frame_number)

                if depth_worker:
                    (
                        new_depth,
                        new_latency,
                        new_depth_id,
                        depth_error,
                        new_completed_at,
                    ) = depth_worker.snapshot()

                    if new_depth is not None and new_depth_id != latest_depth_id:
                        latest_depth = new_depth
                        latest_depth_id = new_depth_id
                        latest_depth_completed_at = new_completed_at
                        depth_latency = new_latency

                    if depth_error and depth_error != previous_depth_error:
                        logging.warning(
                            "Depth unavailable for this update: %s",
                            depth_error,
                        )
                    previous_depth_error = depth_error

                depth_age_seconds = (
                    time.perf_counter() - latest_depth_completed_at
                    if latest_depth_completed_at > 0
                    else float("inf")
                )

                depth_is_fresh = (
                    latest_depth is not None
                    and depth_age_seconds <= max_depth_age_seconds
                )

                if depth_is_fresh and estimator is not None:
                    estimator.attach_depth(detections, latest_depth)
                else:
                    # Do not use stale depth for navigation decisions.
                    for detection in detections:
                        if hasattr(detection, "relative_depth"):
                            detection.relative_depth = None

                tracker_started = time.perf_counter()
                if tracker:
                    detections = tracker.update(detections)
                tracker_latency = time.perf_counter() - tracker_started
                scene = world.build(detections, frame.shape)
                spaces = free_space.analyze(detections, frame.shape)
                decision = navigation.decide(spaces)
                spatial_latency = time.perf_counter() - tracker_started - tracker_latency

                if scene_worker:
                    sd_res = scene_worker.poll_result()
                    if sd_res and sd_res.get("text"):
                        last_response = sd_res["text"]
                        speech.say(sd_res["text"])

                while True:
                    try:
                        raw_command = commands.get_nowait()
                    except queue.Empty:
                        break

                    command = parser.parse(raw_command)
                    intent = command["intent"]
                    response = ""

                    if intent == "scene":
                        if scene_worker:
                            accepted = scene_worker.request(frame, detections, {"free_space": spaces})
                            if accepted:
                                response = "Checking surroundings."
                            else:
                                response = "Still analyzing the previous scene request."
                        else:
                            objects = ", ".join(
                                f"{obj.label} on the {obj.direction}"
                                for obj in scene["objects"][:6]
                            )
                            response = (
                                "Detected: "
                                + (objects if objects else "no objects confidently detected")
                                + ". "
                                + decision["text"]
                            )
                        last_response = response
                        speech.say(response)
                        continue

                    elif intent == "distance":
                        # Prefer calibrated metric distances if available
                        metric_objects = [
                            obj
                            for obj in scene["objects"]
                            if getattr(obj, "distance_m", None) is not None
                        ]

                        if metric_objects:
                            nearest = min(metric_objects, key=lambda o: o.distance_m)
                            response = (
                                f"The nearest object is {nearest.label}, "
                                f"approximately {nearest.distance_m:.2f} metres away."
                            )
                        else:
                            known = [
                                obj
                                for obj in scene["objects"]
                                if obj.relative_depth is not None
                            ]

                            if known:
                                nearest = max(
                                    known,
                                    key=lambda obj: obj.relative_depth,
                                )
                                response = (
                                    "I can provide relative depth only, not a verified "
                                    "distance in metres. "
                                    f"Nearest-looking detected object: {nearest.label}."
                                )
                            else:
                                response = "I cannot estimate object depth right now."

                    elif intent == "find":
                        requested_target = command.get("target", "")
                        target_supported = True
                        ensure_target = getattr(
                            detector,
                            "ensure_target_class",
                            None,
                        )

                        if ensure_target is not None:
                            try:
                                target_supported = bool(
                                    ensure_target(requested_target)
                                )
                            except Exception as exc:
                                logging.warning(
                                    "Target class activation failed: %s",
                                    exc,
                                )
                                target_supported = False

                        if not target_supported:
                            target_guidance.stop()
                            response = (
                                f"I cannot search for {requested_target} "
                                "with the active detector. "
                                "It is not in the supported COCO classes "
                                "and YOLO-World is unavailable."
                            )
                        else:
                            target_guidance.start(requested_target)
                            response = (
                                f"Searching the current camera view for "
                                f"{target_guidance.target}. Turn slowly as I scan."
                            )

                    elif intent == "stop":
                        guidance_muted = True
                        target_guidance.stop()
                        response = "Voice guidance muted. Visual processing continues."

                    elif intent == "resume":
                        guidance_muted = False
                        response = "Voice guidance resumed."

                    elif intent == "repeat":
                        response = last_response or decision["text"]

                    else:
                        response = (
                            "Sorry, I did not understand. Try asking what is ahead, "
                            "how far the nearest object is, or find a chair."
                        )

                    last_response = response
                    speech.say(response)

                if target_guidance.state != "idle":
                    guidance_now = time.monotonic()
                    centre_blocked = spaces.get("regions", {}).get("centre") == "blocked"
                    target_message = target_guidance.update(
                        scene,
                        guidance_now,
                        centre_blocked=centre_blocked,
                    )

                    if target_message and not guidance_muted:
                        speech.say(
                            target_message,
                            urgent=(
                                target_guidance.state == "lost"
                                or centre_blocked
                            ),
                        )
                        target_guidance.mark_spoken(guidance_now)
                        last_response = target_message

                now = time.monotonic()
                should_announce_warning, hazard_priority, hazard_reason = hazard_tracker.evaluate(
                    decision,
                    now,
                )

                if not guidance_muted and should_announce_warning:
                    speech.say(
                        decision["text"],
                        urgent=hazard_priority >= TTSPriority.HIGH,
                    )
                    last_spoken = decision["text"]
                    last_spoken_at = now

                total_latency = time.perf_counter() - total_started

                if display:
                    view = detector.annotate(frame.copy(), detections)
                    depth_label = (
                        f"depth {depth_latency * 1000:.0f} ms (async)"
                        if depth_worker
                        else "depth off"
                    )

                    cv2.putText(
                        view,
                        f"FPS {fps:.1f} | det {detector_latency * 1000:.0f} ms | {depth_label}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        view,
                        (
                            f"track {tracker_latency * 1000:.1f} ms | "
                            f"spatial {spatial_latency * 1000:.1f} ms | "
                            f"loop {total_latency * 1000:.0f} ms"
                        ),
                        (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        view,
                        decision["text"][:95],
                        (10, view.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 200, 255),
                        2,
                    )

                    cv2.imshow(
                        "Depth-Aware Navigation Assistant (prototype)",
                        view,
                    )

                    if latest_depth is not None and cfg["app"].get(
                        "show_depth_window",
                        True,
                    ):
                        cv2.imshow(
                            "Relative depth (not metres)",
                            DepthEstimator.colorize(latest_depth),
                        )
                        # Show calibrated metric depth if available
                        if getattr(estimator, "last_metric_map", None) is not None:
                            # create a copy to overlay text
                            metric_vis = DepthEstimator.colorize(estimator.last_metric_map).copy()
                            h, w = estimator.last_metric_map.shape
                            cx, cy = w // 2, h // 2
                            center_val = estimator.last_metric_map[cy, cx]
                            cv2.putText(
                                metric_vis,
                                f"Center depth: {center_val:.2f} m",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                            )
                            cv2.imshow(
                                "Metric depth (metres)",
                                metric_vis,
                            )

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break

                    if key == ord("s"):
                        speech.say(decision["text"], urgent=urgent)

                    if key == ord("i"):
                        labels = ", ".join(
                            obj.label for obj in scene["objects"][:6]
                        )
                        speech.say(
                            "Detected: "
                            + (labels if labels else "no objects detected")
                        )

                    if key == ord("d"):
                        if scene_worker:
                            accepted = scene_worker.request(frame, detections, {"free_space": spaces})
                            if accepted:
                                speech.say("Checking surroundings.")
                            else:
                                speech.say("Still analyzing the previous scene request.")
                        else:
                            labels = ", ".join(obj.label for obj in scene["objects"][:6])
                            speech.say("Detected: " + (labels if labels else "no objects detected"))

    finally:
        if listener:
            listener.stop()

        if depth_worker:
            depth_worker.close()

        if scene_worker:
            scene_worker.close()

        speech.close()

        if display:
            cv2.destroyAllWindows()




