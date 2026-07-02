"""
Reusable padel video analysis pipeline for main.py and api_server.

Court keypoints: resolve from JSON file, interactive OpenCV selection, or pass
``court_keypoints_xy`` / per-job JSON via the API.
"""

from __future__ import annotations

import json
import os
import timeit
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

import config
from trackers import (
    BallTracker,
    Keypoint,
    Keypoints,
    KeypointsTracker,
    PlayerKeypointsTracker,
    PlayerTracker,
    SAM2PlayerTracker,
    TrackingRunner,
)

EXPECTED_COURT_KEYPOINTS = 12


def normalize_court_keypoints_xy(raw: List[Any]) -> List[Tuple[float, float]]:
    """Validate JSON-loaded [[x,y], ...] into exactly 12 pixel coordinates."""
    if len(raw) != EXPECTED_COURT_KEYPOINTS:
        raise ValueError(
            f"Court keypoints: expected {EXPECTED_COURT_KEYPOINTS} points, got {len(raw)}"
        )
    out: List[Tuple[float, float]] = []
    for i, p in enumerate(raw):
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            raise ValueError(f"Court keypoints: invalid point at index {i}")
        out.append((float(p[0]), float(p[1])))
    return out


def _interactive_select_keypoints(
    video_path: str,
    display_width: int,
    display_height: int,
) -> List[Tuple[float, float]]:
    selected: List[Tuple[float, float]] = []
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    w_img, h_img = video_info.width, video_info.height
    gen = sv.get_video_frames_generator(video_path, start=0, stride=1, end=1)
    img = next(gen)

    dw, dh = display_width, display_height
    if video_info.width < dw and video_info.height < dh:
        dw, dh = video_info.width, video_info.height

    img_display = cv2.resize(img, (dw, dh))

    def click_event(event, x, y, flags, params):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        original_x = int(x * (w_img / dw))
        original_y = int(y * (h_img / dh))
        selected.append((original_x, original_y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_display, f"{original_x},{original_y}", (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow("frame", img_display)

    cv2.imshow("frame", img_display)
    cv2.setMouseCallback("frame", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return selected


def resolve_court_keypoints(
    video_path: str,
    *,
    keypoints_load_path: Optional[str],
    keypoints_save_path: Optional[str],
    allow_interactive: bool,
    persist_keypoints: bool = True,
    display_width: int = 1280,
    display_height: int = 720,
) -> List[Tuple[float, float]]:
    load_path = keypoints_load_path if keypoints_load_path is not None else config.FIXED_COURT_KEYPOINTS_LOAD_PATH
    save_path = keypoints_save_path if keypoints_save_path is not None else config.FIXED_COURT_KEYPOINTS_SAVE_PATH

    selected: List[Tuple[float, float]] = []
    if load_path is not None and os.path.exists(load_path):
        print(f"run_analysis: Loaded keypoints from {load_path}")
        with open(load_path, "r") as f:
            raw = json.load(f)
            selected = [(float(x), float(y)) for x, y in raw]

    if not selected:
        if not allow_interactive:
            raise FileNotFoundError(
                "Court keypoints are required for non-interactive runs. "
                f"Create {load_path} (12 points) via main.py once or manual_keypoints_selection.py."
            )
        print("run_analysis: Opening manual keypoints selection...")
        selected = _interactive_select_keypoints(video_path, display_width, display_height)

    if persist_keypoints and save_path is not None and selected:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump([[float(x), float(y)] for x, y in selected], f)

    return selected


def run_analysis(
    input_video_path: str,
    output_video_path: str,
    *,
    collect_data: Optional[bool] = None,
    collect_data_path: Optional[str] = None,
    summary_json_path: Optional[str] = None,
    shot_events_json_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    allow_interactive_keypoints: bool = True,
    keypoints_load_path: Optional[str] = None,
    keypoints_save_path: Optional[str] = None,
    persist_keypoints: bool = True,
    court_keypoints_xy: Optional[List[Tuple[float, float]]] = None,
) -> dict[str, Any]:
    """
    Build trackers from config, run TrackingRunner, optionally export CSV and summary JSON.

    If ``court_keypoints_xy`` is set (12 points in pixel coords), it overrides file load / interactive UI.

    Returns metadata dict with paths and timing.
    """
    t1 = timeit.default_timer()

    collect_data = config.COLLECT_DATA if collect_data is None else collect_data
    collect_data_path = collect_data_path or config.COLLECT_DATA_PATH
    max_frames = config.MAX_FRAMES if max_frames is None else max_frames

    if court_keypoints_xy is not None:
        selected_keypoints = normalize_court_keypoints_xy(list(court_keypoints_xy))
    else:
        selected_keypoints = resolve_court_keypoints(
            input_video_path,
            keypoints_load_path=keypoints_load_path,
            keypoints_save_path=keypoints_save_path,
            allow_interactive=allow_interactive_keypoints,
            persist_keypoints=persist_keypoints,
        )

    fixed_keypoints_detection = Keypoints(
        [
            Keypoint(id=i, xy=tuple(float(x) for x in v))
            for i, v in enumerate(selected_keypoints)
        ]
    )

    keypoints_array = np.array(selected_keypoints)
    polygon_zone = sv.PolygonZone(
        polygon=np.concatenate(
            (
                np.expand_dims(keypoints_array[0], axis=0),
                np.expand_dims(keypoints_array[1], axis=0),
                np.expand_dims(keypoints_array[-1], axis=0),
                np.expand_dims(keypoints_array[-2], axis=0),
            ),
            axis=0,
        )
    )

    video_info = sv.VideoInfo.from_video_path(video_path=input_video_path)

    if config.USE_SAM2:
        players_tracker = SAM2PlayerTracker(
            config.PLAYERS_TRACKER_MODEL,
            sam2_weights=config.SAM2_MODEL_WEIGHTS,
            sam2_config=config.SAM2_MODEL_CONFIG,
            polygon_zone=polygon_zone,
            batch_size=config.PLAYERS_TRACKER_BATCH_SIZE,
            annotator=config.PLAYERS_TRACKER_ANNOTATOR,
            show_confidence=True,
            load_path=config.PLAYERS_TRACKER_LOAD_PATH,
            save_path=config.PLAYERS_TRACKER_SAVE_PATH,
        )
        players_tracker.video_info_post_init(video_info)
        players_tracker.init_video_state(input_video_path)
    else:
        players_tracker = PlayerTracker(
            config.PLAYERS_TRACKER_MODEL,
            polygon_zone=polygon_zone,
            batch_size=config.PLAYERS_TRACKER_BATCH_SIZE,
            annotator=config.PLAYERS_TRACKER_ANNOTATOR,
            show_confidence=True,
            load_path=config.PLAYERS_TRACKER_LOAD_PATH,
            save_path=config.PLAYERS_TRACKER_SAVE_PATH,
        )

    player_keypoints_tracker = PlayerKeypointsTracker(
        config.PLAYERS_KEYPOINTS_TRACKER_MODEL,
        train_image_size=config.PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE,
        batch_size=config.PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE,
        load_path=config.PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=config.PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH,
    )

    ball_tracker = BallTracker(
        config.BALL_TRACKER_MODEL,
        config.BALL_TRACKER_INPAINT_MODEL,
        batch_size=config.BALL_TRACKER_BATCH_SIZE,
        median_max_sample_num=config.BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM,
        median=None,
        load_path=config.BALL_TRACKER_LOAD_PATH,
        save_path=config.BALL_TRACKER_SAVE_PATH,
    )

    keypoints_tracker = KeypointsTracker(
        model_path=config.KEYPOINTS_TRACKER_MODEL,
        batch_size=config.KEYPOINTS_TRACKER_BATCH_SIZE,
        model_type=config.KEYPOINTS_TRACKER_MODEL_TYPE,
        fixed_keypoints_detection=None if config.DYNAMIC_COURT_KEYPOINTS else fixed_keypoints_detection,
        load_path=config.KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=config.KEYPOINTS_TRACKER_SAVE_PATH,
    )

    scorer_reporter_path = None
    if collect_data and shot_events_json_path:
        scorer_reporter_path = shot_events_json_path
        os.makedirs(os.path.dirname(scorer_reporter_path) or ".", exist_ok=True)

    runner = TrackingRunner(
        trackers=[
            players_tracker,
            player_keypoints_tracker,
            ball_tracker,
            keypoints_tracker,
        ],
        video_path=input_video_path,
        inference_path=output_video_path,
        start=0,
        end=max_frames,
        collect_data=collect_data,
        scorer_reporter_path=scorer_reporter_path,
    )

    runner.run()

    result: dict[str, Any] = {
        "input_video_path": input_video_path,
        "output_video_path": output_video_path,
        "duration_min": (timeit.default_timer() - t1) / 60.0,
        "collect_data_path": None,
        "summary_json_path": None,
        "shot_events_json_path": None,
    }

    if collect_data:
        data = runner.data_analytics.into_dataframe(runner.video_info.fps)
        os.makedirs(os.path.dirname(collect_data_path) or ".", exist_ok=True)
        data.to_csv(collect_data_path)
        result["collect_data_path"] = collect_data_path

        summary_path = summary_json_path or "summary.json"
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        summary = {
            "player_shoots": {
                str(k): v for k, v in runner.scorer.player_shoots.items()
            } if runner.scorer else {},
            "score_top": runner.scorer.score_top if runner.scorer else 0,
            "score_bottom": runner.scorer.score_bottom if runner.scorer else 0,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f)
        result["summary_json_path"] = summary_path

        if runner.scorer and hasattr(runner.scorer, "reporter"):
            runner.scorer.reporter.save()
        if scorer_reporter_path:
            result["shot_events_json_path"] = scorer_reporter_path

    print(f"run_analysis: Duration (min): {result['duration_min']:.2f}")
    return result
