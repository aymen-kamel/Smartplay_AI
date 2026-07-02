"""
Legacy monolithic entrypoint (before run_analysis.py refactor).

Equivalent behaviour to calling ``run_analysis()`` from ``main.py`` with interactive keypoints.
Kept for reference and regression comparison.

Run: python old_main.py
"""

import timeit
import json
import cv2
import numpy as np
import supervision as sv

from trackers import (
    PlayerTracker,
    SAM2PlayerTracker,
    BallTracker,
    KeypointsTracker,
    Keypoint,
    Keypoints,
    PlayerKeypointsTracker,
    TrackingRunner,
)
from config import *

SELECTED_KEYPOINTS = []
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
img_display = None
w, h = 0, 0

"""
PADEL COURT KEYPOINTS

-> To be selected using the image pop-up

        k11--------------------k12
        |                       |
        k8-----------k9--------k10
        |            |          |
        |            |          |
        |            |          |
        k6----------------------k7
        |            |          |
        |            |          |
        |            |          |
        k3-----------k4---------k5
        |                       |
        k1----------------------k2

"""


def click_event(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:

        original_x = int(x * (w / DISPLAY_WIDTH))
        original_y = int(y * (h / DISPLAY_HEIGHT))

        SELECTED_KEYPOINTS.append((original_x, original_y))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_display, f"{original_x},{original_y}", (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow("frame", img_display)


if __name__ == "__main__":

    t1 = timeit.default_timer()

    video_info = sv.VideoInfo.from_video_path(video_path=INPUT_VIDEO_PATH)
    fps, w, h, total_frames = (
        video_info.fps,
        video_info.width,
        video_info.height,
        video_info.total_frames,
    )

    first_frame_generator = sv.get_video_frames_generator(
        INPUT_VIDEO_PATH,
        start=0,
        stride=1,
        end=1,
    )

    img = next(first_frame_generator)

    if FIXED_COURT_KEYPOINTS_LOAD_PATH is not None and os.path.exists(FIXED_COURT_KEYPOINTS_LOAD_PATH):
        print(f"old_main: Found existing keypoints at {FIXED_COURT_KEYPOINTS_LOAD_PATH}")
        with open(FIXED_COURT_KEYPOINTS_LOAD_PATH, "r") as f:
            SELECTED_KEYPOINTS = json.load(f)
            print(f"old_main: Auto-loaded {len(SELECTED_KEYPOINTS)} keypoints.")

    if not SELECTED_KEYPOINTS:
        print("old_main: Opening manual selection window...")
        if video_info.width < DISPLAY_WIDTH and video_info.height < DISPLAY_HEIGHT:
            DISPLAY_WIDTH = video_info.width
            DISPLAY_HEIGHT = video_info.height

        w, h = video_info.width, video_info.height
        img_display = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        cv2.imshow("frame", img_display)
        cv2.setMouseCallback("frame", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if FIXED_COURT_KEYPOINTS_SAVE_PATH is not None:
        with open(FIXED_COURT_KEYPOINTS_SAVE_PATH, "w") as f:
            json.dump(SELECTED_KEYPOINTS, f)

    fixed_keypoints_detection = Keypoints(
        [
            Keypoint(
                id=i,
                xy=tuple(float(x) for x in v),
            )
            for i, v in enumerate(SELECTED_KEYPOINTS)
        ]
    )

    keypoints_array = np.array(SELECTED_KEYPOINTS)
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

    if USE_SAM2:
        players_tracker = SAM2PlayerTracker(
            PLAYERS_TRACKER_MODEL,
            sam2_weights=SAM2_MODEL_WEIGHTS,
            sam2_config=SAM2_MODEL_CONFIG,
            polygon_zone=polygon_zone,
            batch_size=PLAYERS_TRACKER_BATCH_SIZE,
            annotator=PLAYERS_TRACKER_ANNOTATOR,
            show_confidence=True,
            load_path=PLAYERS_TRACKER_LOAD_PATH,
            save_path=PLAYERS_TRACKER_SAVE_PATH,
        )
        players_tracker.video_info_post_init(video_info)
        players_tracker.init_video_state(INPUT_VIDEO_PATH)
    else:
        players_tracker = PlayerTracker(
            PLAYERS_TRACKER_MODEL,
            polygon_zone=polygon_zone,
            batch_size=PLAYERS_TRACKER_BATCH_SIZE,
            annotator=PLAYERS_TRACKER_ANNOTATOR,
            show_confidence=True,
            load_path=PLAYERS_TRACKER_LOAD_PATH,
            save_path=PLAYERS_TRACKER_SAVE_PATH,
        )

    player_keypoints_tracker = PlayerKeypointsTracker(
        PLAYERS_KEYPOINTS_TRACKER_MODEL,
        train_image_size=PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE,
        batch_size=PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE,
        load_path=PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH,
    )

    ball_tracker = BallTracker(
        BALL_TRACKER_MODEL,
        BALL_TRACKER_INPAINT_MODEL,
        batch_size=BALL_TRACKER_BATCH_SIZE,
        median_max_sample_num=BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM,
        median=None,
        load_path=BALL_TRACKER_LOAD_PATH,
        save_path=BALL_TRACKER_SAVE_PATH,
    )

    keypoints_tracker = KeypointsTracker(
        model_path=KEYPOINTS_TRACKER_MODEL,
        batch_size=KEYPOINTS_TRACKER_BATCH_SIZE,
        model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
        fixed_keypoints_detection=None if DYNAMIC_COURT_KEYPOINTS else fixed_keypoints_detection,
        load_path=KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=KEYPOINTS_TRACKER_SAVE_PATH,
    )

    runner = TrackingRunner(
        trackers=[
            players_tracker,
            player_keypoints_tracker,
            ball_tracker,
            keypoints_tracker,
        ],
        video_path=INPUT_VIDEO_PATH,
        inference_path=OUTPUT_VIDEO_PATH,
        start=0,
        end=MAX_FRAMES,
        collect_data=COLLECT_DATA,
    )

    runner.run()

    if COLLECT_DATA:
        data = runner.data_analytics.into_dataframe(runner.video_info.fps)
        data.to_csv(COLLECT_DATA_PATH)

        summary = {
            "player_shoots": {str(k): v for k, v in runner.scorer.player_shoots.items()} if runner.scorer else {}
        }
        with open("summary.json", "w") as f:
            json.dump(summary, f)

        if runner.scorer and hasattr(runner.scorer, "reporter"):
            runner.scorer.reporter.save()

    t2 = timeit.default_timer()

    print("Duration (min): ", (t2 - t1) / 60)
