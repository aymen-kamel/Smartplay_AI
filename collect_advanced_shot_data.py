import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from copy import deepcopy
import json

from trackers import (
    SAM2PlayerTracker, PlayerTracker, BallTracker, 
    KeypointsTracker, PlayerKeypointsTracker, TrackingRunner,
    Keypoint, Keypoints
)
from trackers.players_tracker.players_tracker import Players
from trackers.ball_tracker.ball_tracker import Ball
from trackers.players_keypoints_tracker.players_keypoints_tracker import PlayersKeypoints
from analytics.advanced_features import BiomechanicalFeatureExtractor
from config import *

def main():
    video_info = sv.VideoInfo.from_video_path(video_path=INPUT_VIDEO_PATH)
    
    # Load fixed keypoints
    with open(FIXED_COURT_KEYPOINTS_SAVE_PATH, "r") as f:
        selected_kp = json.load(f)
    
    fixed_keypoints_detection = Keypoints([
        Keypoint(id=i, xy=tuple(float(x) for x in v))
        for i, v in enumerate(selected_kp)
    ])

    # Polygons
    keypoints_array = np.array(selected_kp)
    polygon_zone = sv.PolygonZone(
        polygon=np.array([keypoints_array[0], keypoints_array[1], keypoints_array[-1], keypoints_array[-2]])
    )

    # Trackers
    players_tracker = SAM2PlayerTracker(PLAYERS_TRACKER_MODEL, sam2_weights=SAM2_MODEL_WEIGHTS, sam2_config=None, polygon_zone=polygon_zone, batch_size=PLAYERS_TRACKER_BATCH_SIZE)
    ball_tracker = BallTracker(BALL_TRACKER_MODEL, BALL_TRACKER_INPAINT_MODEL, batch_size=BALL_TRACKER_BATCH_SIZE)
    player_kp_tracker = PlayerKeypointsTracker(PLAYERS_KEYPOINTS_TRACKER_MODEL, train_image_size=1280, batch_size=8, load_path=None, save_path=None)
    keypoints_tracker = KeypointsTracker(model_path=KEYPOINTS_TRACKER_MODEL, batch_size=8, model_type="yolo", fixed_keypoints_detection=fixed_keypoints_detection)

    runner = TrackingRunner(
        trackers=[players_tracker, ball_tracker, player_kp_tracker, keypoints_tracker],
        video_path=INPUT_VIDEO_PATH,
        inference_path="shot_collection_debug.mp4",
        collect_data=True
    )

    print("Running tracking for shot data collection...")
    runner.run()

    # Feature Extraction Phase
    extractor = BiomechanicalFeatureExtractor(window_size=15)
    
    ball_results = runner.trackers['ball_tracker'].results
    player_kp_results = runner.trackers['players_keypoints_tracker'].results
    player_results = runner.trackers['sam2_players_tracker'].results if USE_SAM2 else runner.trackers['players_tracker'].results
    
    # Detect hits (simple logic)
    print("Extracting features from hits...")
    potential_changes = 0
    hits_found = 0
    
    for frame_idx in range(10, len(ball_results) - 11):
        ball_det = ball_results[frame_idx]
        if not ball_det.xy: continue
        
        b_prev = ball_results[frame_idx-1].xy
        b_curr = ball_det.xy
        b_next = ball_results[frame_idx+1].xy
        
        if b_prev and b_curr and b_next:
            v1_y = b_curr[1] - b_prev[1]
            v2_y = b_next[1] - b_curr[1]
            
            # Look for direction change in Y
            if v1_y * v2_y <= 0 and abs(v1_y - v2_y) > 1:
                potential_changes += 1
                
                # Check proximity to players
                found_player_for_hit = False
                for player in player_results[frame_idx].players:
                    # distance to player center
                    p_center = ((player.xyxy[0] + player.xyxy[2])/2, (player.xyxy[1] + player.xyxy[3])/2)
                    dist = np.hypot(b_curr[0] - p_center[0], b_curr[1] - p_center[1])
                    
                    if dist < 400: # Increased threshold for extraction
                        # Extract window of data
                        ball_window = []
                        kp_window = []
                        pos_m_window = []
                        
                        for i in range(frame_idx - 15, frame_idx + 16):
                            ball_window.append({'xy': ball_results[i].xy})
                            
                            kps = {}
                            if i < len(player_kp_results):
                                for pk in player_kp_results[i].players_keypoints:
                                    ref = pk.keypoints_by_name.get('neck') or pk.keypoints_by_name.get('torso')
                                    if ref:
                                        # Match by proximity to player box center
                                        d = np.hypot(ref.xy[0] - p_center[0], ref.xy[1] - p_center[1])
                                        if d < 300:
                                            kps[player.id] = pk
                            kp_window.append(kps)
                            
                            pos_m = {}
                            if i < len(player_results) and player_results[i]:
                                for p in player_results[i].players:
                                    if p.projection:
                                        pos_m[p.id] = runner.projected_court.court_keypoints.shift_point_origin(p.projection, dimension="meters")
                            pos_m_window.append(pos_m)

                        feat = extractor.extract_features(frame_idx, player.id, ball_window, kp_window, pos_m_window)
                        if feat:
                            extractor.dataset.append(feat)
                            hits_found += 1
                            found_player_for_hit = True
                            break
                
                if found_player_for_hit:
                    continue

    print(f"Finished. Potential trajectory changes: {potential_changes}, Hits captured: {hits_found}")
    extractor.save_to_csv("advanced_shot_features.csv")

if __name__ == "__main__":
    main()
