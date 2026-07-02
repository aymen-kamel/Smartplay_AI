import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import sys
import os

# Ensure paths are correct
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../The_best')))

from trackers.ball_tracker.ball_tracker import BallTracker
from analytics.scoring import PadelScorer
from paddle_rally_extractor import PaddleRallyExtractor
import supervision as sv

def analyze_match(video_path, output_json_path):
    print("1. Fast Rally Extraction (Skipping dead time)...")
    extractor = PaddleRallyExtractor(video_path)
    rallies = extractor.extract_rallies()
    
    if not rallies:
        print("No rallies detected.")
        return

    print("\n2. Initializing AI Trackers for Shot Detection...")
    player_model = YOLO("weights/yolo26m-pose.pt") 
    ball_tracker = BallTracker(
        tracking_model_path="weights/ball_detection.pt",
        inpainting_model_path="weights/InpaintNet_best.pt", 
        batch_size=8
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ball_tracker.to(device)
    
    scorer = PadelScorer()
    video_info = sv.VideoInfo.from_video_path(video_path)
    ball_tracker.video_info_post_init(video_info)
    
    cap = cv2.VideoCapture(video_path)
    
    print("\n3. Analyzing Shots within Rallies...")
    for rally_idx, (start_frame, end_frame) in enumerate(rallies):
        print(f"--- Processing Rally {rally_idx+1}/{len(rallies)} (Frames: {start_frame} to {end_frame}) ---")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
            
        if not frames: continue
        
        # Track ball for this segment
        ball_detections = ball_tracker.predict_frames(frames, total_frames=len(frames))
        
        # Process frames for shots
        for local_idx, frame in enumerate(tqdm(frames)):
            global_frame_idx = start_frame + local_idx
            
            # Players
            results = player_model.predict(frame, conf=0.3, verbose=False)[0]
            players_kp = {}
            players_pos = {}
            if results.keypoints is not None:
                for i, kp_data in enumerate(results.keypoints.xy.cpu().numpy()):
                    pid = i + 1
                    players_kp[pid] = kp_data 
                    players_pos[pid] = kp_data[0] # Nose
                    
            # Ball
            ball_pos = None
            is_bounce = False
            if local_idx < len(ball_detections):
                ball_data = ball_detections[local_idx]
                if ball_data.visibility > 0.5:
                    ball_pos = ball_data.xy
                    is_bounce = ball_data.is_bounce

            scale = 0.02
            ball_pos_m = (ball_pos[0]*scale, ball_pos[1]*scale) if ball_pos else None
            players_pos_m = {pid: (pos[0]*scale, pos[1]*scale) for pid, pos in players_pos.items()}

            scorer.process_frame(
                frame_idx=global_frame_idx,
                ball_pos_m=ball_pos_m,
                is_bounce=is_bounce,
                players_pos_m=players_pos_m,
                players_kp=players_kp,
                ball_xy=ball_pos,
                players_xy=players_pos
            )

    cap.release()
    
    # Override output path and save
    scorer.reporter.output_path = output_json_path
    scorer.reporter.save()
    print(f"\nDone! Shot analysis saved to {output_json_path}")

if __name__ == "__main__":
    analyze_match("../../The_best/vedio.mp4", "cache/full_match_shots.json")
