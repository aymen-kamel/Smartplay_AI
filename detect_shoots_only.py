import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from trackers.ball_tracker.ball_tracker import BallTracker, Ball
from trackers.players_keypoints_tracker.players_keypoints_tracker import PlayerKeypointsTracker
from analytics.scoring import PadelScorer
import supervision as sv

def run_shot_detection(video_path, output_path):
    # 1. Setup Models
    print("Initializing models...")
    # Use YOLO-Pose for fast player keypoints (17 COCO points)
    player_model = YOLO("weights/yolo26m-pose.pt") 
    
    # Use TrackNet for ball
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
    
    # 2. Pre-detect Ball for the whole video (more efficient for TrackNet)
    print("Tracking ball...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    ball_detections = ball_tracker.predict_frames(frames, total_frames=len(frames))
    
    # 3. Process Video for Players & Shoots
    print("Analyzing shoots...")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), video_info.fps, (video_info.width, video_info.height))
    
    for frame_idx, frame in enumerate(tqdm(frames)):
        # A. Detect Players & Keypoints
        results = player_model.predict(frame, conf=0.3, verbose=False)[0]
        players_kp = {}
        players_pos = {}
        
        if results.keypoints is not None:
            for i, kp_data in enumerate(results.keypoints.xy.cpu().numpy()):
                pid = i + 1
                players_kp[pid] = kp_data 
                players_pos[pid] = kp_data[0] # Nose

        # B. Get Pre-detected Ball
        ball_pos = None
        is_bounce = False
        if frame_idx < len(ball_detections):
            ball_data = ball_detections[frame_idx]
            if ball_data.visibility > 0.5:
                ball_pos = ball_data.xy
                is_bounce = ball_data.is_bounce

        # C. Scoring & Shoot Detection
        # We need to map pixel coords to meters for our Scorer (simplified here)
        # If no calibration, we use a rough scale
        scale = 0.02 # approx meters per pixel
        ball_pos_m = (ball_pos[0]*scale, ball_pos[1]*scale) if ball_pos else None
        players_pos_m = {pid: (pos[0]*scale, pos[1]*scale) for pid, pos in players_pos.items()}

        scorer.process_frame(
            frame_idx=frame_idx,
            ball_pos_m=ball_pos_m,
            is_bounce=is_bounce,
            players_pos_m=players_pos_m
        )

        # D. Visualization
        # Draw counts above players
        for pid, pos in players_pos.items():
            count = scorer.player_shoots.get(pid, 0)
            cv2.putText(frame, f"SHOTS: {count}", (int(pos[0]), int(pos[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw Ball
        if ball_pos:
            cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 5, (0, 0, 255), -1)

        out.write(frame)
        frame_idx += 1

    out.release()
    print(f"Done! Results saved in {output_path}")

if __name__ == "__main__":
    run_shot_detection("input/rally.mp4", "output_shoots.mp4")
