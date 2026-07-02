import os
import time
import json
import sys
import numpy as np
import cv2
import torch

# Ensure we can load local trackers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trackers.ball_tracker.ball_tracker import get_model
from ultralytics import YOLO

def benchmark():
    print("=" * 60)
    print("SmartPlay Pipeline CPU Benchmark Tool")
    print("=" * 60)
    
    device = "cpu"
    print(f"Device: CPU")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python: {sys.executable}")
    
    video_path = "input/rally_001.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return
        
    print(f"Using video: {video_path}")
    
    # 1. Load models
    print("\n[1/3] Loading models into CPU memory...")
    t_start = time.time()
    
    models = {}
    
    # Player detection
    try:
        yolo_player_path = "weights/yolo26m.pt"
        print(f"  Loading Player Detector ({yolo_player_path})...")
        t0 = time.time()
        models["player"] = YOLO(yolo_player_path)
        print(f"    Loaded in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"  Error loading player detector: {e}")
        
    # Player keypoints
    try:
        yolo_kpts_path = "weights/player_keypoints_detection.pt"
        print(f"  Loading Player Keypoints Detector ({yolo_kpts_path})...")
        t0 = time.time()
        models["pose"] = YOLO(yolo_kpts_path)
        print(f"    Loaded in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"  Error loading player keypoints: {e}")
        
    # Court keypoints
    try:
        court_path = "weights/court_keypoints_detection.pt"
        print(f"  Loading Court Keypoints Detector ({court_path})...")
        t0 = time.time()
        models["court"] = YOLO(court_path)
        print(f"    Loaded in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"  Error loading court keypoints: {e}")
        
    # TrackNet Ball detection
    try:
        ball_path = "weights/ball_detection.pt"
        print(f"  Loading TrackNet Ball Detector ({ball_path})...")
        t0 = time.time()
        ckpt = torch.load(ball_path, map_location="cpu")
        seq_len = ckpt['param_dict']['seq_len']
        bg_mode = ckpt['param_dict']['bg_mode']
        tracknet = get_model("TrackNet", seq_len, bg_mode)
        tracknet.load_state_dict(ckpt['model'])
        tracknet.eval()
        models["ball"] = tracknet
        models["ball_seq_len"] = seq_len
        print(f"    Loaded in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"  Error loading ball detector: {e}")
        
    print(f"Total model load time: {time.time() - t_start:.2f}s")
    
    # Read first 5 frames for testing
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        print("Error: Could not read frames from video.")
        return
        
    print(f"\nRead {len(frames)} test frames from video.")
    
    # Resolutions to test
    resolutions = {
        "720p (1280x720)": (1280, 720),
        "1080p (1920x1080)": (1920, 1080),
        "4K (3840x2160)": (3840, 2160)
    }
    
    results = {}
    
    print("\n[2/3] Running benchmarks per resolution (5 frames per run)...")
    for res_name, (w, h) in resolutions.items():
        print(f"\n--- Benchmark for {res_name} ---")
        
        # Resize test frames
        t0 = time.time()
        resized_frames = [cv2.resize(f, (w, h)) for f in frames]
        t_resize = ((time.time() - t0) / len(frames)) * 1000  # ms per frame
        print(f"  Frame Decode & Resize Latency: {t_resize:.2f} ms")
        
        # Player detection (YOLOv8 imgsz=1280)
        t_player = None
        if "player" in models:
            print("  Running Player Detection (imgsz=1280)...")
            t0 = time.time()
            for img in resized_frames[:2]: # 2 frames is enough to get average
                _ = models["player"].predict(img, imgsz=1280, conf=0.5, verbose=False, device="cpu", classes=[0])
            t_player = ((time.time() - t0) / 2.0) * 1000
            print(f"    Player Detection Latency: {t_player:.2f} ms")
            
        # Player Keypoint Pose (YOLOv8 imgsz=1280)
        t_pose = None
        if "pose" in models:
            print("  Running Player Pose Estimation (imgsz=1280)...")
            t0 = time.time()
            for img in resized_frames[:2]:
                _ = models["pose"].predict(img, imgsz=1280, conf=0.25, verbose=False, device="cpu", classes=[0])
            t_pose = ((time.time() - t0) / 2.0) * 1000
            print(f"    Player Pose Latency: {t_pose:.2f} ms")
            
        # Court Keypoints (YOLOv8 imgsz=640)
        t_court = None
        if "court" in models:
            print("  Running Court Keypoints Detection (imgsz=640)...")
            t0 = time.time()
            for img in resized_frames[:3]:
                _ = models["court"].predict(img, imgsz=640, conf=0.25, verbose=False, device="cpu")
            t_court = ((time.time() - t0) / 3.0) * 1000
            print(f"    Court Keypoints Latency: {t_court:.2f} ms")
            
        # Ball Tracker (TrackNet)
        t_ball = None
        if "ball" in models:
            print("  Running TrackNet Ball Detection (512x288)...")
            seq_len = models["ball_seq_len"]
            
            # Prepare dummy batch (seq_len frames stack, e.g. 8 frames)
            # TrackNet input channels: seq_len * 3 (RGB concatenated) + 3 background if concat bg
            # Let's create a tensor matching the shape: [1, seq_len * 3 + 3, 288, 512] or similar
            # TrackNet: concat mode means (seq_len + 1) * 3 = 27 channels for seq_len=8
            channels = (seq_len + 1) * 3
            dummy_input = torch.randn(1, channels, 288, 512, dtype=torch.float32)
            
            t0 = time.time()
            num_runs = 5
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = models["ball"](dummy_input)
            t_ball = ((time.time() - t0) / num_runs) * 1000
            print(f"    TrackNet Ball Detection Latency: {t_ball:.2f} ms")
            
        # Store results
        results[res_name] = {
            "res_w": w,
            "res_h": h,
            "t_resize": t_resize,
            "t_player": t_player,
            "t_pose": t_pose,
            "t_court": t_court,
            "t_ball": t_ball
        }
        
    print("\n[3/3] Saving results and printing summary...")
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nBenchmark results saved to benchmark_results.json")
    
    # Print markdown table
    print("\n" + "=" * 60)
    print("MARKDOWN PERFORMANCE TABLE")
    print("=" * 60)
    print("| Résolution | Resize/IO (ms) | Player Det (ms) | Pose Est (ms) | Court Det (ms) | TrackNet Ball (ms) | Total / Frame (ms) | Max FPS |")
    print("|---|---|---|---|---|---|---|---|")
    for res_name, data in results.items():
        tr = data["t_resize"] or 0
        tp = data["t_player"] or 0
        tk = data["t_pose"] or 0
        tc = data["t_court"] or 0
        tb = data["t_ball"] or 0
        
        t_total = tr + tp + tk + tc + tb
        fps = 1000.0 / t_total if t_total > 0 else 0
        
        tp_str = f"{tp:.1f}" if tp > 0 else "N/A"
        tk_str = f"{tk:.1f}" if tk > 0 else "N/A"
        tc_str = f"{tc:.1f}" if tc > 0 else "N/A"
        tb_str = f"{tb:.1f}" if tb > 0 else "N/A"
        
        print(f"| {res_name} | {tr:.1f} | {tp_str} | {tk_str} | {tc_str} | {tb_str} | {t_total:.1f} | {fps:.1f} |")
    print("=" * 60)

if __name__ == "__main__":
    benchmark()
