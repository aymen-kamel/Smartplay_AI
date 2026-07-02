import cv2
import supervision as sv
import numpy as np
import json
import os
from trackers import PlayerTracker, SAM2PlayerTracker
from config import *

# Global variables for interactive selection
SELECTED_KEYPOINTS = []
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

def click_event(event, x, y, flags, params): 
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        w, h, img_display = params
        # Scale back to original resolution
        original_x = int(x * (w / DISPLAY_WIDTH))
        original_y = int(y * (h / DISPLAY_HEIGHT))
  
        SELECTED_KEYPOINTS.append((original_x, original_y))
  
        # displaying the coordinates 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img_display, f"{original_x},{original_y}", (x, y), font, 
                    0.5, (255, 0, 0), 2) 
        cv2.imshow('Court Selection', img_display) 

def select_keypoints(img, w, h):
    global DISPLAY_WIDTH, DISPLAY_HEIGHT
    # Ensure we don't upscale small images
    if w < DISPLAY_WIDTH and h < DISPLAY_HEIGHT:
        DISPLAY_WIDTH = w
        DISPLAY_HEIGHT = h
    
    img_display = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    
    print("\n" + "="*50)
    print("COURT KEYPOINT SELECTION")
    print("Please click on the 12 court keypoints in the following order:")
    print("k11, k12, k8, k9, k10, k6, k7, k3, k4, k5, k1, k2")
    print("Press any key when finished.")
    print("="*50 + "\n")

    cv2.imshow('Court Selection', img_display)
    cv2.setMouseCallback('Court Selection', click_event, param=(w, h, img_display)) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    return SELECTED_KEYPOINTS

def track_players(input_path, output_path):
    """
    Standalone script to track players in a padel video.
    """
    print(f"Starting player tracking for: {input_path}")
    
    # 1. Load video info
    video_info = sv.VideoInfo.from_video_path(video_path=input_path)
    
    # Padel court keypoints are usually saved in this path
    keypoints_path = "./cache/fixed_keypoints_detection.json"
    keypoints = []
    
    if os.path.exists(keypoints_path):
        try:
            with open(keypoints_path, "r") as f:
                keypoints = json.load(f)
                print("Using court boundaries from cache.")
        except Exception as e:
            print(f"Warning: Could not load keypoints from {keypoints_path}: {e}")

    if not keypoints:
        # Get first frame for selection
        first_frame_generator = sv.get_video_frames_generator(input_path, start=0, end=1)
        img = next(first_frame_generator)
        keypoints = select_keypoints(img, video_info.width, video_info.height)
        
        # Save selected keypoints to cache for next time
        if keypoints:
            os.makedirs(os.path.dirname(keypoints_path), exist_ok=True)
            with open(keypoints_path, "w") as f:
                json.dump(keypoints, f)
            print(f"Keypoints saved to {keypoints_path}")

    if len(keypoints) >= 4:
        # Using corners of the court (usually k1, k2, k12, k11 in the 12-point system)
        # However, the order depends on how they were clicked. 
        # In main.py's doc: 
        # k11 is index 0, k12 is index 1... no, wait.
        # Let's check main.py's click order or indices.
        # k1: 0, k2: 1 ... k12: 11
        # Polygon in main.py uses [0, 1, -1, -2] which are k1, k2, k12, k11.
        polygon = np.array([
            keypoints[0],  # k1
            keypoints[1],  # k2
            keypoints[-1], # k12
            keypoints[-2]  # k11
        ])
    else:
        print("Not enough keypoints selected. Using full frame.")

    polygon_zone = sv.PolygonZone(polygon=polygon)

    # 3. Initialize the PlayerTracker
    if USE_SAM2:
        print("Using SAM2 for advanced tracking...")
        tracker = SAM2PlayerTracker(
            model_path=PLAYERS_TRACKER_MODEL,
            sam2_config=SAM2_MODEL_CONFIG,
            sam2_weights=SAM2_MODEL_WEIGHTS,
            polygon_zone=polygon_zone,
            batch_size=PLAYERS_TRACKER_BATCH_SIZE,
            annotator=PLAYERS_TRACKER_ANNOTATOR,
            show_confidence=True
        )
        tracker.video_info_post_init(video_info)
        tracker.init_video_state(input_path)
    else:
        tracker = PlayerTracker(
            model_path=PLAYERS_TRACKER_MODEL,
            polygon_zone=polygon_zone,
            batch_size=PLAYERS_TRACKER_BATCH_SIZE,
            annotator=PLAYERS_TRACKER_ANNOTATOR,
            show_confidence=True
        )
        tracker.video_info_post_init(video_info)

    tracker.to(tracker.DEVICE)

    # 4. Process Video
    frame_generator = sv.get_video_frames_generator(source_path=input_path)
    
    print(f"Tracking and saving results to {output_path}...")
    
    with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
        # We process in batches as defined in configuration for efficiency
        # However, for simplicity and real-time visualization, we can also do frame by frame
        # ByteTrack keeps state internally in the tracker instance
        
        for frame in frame_generator:
            # Predict for the current frame
            # predict_sample expects a list of frames and it converts to RGB internally
            predictions = tracker.predict_sample([frame])[0]
            
            # Convert frame to RGB for drawing (per the project's internal logic)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Draw tracking results on the frame
            # predictions is an instance of Players class
            annotated_frame_rgb = predictions.draw(
                frame=frame_rgb, 
                video_info=video_info,
                annotator=tracker.annotator,
                show_confidence=tracker.show_confidence
            )
            
            # Convert back to BGR for saving to video
            annotated_frame_bgr = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Save the frame
            sink.write_frame(annotated_frame_bgr)

    print("Success! Player tracking video generated.")

if __name__ == "__main__":
    # Use default input from config if available
    input_video = "./input/rally.mp4"
    output_video = "player_tracking_output.mp4"
    
    if not os.path.exists(input_video):
        print(f"Error: Input video not found at {input_video}")
        # Try a fallback if standard workspace path is different
        if os.path.exists("./input/Adobe Express - padel_match1.mp4"):
            input_video = "./input/Adobe Express - padel_match1.mp4"
        else:
            print("Please check INPUT_VIDEO_PATH in config.py or the input folder.")
            exit(1)

    track_players(input_video, output_video)
