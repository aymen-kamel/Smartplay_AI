import cv2
import pandas as pd
import os
from config import INPUT_VIDEO_PATH

def main():
    if not os.path.exists("shot_features.csv"):
        print("Run collect_shot_data.py first!")
        return

    df = pd.read_csv("shot_features.csv")
    if 'label' not in df.columns:
        df['label'] = None

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    
    print("\n--- Padel Shot Annotation Tool ---")
    print("Keys: [f] Forehand, [b] Backhand, [s] Smash, [v] Volley, [x] Skip/Other, [q] Save and Quit")
    
    label_map = {
        ord('f'): 'forehand',
        ord('b'): 'backhand',
        ord('s'): 'smash',
        ord('v'): 'volley',
        ord('x'): 'other'
    }

    try:
        for idx, row in df.iterrows():
            if pd.notnull(df.at[idx, 'label']):
                continue
                
            frame_idx = int(row['frame'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Draw player circle to help identification
            # p_x = int(row['player_x_m']) # We don't have pixel coords here easily without more logic
            # Just show the frame
            
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.putText(display_frame, f"Shot {idx+1}/{len(df)} | Frame {frame_idx}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "F: Forehand | B: Backhand | S: Smash | V: Volley | X: Other", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Annotator", display_frame)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key in label_map:
                df.at[idx, 'label'] = label_map[key]
                print(f"Labeled shot {idx} as {label_map[key]}")
            else:
                print("Skipped.")

    finally:
        df.to_csv("shot_features_labeled.csv", index=False)
        print("\nProgress saved to shot_features_labeled.csv")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
