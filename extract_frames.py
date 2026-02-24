import cv2
import os

def extract_all_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    frame_index = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_index:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1
        frame_index += 1

    cap.release()
    print(f"Done! Saved {saved_count} frames to {output_folder}")

output_folder = "/home/mariesantillo/resmem/code/stranet_video_frames"  
video_path = "/home/mariesantillo/resmem/code/stranet_video"  

for video_file in os.listdir(video_path):
    if video_file.endswith(".mp4"):
        full_video_path = os.path.join(video_path, video_file)
        video_output_folder = os.path.join(output_folder, os.path.splitext(video_file)[0])
        print(f"Processing video: {video_file}")
        extract_all_frames(full_video_path, video_output_folder)
