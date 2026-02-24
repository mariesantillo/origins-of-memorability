#run via demo_test.py, download from (https://github.com/ashleylqx/STRA-Net)

import numpy as np
import pandas as pd
import os
from PIL import Image
from natsort import natsorted

def load_attention_frames(folder_path):
    frame_files = natsorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    frames = []

    for file in frame_files:
        img = Image.open(os.path.join(folder_path, file)).convert('L')  # grayscale
        frames.append(np.array(img, dtype=np.float32))

    return frames

def compute_rms_difference(attention_blobs):
    framewise_rms = []
    for i in range(1, len(attention_blobs)):
        diff = attention_blobs[i] - attention_blobs[i - 1]
        rms = np.sqrt(np.mean(diff**2))
        framewise_rms.append(rms)
    return np.array(framewise_rms)

def process_all_videos(main_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        print(f"Processing {subfolder}...")
        frames = load_attention_frames(subfolder_path)
        rms_values = compute_rms_difference(frames)

        output_path = os.path.join(output_folder, f"{subfolder}_rms_values.csv")
        df = pd.DataFrame({'RMS': rms_values})
        df.to_csv(output_path, index=False)

main_video_folder = r"C:\Users\MSANTILL\Desktop\vap_predictions\FOUNDCOG\UHD_dcross_res_matt_res" 
output_regressor_folder = r"C:\Users\MSANTILL\Desktop\RMS_stranetvideos"      

process_all_videos(main_video_folder, output_regressor_folder)
