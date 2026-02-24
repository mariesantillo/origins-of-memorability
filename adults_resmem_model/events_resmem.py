import pandas as pd
import os
import glob
import numpy as np

# Input paths

resmem_files_folder = '/home/mariesantillo/resmem/videos/foundcog_videos/'
resmem_files_folder_othervideos = '/home/mariesantillo/resmem/videos/adult_othervideos/'

output_folder = '/foundcog/forrestgump/foundcog-adult-2/resmem_model/expanded_events/'

os.makedirs(output_folder, exist_ok=True)

trial_types = [
    "minions_supermarket.mp4", "new_orleans.mp4", "bathsong.mp4",
    "dog.mp4", "moana.mp4", "forest.mp4"
]

other_videos_trial_types = [
    "bedtime.mp4", "brother_bear.mp4", "kids_kitchen.mp4", "our_planet.mp4",
    "pingu.mp4", "piper.mp4", "playground.mp4", "ratatouille.mp4",
    "real_cars.mp4", "rio_jungle_jig.mp4"
]

fps = 25

# Load all event files
event_files = glob.glob(os.path.join(event_files_folder, '*_events.tsv'))

for events_file in event_files:
    base_name = os.path.basename(events_file)
    subject, run = base_name.split('_')[:2]

    try:
        events_df = pd.read_csv(events_file, sep='\t')
    except FileNotFoundError:
        continue

    expanded_rows = []

    for trial_type in trial_types:
        trial_type_stripped = trial_type.replace('.mp4', '')
        original_rows = events_df[events_df['trial_type'] == trial_type]
        if original_rows.empty:
            continue

        for _, original_row in original_rows.iterrows():

            # ResMem (grouped)
            resmem_file = os.path.join(resmem_files_folder, f'{trial_type_stripped}_predictions.csv')
            if os.path.exists(resmem_file):
                resmem_df = pd.read_csv(resmem_file)
                if {'prediction', 'filename'}.issubset(resmem_df.columns):
                    pred_values = resmem_df['prediction'].values
                    group_len = 15
                    for i in range(0, len(pred_values), group_len):
                        segment = pred_values[i:i+group_len]
                        avg_pred = np.mean(segment)
                        duration = len(segment) / fps
                        onset = original_row['onset'] + (i / fps)
                        expanded_rows.append({"onset": onset, "duration": duration,
                                                "trial_type": 'resmem', "stim_file": trial_type,
                                                "original_trial_type": trial_type, "beta_video_part": np.nan,
                                                "modulation": avg_pred, "POC": i})


    # Process other videos
    for trial_type in other_videos_trial_types:
        trial_type_stripped = trial_type.replace('.mp4', '')
        original_rows = events_df[events_df['trial_type'] == trial_type]
        if original_rows.empty:
            continue

        for _, original_row in original_rows.iterrows():
            # VCA Other
            entropy_file = os.path.join(entropy_files_folder_othervideos, f'{trial_type_stripped}_32.csv')
            if os.path.exists(entropy_file):
                entropy_df = pd.read_csv(entropy_file)
                if 'entropy' in entropy_df.columns:
                    entropy_values = entropy_df['entropy'].values
                    group_len = 15
                    for i in range(0, len(entropy_values), group_len):
                        segment = entropy_values[i:i+group_len]
                        avg_entropy = np.mean(segment)
                        duration = len(segment) / fps
                        onset = original_row['onset'] + (i / fps)
                        expanded_rows.append({
                            "onset": onset,
                            "duration": duration,
                            "trial_type": "vca_entropy_other",
                            "stim_file": trial_type,
                            "original_trial_type": trial_type,
                            "beta_video_part": np.nan,
                            "modulation": avg_entropy,
                            "POC": i
                        })

            # ResMem (grouped)
            resmem_file = os.path.join(resmem_files_folder_othervideos, f'{trial_type_stripped}_predictions.csv')
            if os.path.exists(resmem_file):
                resmem_df = pd.read_csv(resmem_file)
                if {'prediction', 'filename'}.issubset(resmem_df.columns):
                    pred_values = resmem_df['prediction'].values
                    group_len = 15
                    for i in range(0, len(pred_values), group_len):
                        segment = pred_values[i:i+group_len]
                        avg_pred = np.mean(segment)
                        duration = len(segment) / fps
                        onset = original_row['onset'] + (i / fps)
                        expanded_rows.append({"onset": onset, "duration": duration,
                                                "trial_type": 'resmem_other', "stim_file": trial_type,
                                                "original_trial_type": trial_type, "beta_video_part": np.nan,
                                                "modulation": avg_pred, "POC": i})


    # Sort and save
    sorted_rows = sorted(expanded_rows, key=lambda x: x['onset'])
    output_file = os.path.join(output_folder, f'{os.path.splitext(base_name)[0]}_expanded.tsv')
    with open(output_file, 'w') as f:
        f.write("\t".join(["onset", "duration", "trial_type", "stim_file",
                           "original_trial_type", "beta_video_part", "modulation", "POC"]) + "\n")
        for row in sorted_rows:
            f.write("\t".join(str(row.get(col, "")) for col in ["onset", "duration", "trial_type",
                                                                 "stim_file", "original_trial_type",
                                                                 "beta_video_part", "modulation", "POC"]) + "\n")
    print(f"Saved sorted expanded file: {output_file}")
