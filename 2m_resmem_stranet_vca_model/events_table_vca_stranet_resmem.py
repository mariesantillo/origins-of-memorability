import pandas as pd
import os
import glob
import numpy as np

event_files_folder = '/foundcog/forrestgump/foundcog-infants-2m/events/'
entropy_files_folder = '/foundcog/forrestgump/vca_results/'
rms_files_folder = '/foundcog/forrestgump/foundcog-infants-2m/infant2m_betas_vca_stranet_rsm_model/RMS_stranetvideos/values/'
resmem_files_folder = '/home/mariesantillo/resmem/predictions/'
output_folder = '/foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/expanded_events/'

os.makedirs(output_folder, exist_ok=True)

# Trial types
trial_types = [
    "minions_supermarket.mp4",
    "new_orleans.mp4",
    "bathsong.mp4",
    "dog.mp4",
    "moana.mp4",
    "forest.mp4"
]

fps = 25

# Get event files
event_files = glob.glob(os.path.join(event_files_folder, '*_events.tsv'))

if not event_files:
    print(f"No event files found in {event_files_folder}.")
# else:
#     print(f"Found {len(event_files)} event files.")

# Process event file
for events_file in event_files:
    base_name = os.path.basename(events_file)
    subject, run = base_name.split('_')[:2]


    # print(f"Processing {events_file} for subject: {subject}, run: {run}")

    try:
        events_df = pd.read_csv(events_file, sep='\t')
    except FileNotFoundError:
        print(f"Event file {events_file} not found. Skipping...")
        continue

    expanded_rows_list = []

    # Loop over trial types 
    for trial_type in trial_types:
        trial_type_stripped = trial_type.replace('.mp4', '')

        # get original rows from events.tsv
        original_rows = events_df[events_df['trial_type'] == trial_type]

        if original_rows.empty:
            print(f"No events found for trial type {trial_type} in {events_file}, keeping original rows.")
            continue  

        # Loop over each trial_type
        for idx, original_row in original_rows.iterrows():
            original_row = original_row.copy()
            original_row['original_trial_type'] = trial_type

            original_row['beta_video_part'] = np.nan

            # 1: vca_entropy - Modulator is the entropy scores
            entropy_file = os.path.join(entropy_files_folder, f'{trial_type_stripped}_y4m32_aggregated.csv')

            if os.path.exists(entropy_file):
                try:
                    entropy_df = pd.read_csv(entropy_file, sep=',')
                except FileNotFoundError:
                    print(f"Entropy file {entropy_file} not found, skipping.")
                    continue  

                if 'entropy' not in entropy_df.columns or 'POC' not in entropy_df.columns:
                    print(f"'entropy' or 'POC' column not found in {entropy_file}, skipping.")
                    continue  

                # Create Df for vca_entropy
                num_frames = len(entropy_df)
                repeated_video_rows = pd.DataFrame([original_row] * num_frames)

                # Set trial_type, modulation, and POC columns
                repeated_video_rows['trial_type'] = 'vca_entropy'
                repeated_video_rows['modulation'] = entropy_df['entropy'].values  # Set modulation to "vca"
                repeated_video_rows['POC'] = entropy_df['POC'].values

                # Calculate durations based on POC differences
                poc_values = entropy_df['POC'].values
                poc_differences = np.diff(poc_values, prepend=0)  # Prepend 0 for the first frame
                durations = poc_differences / fps  # Divide by fps to get duration in s

                # Calculate cumulative durations for onsets, starting from zero
                cumulative_durations = np.concatenate(([0], np.cumsum(durations[:-1])))

                # Set the onset for each frame
                repeated_video_rows['onset'] = original_row['onset'] + cumulative_durations

                # Assign durations
                repeated_video_rows['duration'] = durations

                # beta_video_part is NaN for vca
                repeated_video_rows['beta_video_part'] = np.nan

                expanded_rows_list.append(repeated_video_rows)
                # print(f"Processed vca_entropy for trial type {trial_type}, occurrence {idx}.")
            else:
                print(f"Entropy file {entropy_file} does not exist, skipping vca_entropy for {trial_type}.")
            
            # 1: resmem - Modulator is the entropy scores
            resmem_file = os.path.join(resmem_files_folder, f'predictions_{trial_type_stripped}.csv')

            if os.path.exists(resmem_file):
                try:
                    resmem_df = pd.read_csv(resmem_file, sep=',')
                except FileNotFoundError:
                    print(f"Resmem file {resmem_file} not found, skipping.")
                    continue

                if 'prediction' not in resmem_df.columns or 'filename' not in resmem_df.columns:
                    print(f"'prediction' or 'filename' column not found in {resmem_file}, skipping.")
                    continue

                # Create Df for vca_entropy
                num_frames = len(resmem_df)
                repeated_video_rows = pd.DataFrame([original_row] * num_frames)

                # Set trial_type, modulation, and POC columns
                repeated_video_rows['trial_type'] = 'resmem'
                repeated_video_rows['modulation'] = resmem_df['prediction'].values  # Set modulation to "resmem"
                repeated_video_rows['POC'] = resmem_df['filename'].values

                # Calculate durations based on POC differences
                poc_values = resmem_df['filename'].values
                poc_differences = np.diff(poc_values, prepend=0)  # Prepend 0 for the first frame
                durations = poc_differences / fps  # Divide by fps to get duration in s

                # Calculate cumulative durations for onsets, starting from zero
                cumulative_durations = np.concatenate(([0], np.cumsum(durations[:-1])))

                # Set the onset for each frame
                repeated_video_rows['onset'] = original_row['onset'] + cumulative_durations

                # Assign durations
                repeated_video_rows['duration'] = durations

                # beta_video_part is NaN for vca
                repeated_video_rows['beta_video_part'] = np.nan

                expanded_rows_list.append(repeated_video_rows)
                # print(f"Processed vca_entropy for trial type {trial_type}, occurrence {idx}.")
            else:
                print(f"resmem file {resmem_file} does not exist, skipping vca_entropy for {trial_type}.")

            # 2: rsm_difference - Modulator is the RMS regressor averaged over every 15 frames
            frames_per_group = 15
            frame_duration = 1 / fps  # duration of one frame
            group_duration = frames_per_group * frame_duration  # duration of a group

            # Load RMS CSV
            rms_file = os.path.join(rms_files_folder, f'{trial_type_stripped}_rms_values.csv')
            if not os.path.exists(rms_file):
                print(f"RMS file {rms_file} does not exist, skipping rsm_difference for {trial_type}.")
                continue

            try:
                rms_df = pd.read_csv(rms_file)
            except Exception as e:
                print(f"Error reading {rms_file}: {e}")
                continue

            rms_values = rms_df['RMS'].values
            num_groups = int(np.ceil(len(rms_values) / frames_per_group))

            rsm_rows = []

            for group_idx in range(num_groups):
                start_frame = group_idx * frames_per_group
                end_frame = min((group_idx + 1) * frames_per_group, len(rms_values))
                
                avg_rms = np.mean(rms_values[start_frame:end_frame])
                group_len = end_frame - start_frame
                duration = group_len * frame_duration
                onset = original_row['onset'] + (start_frame * frame_duration)

                row = original_row.copy()
                row['trial_type'] = 'rsm_difference'
                row['modulation'] = avg_rms
                row['POC'] = start_frame
                row['onset'] = onset
                row['duration'] = duration
                row['beta_video_part'] = np.nan
                rsm_rows.append(row)

            # Convert to DataFrame and append
            if rsm_rows:
                rsm_regressor_df = pd.DataFrame(rsm_rows)
                expanded_rows_list.append(rsm_regressor_df)

    # Combine the expanded rows and save file
    if expanded_rows_list:
        expanded_df = pd.concat(expanded_rows_list, ignore_index=True)

        # attention_getter rows
        attention_getter_rows = events_df[events_df['trial_type'] == "attention_getter"]
        attention_getter_rows['beta_video_part'] = np.nan  
        if 'modulation' not in attention_getter_rows.columns:
            attention_getter_rows['modulation'] = np.nan
        else:
            attention_getter_rows['modulation'] = attention_getter_rows['modulation'].fillna(np.nan)

        final_df = pd.concat([expanded_df, attention_getter_rows], ignore_index=True)

        final_df = final_df.sort_values(by='onset').reset_index(drop=True)

        # output file path and name
        new_file_name = os.path.join(output_folder, f'{os.path.splitext(base_name)[0]}_expanded.tsv')

        try:
            final_df.to_csv(new_file_name, sep='\t', index=False)
            print(f"Saved expanded file as {new_file_name}")
        except Exception as e:
            print(f"Error saving the expanded file {new_file_name}: {e}")
    else:
        print(f"No trial types were successfully processed for {events_file}.")
