import warnings
import os
import pickle
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import defaultdict
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def compute_vif(design_matrix):
    """Compute Variance Inflation Factor (VIF) for each column in the design matrix."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = design_matrix.columns
    vif_data["VIF"] = [
        variance_inflation_factor(design_matrix.values, i) 
        for i in range(design_matrix.shape[1])
    ]
    return vif_data

def _get_betas(model):
    labels = model.labels_[0]
    regression_result = model.results_[0]

    label =next(iter(regression_result.keys()))
    n_columns = len(regression_result[label].theta)

    # Initialize effect and vcov as zeros of the same length as the labels.
    effect_matrix = np.zeros((labels.size, n_columns))
    vcov_matrix = np.zeros((labels.size, n_columns))
    cov= None 

    for label_ in regression_result.keys():
        if cov is None:
            cov = regression_result[label_].cov
        label_mask = (labels == label_)
        if label_mask.any():
            for colind in range(n_columns):
                resl = regression_result[label_].theta[colind]
                vcov_col = regression_result[label_].vcov(column=colind)
                effect_matrix[label_mask, colind] = resl
                vcov_matrix[label_mask, colind] = vcov_col

    return effect_matrix, vcov_matrix, cov

def model_run(sub, task, recorded_tr=0.656, brain_mask='/foundcog/forrestgump/mask/adults/mni152_resampled_brain_mask_adult.nii.gz', fwd_cutoff=0.5, derivs='normalized_to_common_space'):
   
    import glob
    from os.path import join, exists

    def _get_fnames(sub, task, derivs):
        expt_dir = '/foundcog/forrestgump/foundcog-adult-2/data'
        output_dir = '/foundcog/forrestgump/foundcog-adult-2/resmem_stranet_vca_unscoredvideotest/'
        funcpaths = []
        eventpaths = []
        motionpaths = []

        for run in range(1, 3):
            run_str = f'run-{run}'

            func_file = os.path.join(expt_dir, f'{sub}', 'ses-001', 'func', f'{sub}_ses-001_task-{task}_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
            event_file = os.path.join(output_dir, 'expanded_events', f'{sub}_ses-001_task-{task}_run-00{run}_events_expanded.tsv')
            motion_file = os.path.join(expt_dir, f'{sub}', 'ses-001', 'func', f'{sub}_ses-001_task-{task}_run-{run}_desc-confounds_timeseries.tsv')

            if os.path.exists(func_file):
                funcpaths.append(func_file)
            if os.path.exists(event_file):
                eventpaths.append(event_file)
            if os.path.exists(motion_file):
                motionpaths.append(motion_file)

            print(f'Checking files for subject {sub}:')
            print(f'  Functional file: {func_file}, exists: {os.path.exists(func_file)}')
            print(f'  Event file: {event_file}, exists: {os.path.exists(event_file)}')
            print(f'  Motion file: {motion_file}, exists: {os.path.exists(motion_file)}')

        return {'func': funcpaths, 'events': eventpaths, 'motion': motionpaths}

    brain_mask = os.path.abspath(brain_mask)
    if not os.path.exists(brain_mask):
        raise FileNotFoundError(f"Brain mask file {brain_mask} does not exist.")
    else:
        print(f"Brain mask file found: {brain_mask}")

    paths = _get_fnames(sub, task, derivs)
    events_list = [pd.read_csv(ev, sep='\t') for ev in paths['events']]
    outpaths = []

    for runidx in range(len(paths['func'])):
        run_events = events_list[runidx]
        run_img = paths['func'][runidx]
        motion_file = paths['motion'][runidx]

        model_path = os.path.abspath(f'/foundcog/forrestgump/foundcog-adult-2/resmem_stranet_vca_unscoredvideotest/glm/model/{sub}_task-{task}_run-{runidx + 1}_model.pkl')

        if os.path.exists(model_path):
            print(f"Model already exists for sub-{sub}, run {runidx + 1}, skipping.")
            with open(model_path, 'rb') as f:
                model_save = pickle.load(f)
                model = model_save['model']
                design = model_save['design_matrix']
        else: 
            print(f"Model does not exist for sub-{sub}, run {runidx + 1}, initializing.")

            # --- Load motion regressors ---
            motion_df = pd.read_csv(motion_file, sep='\t', header=0)
            motion_df.columns = motion_df.columns.str.strip()

            # Keep only the 6 rigid body parameters
            motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            motion_df = motion_df[motion_df.columns.intersection(motion_cols)].copy()

            # Fill NaNs in motion parameters
            motion_df.fillna(0, inplace=True)

            # --- Load framewise displacement ---
            fwd_df = pd.read_csv(motion_file, sep='\t', header=0)
            fwd_df.columns = fwd_df.columns.str.strip()

            if 'framewise_displacement' not in fwd_df.columns:
                raise ValueError(f"Missing 'framewise_displacement' in {motion_file}")

            fwd = pd.to_numeric(fwd_df['framewise_displacement'], errors='coerce')

            # Print and fill missing
            if fwd.isnull().any():
                print(f"[WARNING] Found {fwd.isnull().sum()} missing or non-numeric values in 'framewise_displacement'.")
                print("Missing indices:", fwd[fwd.isnull()].index.tolist())
                for idx in fwd[fwd.isnull()].index:
                    print(fwd.iloc[max(idx - 2, 0):idx + 3])
                fwd.fillna(0, inplace=True)

            # --- Motion spikes ---
            above_idxs = fwd.index[fwd > fwd_cutoff].values
            spike_arr = np.zeros((len(fwd), len(above_idxs)))
            spike_arr[above_idxs, np.arange(len(above_idxs))] = 1
            spike_names = [f'spike_{i}' for i in range(len(above_idxs))]

            # --- Check alignment ---
            if spike_arr.shape[0] != motion_df.shape[0]:
                raise ValueError(f"Row mismatch: motion_df {motion_df.shape[0]} vs spikes {spike_arr.shape[0]}")

            # --- Concatenate ---
            confounds = np.hstack([motion_df, spike_arr])
            confound_names = motion_cols + spike_names
            confounds = np.nan_to_num(confounds)

            # --- Check for excessive motion ---
            if len(above_idxs) > len(fwd) * 0.5:
                print(f"[SKIP] sub-{sub}, run-{runidx + 1}: excessive motion ({len(above_idxs)} of {len(fwd)} frames)")
                continue
            else:
                print(f"No significant motion detected in sub-{sub}, run-{runidx + 1}")

            # --- Optional: Debug motion collinearity ---
            print("\n[INFO] Motion regressor correlation matrix:")
            print(pd.DataFrame(motion_df).corr().round(2))

            print(f'Initializing model with brain mask: {brain_mask}')

            model = FirstLevelModel(t_r=recorded_tr, mask_img=brain_mask, smoothing_fwhm=8)

            # Load the functional image to determine the number of scans
            n_scans = nib.load(run_img).shape[3]

            frame_times = np.arange(n_scans) * recorded_tr

            run_events['trial_type'] = run_events['trial_type'].str.replace('.mp4', '', regex=False)

            run_events = run_events[~run_events['trial_type'].isin(['attention_getter'])]

            # 'resmem_other', 'vca_entropy_other', 'rms_other'
            minimal_events = run_events[run_events['trial_type'].isin(['resmem', 'vca_entropy', 'rms_difference', 'resmem_other', 'vca_entropy_other', 'rms_other'])].copy()

            # Create design matrix
            try:
                design = make_first_level_design_matrix(
                    frame_times,
                    events=minimal_events,
                    hrf_model=model.hrf_model,
                    drift_model=model.drift_model,
                    high_pass=model.high_pass,
                    drift_order=model.drift_order,
                    fir_delays=model.fir_delays,
                    add_regs=confounds,
                    add_reg_names=confound_names,
                    min_onset=model.min_onset
                )

                print("Design matrix created successfully.")

                 # Compute VIF
                vif = compute_vif(design)
                print("Variance Inflation Factor (VIF) for design matrix:")
                print(vif)

                # Optionally save VIF results to a file
                vif_output_path = f'/foundcog/forrestgump/foundcog-adult-2/resmem_stranet_vca_unscoredvideotest/glm/vif_{sub}_run_{runidx + 1}.csv'
                vif.to_csv(vif_output_path, index=False)
                print(f"VIF saved to {vif_output_path}")

                # Option B: If design_matrix is a NumPy array:
                # full_design_df = pd.DataFrame(np.hstack((design_matrix, confounds)), 
                #                               columns=design_matrix_columns + confound_names)

                # Step 2: Calculate correlation matrix
                corr_matrix = design.corr()

                # Step 3: Plot heatmap
                plt.figure(figsize=(18, 15))
                sns.heatmap(corr_matrix, cmap="coolwarm", center=0, 
                            square=True, linewidths=0.5, xticklabels=True, yticklabels=True)
                plt.title("Correlation Matrix of Full Design Matrix", fontsize=18)
                plt.tight_layout()
                plt.savefig(f'/foundcog/forrestgump/foundcog-adult-2/resmem_stranet_vca_unscoredvideotest/glm/confound_corr_{sub}_run_{runidx + 1}.png')
                plt.show()

            except Exception as e:
                print(f"Error creating design matrix: {e}")
                continue
            print(design.columns)

            # Plot design matrix
            try:
                fig, ax = plt.subplots(figsize=(12, 10))
                plot_design_matrix(design, ax=ax)
                plt.title(f'Design Matrix for Subject {sub} Run {runidx + 1}')
                plot_path = f'/foundcog/forrestgump/foundcog-adult-2/resmem_stranet_vca_unscoredvideotest/glm/design_matrices/design_matrix_{sub}_run_{runidx + 1}.png'
                plt.savefig(plot_path)
                plt.close(fig)
                print(f"Design matrix saved to {plot_path}.")
            except Exception as e:
                print(f"Error plotting or saving design matrix: {e}")
                continue

            # Fit the model
            try:
                model.fit(run_img, design_matrices=design)
                print("Model fitting completed successfully.")
            except Exception as e:
                print(f"Error during model fitting: {e}")
                continue
            
            all_effects, all_vcovs, full_cov = _get_betas(model)

            # Save the model
            model_save = {
                'model': model,
                'input_events': run_events,
                'funcfile': run_img,
                'fwdcutoff': fwd_cutoff,
                'design_matrix': design,
                'all_effects': all_effects,
                'all_vcovs': all_vcovs,
                'full_cov': full_cov
            }

            model_path = os.path.abspath(f'/foundcog/forrestgump/foundcog-adult-2/resmem_stranet_vca_unscoredvideotest/glm/model/{sub}_task-{task}_run-{runidx + 1}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_save, f)
            print(f"Model saved to {model_path}")
            outpaths.append(model_path)

        # Compute contrasts
            contrast_definitions = {
                'resmem': np.eye(design.shape[1])[0],           # 1st column
                'rms_difference': np.eye(design.shape[1])[1],   # 2nd column
                'vca_entropy': np.eye(design.shape[1])[2]       # 3rd column
            }

            for contrast_id, contrast_vector in contrast_definitions.items():
                print(f"Computing contrast '{contrast_id}' with vector: {contrast_vector}")

                try:
                    beta_map = model.compute_contrast(contrast_vector, output_type='stat')
                    beta_map_path = os.path.abspath(
                        f'/foundcog/forrestgump/foundcog-adult-2/resmem_stranet_vca_unscoredvideotest/glm/beta_maps/{sub}_task-{task}_run-{runidx + 1}_{contrast_id}_beta_map.nii.gz'
                    )
                    nib.save(beta_map, beta_map_path)
                    print(f"Contrast {contrast_id} computed and saved to {beta_map_path}.")

                    infant_template_path ='/foundcog/forrestgump/mask/adults/mni152_resampled_brain_mask_adult.nii.gz'
                    plotting.plot_stat_map(
                        beta_map,
                        bg_img=infant_template_path,
                        title=f'Contrast {contrast_id} for Subject {sub} Run {runidx + 1}',
                        display_mode='ortho',
                        cut_coords=(0, 0, 0),
                        output_file=f'/foundcog/forrestgump/foundcog-adult-2/resmem_stranet_vca_unscoredvideotest/glm/contrast/{sub}_task-{task}_run-{runidx + 1}_{contrast_id}_effect_size_map.png'
                    )

                except Exception as e:
                    print(f"Error computing contrast '{contrast_id}': {e}")
                
    return outpaths

if __name__ == '__main__':
    with open('/foundcog/forrestgump/foundcog-adult-2/subjects.txt', 'r') as file:
        subject_list = [line.strip() for line in file.readlines()]

    for sub in subject_list: 
        for task in ['video']:
            paths = model_run(sub, task, fwd_cutoff=1.5)
            print(f'Subject {sub} paths:', paths)
