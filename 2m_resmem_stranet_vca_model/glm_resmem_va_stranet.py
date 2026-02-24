import warnings
import os
import pickle
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from nilearn.plotting import plot_design_matrix
from collections import defaultdict

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

def model_run(sub, task, recorded_tr=0.610, brain_mask='/foundcog/forrestgump/mask/infants/nihpd_asym_02-05_fcgmask_2mm.nii.gz', fwd_cutoff=0.5, derivs='smoothing'):
   
    import glob
    import seaborn as sns
    from os.path import join, exists

    def _get_fnames(sub, task, derivs):
        base_dir = '/foundcog/forrestgump/'
        expt_dir = '/foundcog/bids/'
        funcpaths = []
        eventpaths = []
        motionpaths = []
        fwdpaths = []
        
        for sesnum in [1, 2]: 
            for runnum in [1, 2]:
                
                func_file = join(
                    expt_dir, 'derivatives', derivs, f'_subject_id_{sub}', '_referencetype_standard', f'_run_00{runnum}_session_{sesnum}_task_name_videos',
                    f'sub-{sub}_ses-{sesnum}_task-{task}_dir-AP_run-00{runnum}_bold_mcf_corrected_flirt_smooth.nii.gz'
                )
                event_file = join(
                    base_dir, 'foundcog-infants-2m', 'resmem_vca_stranet_rsm_model', 'expanded_events',
                    f'sub-{sub}_ses-{sesnum}_task-{task}_dir-AP_run-00{runnum}_events_expanded.tsv'
                )
                motion_file = join(
                    expt_dir, 'derivatives', 'motion_parameters', f'_subject_id_{sub}', f'_run_00{runnum}_session_{sesnum}_task_name_{task}', '_referencetype_standard',
                    f'sub-{sub}_ses-{sesnum}_task-{task}_dir-AP_run-00{runnum}_bold_mcf.nii.par'
                )

                fwd_file = join(
                    expt_dir, 'derivatives', 'motion_fwd', f'_subject_id_{sub}', f'_run_00{runnum}_session_{sesnum}_task_name_{task}', '_referencetype_standard',
                    f'fd_power_2012.txt'
                )

                # Check if files exist and add to lists
                if exists(func_file):
                    funcpaths.append(func_file)
                else:
                    print(f'Functional file not found: {func_file}')
                if exists(event_file):
                    eventpaths.append(event_file)
                else:
                    print(f'Event file not found: {event_file}')
                if exists(motion_file):
                    motionpaths.append(motion_file)
                else:
                    print(f'Motion file not found: {motion_file}')
                if exists(fwd_file):
                    fwdpaths.append(fwd_file)
                else:
                    print(f'FWD file not found: {fwd_file}')

                if sub=='ICC47':
                    for idx,funcrun in enumerate(funcpaths):
                        if '_run_1_session_1_task_name_videos' in funcrun:
                            funcpaths.pop(idx)

        return {'func': funcpaths, 'events': eventpaths, 'motion': motionpaths, 'fwd': fwdpaths}

    brain_mask = os.path.abspath(brain_mask)
    if not os.path.exists(brain_mask):
        raise FileNotFoundError(f"Brain mask file {brain_mask} does not exist.")
    else:
        print(f"Brain mask file found: {brain_mask}")

    paths = _get_fnames(sub, task, derivs)

    events_list = [pd.read_csv(ev, sep='\t') for ev in paths['events']]
    outpaths = []

    for runidx in range(len(paths['func'])):
        run_img = paths['func'][runidx]
        run_events = events_list[runidx]
        if runidx >= len(events_list):
            print(f"No matching event file for run index {runidx}, skipping.")
            continue
        motion_file = paths['motion'][runidx]
        fwd_file = paths['fwd'][runidx]

        model_path = os.path.abspath(f'/foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/glm/model/{sub}_task-{task}_run-{runidx + 1}_model.pkl')

        if os.path.exists(model_path):
            print(f"Model already exists for sub-{sub}, run {runidx + 1}, skipping.")
            with open(model_path, 'rb') as f:
                model_save = pickle.load(f)
                model = model_save['model']
                design = model_save['design_matrix']
                # Step 2: Calculate correlation matrix
                corr_matrix = design.corr()

                # Step 3: Plot heatmap
                plt.figure(figsize=(18, 15))
                sns.heatmap(corr_matrix, cmap="coolwarm", center=0, 
                            square=True, linewidths=0.5, xticklabels=True, yticklabels=True)
                plt.title("Correlation Matrix of Full Design Matrix", fontsize=18)
                plt.tight_layout()
                plt.savefig(f'/foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/glm/confound_corr_{sub}_run_{runidx + 1}.png')
                plt.show()
            
        else: 
            print(f"Model does not exist for sub-{sub}, run {runidx + 1}, initializing.")

            # Read motion parameters
            motion_df = pd.read_csv(motion_file, sep='\s+', header=None)
            motion_df.columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

            # Read framewise displacement
            try:
                fwd_df = pd.read_csv(fwd_file, header=0, delim_whitespace=True)
                if 'FramewiseDisplacement' in fwd_df.columns:
                    # Extract the 'FramewiseDisplacement' column
                    fwd = fwd_df['FramewiseDisplacement']
                else:
                    # If the header is not correctly read, skip the first row
                    fwd_df = pd.read_csv(fwd_file, header=None, skiprows=1, delim_whitespace=True, names=['FramewiseDisplacement'])
                    fwd = fwd_df['FramewiseDisplacement']
            except Exception as e:
                print(f"Error reading FWD file {fwd_file}: {e}")
                continue  

            # Ensure 'fwd' is numeric
            fwd = pd.to_numeric(fwd, errors='coerce')

            # Check for non-numeric values
            if fwd.isnull().any():
                print(f"Non-numeric values found in FWD data for sub-{sub}, session {sesnum}, run {runnum}")
                fwd = fwd.fillna(0)  # Handle NaNs 

            # Proceed with identifying high-motion 
            above_idxs = fwd.index[fwd > fwd_cutoff].values

            # Create spike regressors for high motion timepoints
            spike_arr = np.zeros((len(fwd)+1, above_idxs.size))
            spike_arr[above_idxs, np.arange(above_idxs.size)] = 1

            confounds = np.hstack((motion_df.values, spike_arr))
            confound_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'] + [f'spike_{i}' for i in range(len(above_idxs))]
            confounds = np.nan_to_num(confounds)

            if len(above_idxs) > len(fwd) * 0.5:
                print(f'Too much motion in sub-{sub} run {runidx + 1}, skipping')
                continue

            print(f'Initializing model with brain mask: {brain_mask}')
            model = FirstLevelModel(t_r=recorded_tr, mask_img=brain_mask)
            
            n_scans = nib.load(run_img).shape[3]

            frame_times = np.arange(n_scans) * recorded_tr

            # Prepare run_events df
            run_events['trial_type'] = run_events['trial_type'].str.replace('.mp4', '', regex=False)

            # Exclude 'fixation' or 'attention_getter' events if necessary
            run_events = run_events[~run_events['trial_type'].isin(['fixation', 'attention_getter'])]

            # Ensure modulation values are numeric
            if 'modulation' in run_events.columns:
                run_events['modulation'] = pd.to_numeric(run_events['modulation'], errors='coerce')

            # Create design matrix
            try:
                design = make_first_level_design_matrix(
                    frame_times,
                    events=run_events,
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
            except Exception as e:
                print(f"Error creating design matrix: {e}")
                continue

            # Plot design matrix
            try:
                fig, ax = plt.subplots(figsize=(12, 10))
                plot_design_matrix(design, ax=ax)
                plt.title(f'Design Matrix for Subject {sub} Run {runidx + 1}')
                plot_path = f'/foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/glm/design_matrices/design_matrix_{sub}_run_{runidx + 1}.png'
                plt.savefig(plot_path)
                plt.close(fig)
                print(f"Design matrix saved to {plot_path}.")

                vif = compute_vif(design)
                print("Variance Inflation Factor (VIF) for design matrix:")
                print(vif)

                # Optionally save VIF results to a file
                vif_output_path = f'/foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/glm/vif/vif_{sub}_run_{runidx + 1}.csv'
                vif.to_csv(vif_output_path, index=False)
                print(f"VIF saved to {vif_output_path}")

                # Step 2: Calculate correlation matrix
                corr_matrix = design.corr()

                # Step 3: Plot heatmap
                plt.figure(figsize=(18, 15))
                sns.heatmap(corr_matrix, cmap="coolwarm", center=0, 
                            square=True, linewidths=0.5, xticklabels=True, yticklabels=True)
                plt.title("Correlation Matrix of Full Design Matrix", fontsize=18)
                plt.tight_layout()
                plt.savefig(f'/foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/glm/confound_corr_{sub}_run_{runidx + 1}.png')
                plt.show()

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

            # Save the model
            model_save = {
                'model': model,
                'input_events': run_events,
                'funcfile': run_img,
                'fwdcutoff': fwd_cutoff,
                'design_matrix': design,
            }

            model_path = os.path.abspath(f'/foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/glm/model/{sub}_task-{task}_run-{runidx + 1}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_save, f)
            print(f"Model saved to {model_path}")
            outpaths.append(model_path)

        # Compute contrasts
        for contrast_id, contrast_val in {
            'resmem-prediction': [1]+[0]* (design.shape[1]-1),
            'rsm_difference': [0,1]+[0]* (design.shape[1]-2),
            'vca_entropy': [0,0,1]+[0]* (design.shape[1]-3),
        }.items():
            print(f" Computing contrast {contrast_id} with columns {contrast_val} ")

            contrast_vector= np.array(contrast_val)
            try: 
                beta_map = model.compute_contrast(contrast_vector, output_type='stat')
                beta_map_path = os.path.abspath(f'/foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/glm/beta_maps/{sub}_task-{task}_run-{runidx + 1}_{contrast_id}_beta_map.nii.gz')
                nib.save(beta_map, beta_map_path)
                print(f"Contrast {contrast_id} computed successfully and saved to {beta_map_path}.")
                infant_template_path ='/foundcog/forrestgump/mask/infants/nihpd_asym_02-05_fcgmask_2mm.nii.gz'
                infant_template = nib.load(infant_template_path)
                plotting.plot_stat_map(beta_map, bg_img= infant_template, title=f'Contrast {contrast_id} for Subject {sub} Run {runidx + 1}', display_mode ='ortho', cut_coords=(0, 0, 0), output_file=f'/foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/glm/contrast/{sub}_task-{task}_run-{runidx + 1}_{contrast_id}_effect_size_map.png')

            except Exception as e:
                print(f"Error computing contrast {contrast_id}: {e}")
                continue

    return outpaths

if __name__ == '__main__':
    with open('/foundcog/forrestgump/foundcog-infants-2m/subjects_2m.txt', 'r') as file:
        subject_list = [line.strip() for line in file.readlines()]

    for sub in subject_list: 
        for task in ['videos']:
            paths = model_run(sub, task, fwd_cutoff=1.5)
            print(f'Subject {sub} paths:', paths)
