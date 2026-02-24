import os
import nibabel as nib
import numpy as np
from glob import glob

def average_beta_maps(input_dir, output_dir, contrast_name):
    """
    Average beta maps across runs for each subject.

    Args:
        input_dir (str): Directory containing beta map NIFTI files.
        output_dir (str): Directory to save averaged beta maps.
        contrast_name (str): Name of the contrast (e.g., 'adult_betas_vs_rest').

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all beta map files for the given contrast
    beta_files = glob(os.path.join(input_dir, f"*_{contrast_name}_beta_map.nii.gz"))

    # Organize files by subject
    subject_dict = {}
    for beta_file in beta_files:
        file_name = os.path.basename(beta_file)
        subject = file_name.split("_")[0]  # Assumes 'sub-{subject}' in the filename
        if subject not in subject_dict:
            subject_dict[subject] = []
        subject_dict[subject].append(beta_file)

    # Average beta maps across runs for each subject
    for subject, files in subject_dict.items():
        print(f"Processing {subject} for contrast {contrast_name}: {len(files)} runs found.")

        # Load and stack beta maps
        beta_images = [nib.load(f) for f in files]
        beta_data = np.stack([img.get_fdata() for img in beta_images], axis=-1)

        # Compute the average across runs
        averaged_data = np.mean(beta_data, axis=-1)

        # Save the averaged beta map
        averaged_img = nib.Nifti1Image(averaged_data, beta_images[0].affine, beta_images[0].header)
        output_path = os.path.join(output_dir, f"{subject}_{contrast_name}_averaged_beta_map.nii.gz")
        nib.save(averaged_img, output_path)
        print(f"Averaged beta map saved for {subject}: {output_path}")

# Define input and output directories
input_dir = "/foundcog/forrestgump/foundcog-infants-2m/resmem_model/glm/beta_maps/"  # Replace with the path to your beta maps folder
output_dir = "/foundcog/forrestgump/foundcog-infants-2m/resmem_model/glm/beta_maps/averaged_beta_maps/"  # Replace with the path for saving averaged maps

# Define contrasts to process
contrasts = [ "resmem-prediction"]

# Process each contrast
for contrast in contrasts:
    average_beta_maps(input_dir, output_dir, contrast)
