import os
import subprocess
from glob import glob

def merge_beta_maps(input_dir, output_dir, contrast_name):
    """
    Merge averaged beta maps across subjects into a single 4D file for a specific contrast.

    Args:
        input_dir (str): Directory containing averaged beta maps.
        output_dir (str): Directory to save the merged 4D file.
        contrast_name (str): Name of the contrast (e.g., 'adult_betas_vs_rest').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all averaged beta maps for the given contrast
    beta_files = glob(os.path.join(input_dir, f"*_{contrast_name}_averaged_beta_map.nii.gz"))

    if len(beta_files) == 0:
        print(f"No files found for contrast {contrast_name} in {input_dir}.")
        return

    # Output file path
    output_file = os.path.join(output_dir, f"merged_{contrast_name}_4d.nii.gz")

    # Run fslmerge command
    try:
        print(f"Merging files for contrast {contrast_name}...")
        merge_command = ["fslmerge", "-t", output_file] + beta_files
        subprocess.run(merge_command, check=True)
        print(f"4D merged file saved to {output_file}.")
    except Exception as e:
        print(f"Error merging files for contrast {contrast_name}: {e}")

# Define input and output directories
input_dir = "/foundcog/forrestgump/foundcog-infants-2m/resmem_model/glm/beta_maps/averaged_beta_maps/"  
output_dir = "/foundcog/forrestgump/foundcog-infants-2m/resmem_model/second_level/merged_beta_maps/"   
# List of contrasts to process
contrasts = ["resmem-prediction"]

# Merge beta maps for each contrast
for contrast in contrasts:
    merge_beta_maps(input_dir, output_dir, contrast)
