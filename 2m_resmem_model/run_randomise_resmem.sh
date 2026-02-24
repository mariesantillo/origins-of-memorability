#!/bin/bash

module load fsl  

randomise -i /foundcog/forrestgump/foundcog-infants-2m/resmem_model/second_level/merged_beta_maps/merged_resmem-prediction_4d.nii.gz \
          -o /foundcog/forrestgump/foundcog-infants-2m/resmem_model/second_level/results/oneSam_Resmem_Resmem_Model \
          -1 -T
