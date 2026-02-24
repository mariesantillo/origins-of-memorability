#!/bin/bash

module load fsl  

randomise -i /foundcog/forrestgump/foundcog-adult-2/resmem_stranet_vca/second_level/merged_beta_maps/merged_resmem_4d.nii.gz \
          -o /foundcog/forrestgump/foundcog-adult-2/resmem_stranet_vca/second_level/results/oneSam_Resmem_Resmem_STRANET_rsm_VCA_Model \
          -1 -T

randomise -i /foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/second_level/merged_beta_maps/merged_rsm_difference_4d.nii.gz \
          -o /foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/second_level/results/oneSam_attention_rsm_Resmem_STRANET_rsm_VCA_Model \
          -1 -T

randomise -i /foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/second_level/merged_beta_maps/merged_vca_entropy_4d.nii.gz \
          -o /foundcog/forrestgump/foundcog-infants-2m/resmem_vca_stranet_rsm_model/second_level/results/oneSam_vca_entropy_Resmem_STRANET_rsm_VCA_Model \
          -1 -T
