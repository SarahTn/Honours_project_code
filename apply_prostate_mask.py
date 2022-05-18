# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:10:50 2022

@author: ST
"""

import SimpleITK as sitk
from os.path import join
import numpy as np


def prostate_mask_to_dwi_space(ROOT_DIR, patient, treatment_date, wk0_date, t2w3d_image, adc_image):
    
    # Read in prostate mask in 3D T2w space at week 0
    masks_dir = join(ROOT_DIR, 'SiBiRT2_Data', 'prostate_masks', 'segmentations', patient)
    prostate_mask = sitk.ReadImage(join(masks_dir, 'prostate-label.nii'))
    # mask_array = sitk.GetArrayFromImage(prostate_mask)

    # Read in tfm from DWI to 3D T2w space at current week
    registrations_dir = join(ROOT_DIR, 'SiBiRT2_Data', 'prostate_masks', 'transformations', patient)
    dwi_to_3dt2w = sitk.ReadTransform(join(registrations_dir, '{}_dwi_to_3dt2w.tfm'.format(treatment_date)))
    t2w3d_to_dwi = dwi_to_3dt2w.GetInverse()
    
        
    if treatment_date[:8] != wk0_date:
        # if not wk 0,
        coreg_dir = join(ROOT_DIR, 'SiBiRT2_Data', 'prostate_masks', 'coreg_transformations', patient)
        tfm_from_wk0 = sitk.ReadTransform(join(coreg_dir, '{}_to_{}.tfm'.format(wk0_date[:8], treatment_date[:8])))
        
        # Transform prostate mask to follow-up week
        prostate_mask_wk = sitk.Resample(prostate_mask, t2w3d_image, tfm_from_wk0, sitk.sitkNearestNeighbor, 0.0, prostate_mask.GetPixelID())
        
        # Transform prostate mask to dwi space
        prostate_mask_dwi = sitk.Resample(prostate_mask_wk, adc_image, t2w3d_to_dwi, sitk.sitkNearestNeighbor, 0.0, prostate_mask_wk.GetPixelID())
        
    else:  
        # Transform prostate mask to DWI space
        prostate_mask_dwi = sitk.Resample(prostate_mask, adc_image, t2w3d_to_dwi, sitk.sitkNearestNeighbor, 0.0, prostate_mask.GetPixelID())
        
    prostate_array1 = sitk.GetArrayFromImage(prostate_mask_dwi)
    vox = np.where(prostate_array1!=0)

    return prostate_mask_dwi, vox 