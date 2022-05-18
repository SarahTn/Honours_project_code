# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:16:59 2022

@author: ST
"""
from os import walk, listdir
from os.path import join
import sys
import numpy as np
import re
import SimpleITK as sitk
import time
from platipy.imaging import ImageVisualiser
import matplotlib.pyplot as plt
sys.path.append('../')

# Import files
from config.definitions import ROOT_DIR
from mapping_code.kurtosis_mapping_pl import save_kurt_map, func_kurtmapping
from mapping_code.linearised_kurt_mapping_pl import save_lin_kurt_map, func_lin_kurtmapping
from read_in_data_pl import read_in_data


#%%
tic = time.perf_counter()


# Input directories
in_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\SiBiRT2_Data'
tumourmasks_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\5_segmentation_baseline_tumour_b1600_autothresh"
registrations_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\2_registration_follow_to_base'
coreg_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\3_registration_base_to_follow_dispfield"
 
ptx = []    # Empty list to store patients
for root, dirs, files in walk(in_dir):
    for d in dirs:
        if '3497' in d:
            ptx.append(d)
ptx = np.unique(np.array(ptx))

# problems = []
# Problems
# [['3497-3', '20200825_preRT_1', 'single vox'],
#  ['3497-3', '20200930_RT_1', 'single vox last slice'],
#  ['3497-5', '20210216_RT_1', 'single vox last slice'],
#  ['3497-8', '20210330_RT_1', 'single vox last slice'],
#  ['3497-1', '20200825_RT_1', 'single vox, second last slice'],
#  ['3497-10', 'all', 'otsu tumour mask not found']]

bval_cutoff = 400

for p in ptx:
    ptx_dir = join(in_dir, p)
    timepoints = listdir(ptx_dir)
    study_dates = [re.findall(r'\d+', tp)[0] for tp in timepoints]
    for tp in range(len(timepoints)):
        data_dir = join(ptx_dir, timepoints[tp])

        # Read in data
        dw_image1, bvals1, dw_image2, bvals2, adc_image, adcimage_2, t2w3d_image = read_in_data(data_dir)
        
        # Get tumour mask for current patient and timepoint
        tumourmask_file = study_dates[0]+'_tumour_pred_b1600_otsu.nii.gz'
        tumour_mask_3dt2w = sitk.ReadImage(join(tumourmasks_dir,p,tumourmask_file)) 
        
        # Morphological closing
        closingFilter = sitk.BinaryMorphologicalClosingImageFilter()
        closingFilter.SetKernelRadius(1)
        tumour_mask_closed = closingFilter.Execute(tumour_mask_3dt2w)
        
        # Apply erosion
        erosionFilter = sitk.BinaryErodeImageFilter()
        erosionFilter.SetKernelRadius(1)
        erosionFilter.SetForegroundValue(1)
        tumour_mask_3dt2w_eroded = erosionFilter.Execute(tumour_mask_closed)
        
        
        # If not preRT1 timepoint, then also apply tfm to current timepoint
        if study_dates[tp] != study_dates[0]:
            coreg_file = study_dates[0]+'_to_'+study_dates[tp]+'.tfm'
            tfm = sitk.ReadTransform(join(coreg_dir,p,coreg_file))
            tumour_mask_3dt2w_tp = sitk.Resample(tumour_mask_3dt2w_eroded, t2w3d_image, tfm, sitk.sitkNearestNeighbor, 0.0, tumour_mask_3dt2w_eroded.GetPixelID())
            
            # tumour_m_no_erosion = sitk.Resample(tumour_mask_3dt2w, t2w3d_image, tfm, sitk.sitkNearestNeighbor, 0.0, tumour_mask_3dt2w.GetPixelID())
        
        else:
            tumour_mask_3dt2w_tp = tumour_mask_3dt2w_eroded
        
        # Transform mask to DWI space
        tfm_file = timepoints[tp]+'_dwi_to_3dt2w.tfm'
        tfm = sitk.ReadTransform(join(registrations_dir,p,tfm_file))
        tfm_inv = tfm.GetInverse()
        tumour_mask_dwi = sitk.Resample(tumour_mask_3dt2w_tp, adc_image, tfm_inv, sitk.sitkNearestNeighbor, 0.0, tumour_mask_3dt2w_tp.GetPixelID())
        
        # tumour_m_dwi_no_erosion = sitk.Resample(tumour_m_no_erosion, adc_image, tfm_inv, sitk.sitkNearestNeighbor, 0.0, tumour_m_no_erosion.GetPixelID())
        
        tumour_array = sitk.GetArrayFromImage(tumour_mask_dwi)
        
        # Get voxels inside prostate mask
        tumour_vox = np.array(np.where(tumour_array!=0))
        
        # If there is only one voxel in a given  tumour mask slice, we can't fit it so only take slices where there is more than one voxel
        sl, idx, counts = np.unique(tumour_vox[0], return_counts=True, return_index=True)
        
        if 1 in counts:
            idx_to_del = np.where(counts == 1)
            slices_to_del = sl[idx_to_del]
            sl_idx_vox = np.where(tumour_vox[0] == slices_to_del)
            tumour_vox_for_fit = np.delete(tumour_vox, sl_idx_vox, axis=1)
        else:
            tumour_vox_for_fit = tumour_vox
                    
        
        # Kurtosis mapping
        # Dappmap, Kappmap, rmse_map, aic_map = func_kurtmapping(data_dir, bval_cutoff, tumour_vox_for_fit)
        # Dappmap_lin, Kappmap_lin, rmse_map_lin, aic_map_lin = func_lin_kurtmapping(data_dir, bval_cutoff, tumour_vox_for_fit)

        save_kurt_map(data_dir, ROOT_DIR, bval_cutoff, tumour_vox_for_fit)
        save_lin_kurt_map(data_dir, ROOT_DIR, bval_cutoff, tumour_vox_for_fit)
        
    #     count += 1
        
    #     if count == 1:
    #         break
        
    # if count == 1:
    #     break

        
toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
print(f"Downloaded the tutorial in {(toc - tic)/60:0.1f} minutes")  # 2.1 minutes for 1 patient timepoint

  
        

#%% Read in data
# patient = '3497-1'
# treatment_date = '20200804_preRT_2'
# wk0_date = '20200721'
# in_dir = join(ROOT_DIR, 'SiBiRT2_Data', patient, treatment_date)
# dw_image1, bvals1, dw_image2, bvals2, adc_image, adcimage_2, t2w3d_image = read_in_data(in_dir)

# #%% Get prostate mask in dwi space

# prostate_mask_dwi, vox = prostate_mask_to_dwi_space(ROOT_DIR, patient, treatment_date, wk0_date, t2w3d_image, adc_image)

# # image_visualiser = ImageVisualiser(adc_image, window=(200,2000))
# # image_visualiser.add_contour(prostate_mask_dwi)
# # fig1 = image_visualiser
  
# image_visualiser = ImageVisualiser(tumour_mask_dwi, cut=(6,25,50), window=(0,1))
# # image_visualiser.add_contour(prostate_mask_dwi)
# fig1 = image_visualiser
# fig1.show()

# cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(Kappmap[8], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('K map Overseg IVIM')

# cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(aic_map[8], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('aic_map Overseg IVIM')



# #%% ADC mapping test
# bvals_for_fit = [50, 400, 800]
# adcmap = func_adcmapping(in_dir, bvals_for_fit, vox)

# # flip_adcmap = np.flipud(adcmap[7,:,:])  # Only needed for display -> sitk handles this when writing image to file 

# # Plot map
# flip_adcmap = [np.flipud(adcmap[sl,:,:]) for sl in range(adcmap.shape[0])]
# cmap = plt.get_cmap('magma')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(Kappmap[7], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('Kapp map')