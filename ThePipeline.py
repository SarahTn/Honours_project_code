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
sys.path.append('../')

# Import files
from config.definitions import ROOT_DIR
from mapping_code.adc_mapping_pl import save_adc_map
from mapping_code.seg_ivim_mapping_pl import save_seg_ivim_maps
from mapping_code.seg_ivim_bval_thres_alg_mapping import save_seg_bval_thres_maps
from mapping_code.overseg_ivim_mapping_pl import save_overseg_ivim_maps, func_overseg_ivim_mapping
from mapping_code.overseg_ivim_bval_thres_alg_mapping import save_overseg_bval_thres_maps
from mapping_code.kurtosis_mapping_pl import save_kurt_map
from mapping_code.linearised_kurt_mapping_pl import save_lin_kurt_map
from read_in_data_pl import read_in_data


#%%
tic = time.perf_counter()


# Input directories
in_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\SiBiRT2_Data'
prostatemasks_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\6_segmentation_to_follow_dispfield'
registrations_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\2_registration_follow_to_base'
 
ptx = []    # Empty list to store patients
for root, dirs, files in walk(in_dir):
    for d in dirs:
        if '3497' in d:
            ptx.append(d)
ptx = np.unique(np.array(ptx))

bvals_for_adc_fit = [50, 400, 800]
bval_cutoff = 200   # For segmented and over-segmented IVIM fits
bval_threshold_ls = [50, 100, 200, 300]     # For b-value threshold alogrithm IVIM fit
count = 0

for p in ptx:
    ptx_dir = join(in_dir, p)
    timepoints = listdir(ptx_dir)
    study_dates = [re.findall(r'\d+', tp)[0] for tp in timepoints]
    for tp in range(len(timepoints)):
        data_dir = join(ptx_dir, timepoints[tp])
        print(data_dir)
        dw_image1, bvals1, dw_image2, bvals2, adc_image, adcimage_2, t2w3d_image = read_in_data(data_dir)
        
        # Get prostate mask for current patient and timepoint
        mask_file = study_dates[tp]+'_prostate-label.nii.gz'
        prostate_mask_3dt2w = sitk.ReadImage(join(prostatemasks_dir,p,mask_file))
        
        # Transform mask to DWI space
        tfm_file = timepoints[tp]+'_dwi_to_3dt2w.tfm'
        tfm = sitk.ReadTransform(join(registrations_dir,p,tfm_file))
        tfm_inv = tfm.GetInverse()
        prostate_mask_dwi = sitk.Resample(prostate_mask_3dt2w, adc_image, tfm_inv, sitk.sitkNearestNeighbor, 0.0, prostate_mask_3dt2w.GetPixelID())
        prostate_array = sitk.GetArrayFromImage(prostate_mask_dwi)
        
        # Get voxels inside prostate mask
        prostate_vox = np.array(np.where(prostate_array!=0))
        
        # If there is only one voxel in a given  prostate mask slice, we can't fit it so only take slices where there is more than one voxel
        sl, idx, counts = np.unique(prostate_vox[0], return_counts=True, return_index=True)
        
        # if 1 in counts:
        #     sl_min = np.max(np.where(counts < 2)[0]) + 1 
        #     prostate_vox = prostate_vox[:,sl_min:]
            
        if 1 in counts:
            idx_to_del = np.where(counts == 1)
            slices_to_del = sl[idx_to_del]
            sl_idx_vox = np.where(prostate_vox[0] == slices_to_del)
            prostate_vox_for_fit = np.delete(prostate_vox, sl_idx_vox, axis=1)
            
        else:
            prostate_vox_for_fit = prostate_vox
            
        
        
        
        # ADC mapping
        save_adc_map(data_dir, ROOT_DIR, bvals_for_adc_fit, prostate_vox_for_fit)
        
        # ------ IVIM mapping ------
        # Segmented fit
        save_seg_ivim_maps(data_dir, ROOT_DIR, bval_cutoff, prostate_vox_for_fit)
        
        # Segmented adaptive b-value threshold algorithm 
        save_seg_bval_thres_maps(data_dir, ROOT_DIR, bval_threshold_ls, prostate_vox_for_fit)
        
        # Over-segmented fit
        save_overseg_ivim_maps(data_dir, ROOT_DIR, bval_cutoff, prostate_vox_for_fit)
        # D_map, Dstar_map, f_map = func_overseg_ivim_mapping(data_dir, bval_cutoff, prostate_vox_for_fit)
        
        # Over-segmented adaptive b-value threshold algorithm
        save_overseg_bval_thres_maps(data_dir, ROOT_DIR, bval_threshold_ls, prostate_vox_for_fit)
        
    # count += 1
       
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
  


# #%% ADC mapping test
# bvals_for_fit = [50, 400, 800]
# adcmap = func_adcmapping(in_dir, bvals_for_fit, vox)

# # flip_adcmap = np.flipud(adcmap[7,:,:])  # Only needed for display -> sitk handles this when writing image to file 

# # Plot map
# flip_adcmap = [np.flipud(adcmap[sl,:,:]) for sl in range(adcmap.shape[0])]
# cmap = plt.get_cmap('magma')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(flip_adcmap[7], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('ADC map')