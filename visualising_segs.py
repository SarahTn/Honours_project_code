# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 21:59:21 2022

@author: slizz
"""
from os import walk, listdir
from os.path import join
import sys
import numpy as np
import re
import SimpleITK as sitk
import time
import pandas as pd
sys.path.append('../')
from platipy.imaging import ImageVisualiser
# Import files
from config.definitions import ROOT_DIR
from mapping_code.adc_mapping_pl import save_adc_map
# from mapping_code.seg_ivim_mapping_pl import save_seg_ivim_maps
# from mapping_code.seg_ivim_bval_thres_alg_mapping import save_seg_bval_thres_maps
# from mapping_code.overseg_ivim_mapping_pl import save_overseg_ivim_maps, func_overseg_ivim_mapping
# from mapping_code.overseg_ivim_bval_thres_alg_mapping import save_overseg_bval_thres_maps
from read_in_data_pl import read_in_data
from read_in_maps import read_in_adcmap

#%%
def resample(image, tfm, dest_image):
    size = image.GetSize()
    extractor = sitk.ExtractImageFilter()
    new_image = []
    for b in range(size[3]):
        index = [0,0,0,b]
        extractor.SetIndex(index)
        extractor.SetSize(size[0:2])
        b_image = extractor.Execute(image)
        resampled_b_image = sitk.Resample(b_image, dest_image, tfm, sitk.sitkNearestNeighbor, 0.0, b_image.GetPixelID())
        new_image.append(resampled_b_image)
    joinFilter = sitk.JoinSeriesImageFilter()
    return(joinFilter.Execute(new_image))

#%%
def transform_to_3d_baseline(registrations_dir, p, timepoints, tp, parametricmap, destimage_wk, destimage_0):
    # Transform map to 3D T2w - rigid registration
    tfm_file = timepoints[tp]+'_dwi_to_3dt2w.tfm'
    tfm_wk = sitk.ReadTransform(join(registrations_dir,p,tfm_file))       
    map_to_3dt2w = sitk.Resample(parametricmap, destimage_wk, tfm_wk, sitk.sitkNearestNeighbor, 0.0, parametricmap.GetPixelID())
    
    if tp != 0:
        coreg_file = timepoints[tp]+'_to_wk0.tfm'
        tfm_0 = sitk.ReadTransform(join(registrations_dir,p,coreg_file))
        map_to_3dt2w_wk0 = sitk.Resample(map_to_3dt2w, destimage_0, tfm_0, sitk.sitkNearestNeighbor, 0.0, map_to_3dt2w.GetPixelID())
    else:
        map_to_3dt2w_wk0 = map_to_3dt2w
        
    return(map_to_3dt2w_wk0)
#%%

def get_masks(tumourmasks_dir, p, study_dates):
    # Get tumour and benign mask for current patient at preRT1
    tumourmask_file = study_dates[0]+'_tumour_pred_b1600_otsu.nii.gz'
    benignmask_file = study_dates[0]+'_background_pred_b1600_otsu.nii.gz'
    tumour_mask_3dt2w = sitk.ReadImage(join(tumourmasks_dir,p,tumourmask_file))
    benign_mask_3dt2w = sitk.ReadImage(join(tumourmasks_dir,p,benignmask_file))
    
    # Morphological closing
    closingFilter = sitk.BinaryMorphologicalClosingImageFilter()
    closingFilter.SetKernelRadius(1)
    benign_mask_closed = closingFilter.Execute(benign_mask_3dt2w)
    tumour_mask_closed = closingFilter.Execute(tumour_mask_3dt2w)
    
    # Apply erosion
    erosionFilter = sitk.BinaryErodeImageFilter()
    erosionFilter.SetKernelRadius(1)
    erosionFilter.SetForegroundValue(1)
    tumour_mask_3dt2w_eroded = erosionFilter.Execute(tumour_mask_closed)
    benign_mask_3dt2w_eroded = erosionFilter.Execute(benign_mask_closed)
    
    return(tumour_mask_3dt2w_eroded, benign_mask_3dt2w_eroded)

#%%
def get_stats(arr):
    mean = np.mean(arr)
    stdev = np.std(arr)
    median = np.median(arr)
    iqr = np.percentile(arr,75) - np.percentile(arr,25)
    heterogeneity = 100*iqr/median
    return(mean, stdev, median, iqr, heterogeneity)

#%%
def save_roi_stats(tumour_mask, benign_mask, parametricmap, timepoint_names, parameter_names, out_dir):
    results_table = []
    columns = ['patient', 'timepoint','tissue','parameter','measure','value']
    measures = ['mean', 'stdev', 'median', '10th_perc', '90th_perc', 'iqr', 'heterogeneity']
    
    tumour_label = sitk.GetArrayFromImage(tumour_mask)
    benign_label = sitk.GetArrayFromImage(benign_mask)
    
    tumour_vox = np.extract(tumour_label, parametricmap)
    benign_vox = np.extract(benign_label, parametricmap)        
    
    # Call function which extracts mean, 'stdev', median, IQR, heterogeneity, skewness and writes into csv file using pandas
    tumour_stats = get_stats(tumour_vox) # column array
    pt_info = np.tile([p[5:], timepoint_names[tp], 'tumour', '{}'.format(parameter_names)],(len(measures),1))
    data_add = np.hstack((pt_info, np.reshape(measures,(len(measures),1)), np.reshape(tumour_stats,(len(measures),1))))
    results_table.append(data_add)
    
    benign_stats = get_stats(benign_vox) # column array
    pt_info = np.tile([p[5:], timepoint_names[tp], 'benign', '{}'.format(parameter_names)],(len(measures),1))
    data_add = np.hstack((pt_info, np.reshape(measures,(len(measures),1)), np.reshape(benign_stats,(len(measures),1))))
    results_table.append(data_add)
    
    df = pd.DataFrame(np.concatenate(results_table, axis=0), index = None, columns = columns)
    xlsfile = join(out_dir, 'stats_results_pt6.xlsx')
    with pd.ExcelWriter(xlsfile) as writer:
        df.to_excel(writer, index=None, columns=columns)
        
    return()
        
    
#%%
# Input directories
in_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\SiBiRT2_Data'
prostatemasks_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\6_segmentation_to_follow_dispfield'
registrations_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\2_registration_follow_to_base'
coreg_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\3_registration_base_to_follow_dispfield"

out_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\Pipeline\Quantitative_Analysis"
map_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\Pipeline\Derived_SI-BiRT(2)_data'
tumourmasks_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\5_segmentation_baseline_tumour_b1600_autothresh"

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


#%%
for p in ptx:
    ptx_dir = join(in_dir, p)
    timepoints = listdir(ptx_dir)
    study_dates = [re.findall(r'\d+', tp)[0] for tp in timepoints]
    for tp in range(len(timepoints)):
        data_dir = join(ptx_dir, timepoints[tp])
        ptx_maps_dir = join(map_dir, p, timepoints[tp])
        dw_image1, bvals1, dw_image2, bvals2, adc_image, adcimage_2, t2w3d_image = read_in_data(data_dir)
        
        # Read in parametric maps 
        adcmap, rmse_map_image, aic_map_image = read_in_adcmap(ptx_maps_dir) 
        
        # Transform to baseline 3D space
        if tp != 0:
            dummy1, dummy2, dummy3, dummy4,  adc_image_0, adcimage_2_0, t2w3d_image_0 = read_in_data(join(ptx_dir, timepoints[0]))
        else:
            t2w3d_image_0 = t2w3d_image
            
        adc_to_3dt2w_wk0 = transform_to_3d_baseline(registrations_dir, p, timepoints, tp, adcmap, t2w3d_image, t2w3d_image_0)
        
        # Get masks
        tumour_mask, benign_mask = get_masks(tumourmasks_dir, p, study_dates)
        
        # Extract ROI stats
        parameter_array = sitk.GetArrayFromImage(adc_to_3dt2w_wk0)
        # save_roi_stats(tumour_mask, benign_mask, parameter_array, out_dir)

vis = ImageVisualiser(adc_to_3dt2w_wk0, cut=(140,128,44), window=(200,2000), axis='z')
# vis.add_contour(adcmap_prostate, color='blue')
vis.add_contour(tumour_mask, color='red', linewidth=3)
# cmap=plt.get_cmap('magma')
# vis.add_scalar_overlay(adcmap_prostate, colormap=cmap)
vis.show()