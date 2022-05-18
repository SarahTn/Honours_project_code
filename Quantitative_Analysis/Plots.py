# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:46:13 2022

@author: ST
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from os import walk, listdir
import re
from os.path import join
import sys
import SimpleITK as sitk
from platipy.imaging import ImageVisualiser
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma


sys.path.append('../')
from read_in_data_pl import read_in_data
from read_in_maps import read_in_adcmap, read_in_seg_ivim_maps, read_in_seg_bval_thres_maps, read_in_overseg_maps, read_in_overseg_bval_thres_maps


# Apply the default theme
sns.set_theme(style="darkgrid")

# Read in data
data_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\Pipeline\Quantitative_Analysis\stats_results_final_raw.xlsx"
data = pd.read_excel(data_dir)
# data_test_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\Pipeline\Quantitative_Analysis\galaxies.csv"
# data_test = pd.read_csv(data_test_dir)
# data_test.info()
# data_test.head()
# data.head()


#%%
def plot_data(data_dir, param, tissue, measure, cv_ls):
    '''
    

    Parameters
    ----------
    data_dir : path to csv file
        DESCRIPTION.
    param : string
        DESCRIPTION.
    tissue : string: 'tumour' or 'benign'
        DESCRIPTION.
    measure: string: 'median', 'std', etc.
            DESCRIPTION.
            
    Assumes timepoints are in chronological order

    Returns
    -------
    Plot of parameter value at different timepoints for each patient (i.e. produces 10 subplots). It also plots the uncertainty region in orange. 

    '''
    # Get data
    columns = ['patient', 'timepoint', 'tissue', 'parameter', 'measure', 'value']
    param = data.loc[data['parameter'] == param]
    
    # For subplots
    # fig, axes = plt.subplots(3,3,figsize=(20,16))
    fig, axes = plt.subplots(2,2,figsize=(20,16))

    fig.suptitle("Heterogeneity of ADC Values in Tumour Tissue at Different Timepoints", size = 38)
    
    ptx = [5,6,7,8]
    axis = 0
    
    # for pt in range(1,len(np.unique(param['patient']))+1):
    for pt in ptx:
            
        param_pt1 = param.loc[param['patient'] == pt]
        param_tissue = param_pt1.loc[param_pt1['tissue'] == tissue]        
        param_tissue_measure = param_tissue.loc[param_tissue['measure'] == measure]
        
        # Compute uncertainty bounds 
        preRT2val = param_tissue_measure["value"].iloc[1]
        cv_val = (cvs[pt]/100)*preRT2val
        uncert_bd1 = [preRT2val+cv_val for tp in range(len(param_tissue_measure["timepoint"]) - 1)]
        uncert_bd2 = [preRT2val - cv_val for tp in range(len(param_tissue_measure["timepoint"]) - 1)]
        
        # preRT1val = param_tissue_measure["values"].iloc[0]
        # preRT2val = param_tissue_measure["values"].iloc[1]
        # diff = preRT2val - preRT1val
        # uncert_bd1 = [preRT1val + diff for tp in range(len(param_tissue_measure["timepoint"]))]
        # uncert_bd2 = [preRT1val - diff for tp in range(len(param_tissue_measure["timepoint"]))]
        
        # uncert_bd1 = [param_tissue_measure["values"].iloc[1] for tp in range(len(param_tissue_measure["timepoint"]))]   # Bound 1 = param value at preRT2
        # uncert_bd2 = [param_tissue_measure["values"].iloc[0] - (uncert_bd1[tp] - param_tissue_measure["values"].iloc[0]) for tp in range(len(param_tissue_measure["timepoint"]))]   # Bound 2 = param value at preRT1
        
        
        ax = axes.ravel()
        # ax[axis].lineplot(data=data, x=param_tissue_measure["timepoint"].iloc[1:], y=param_tissue_measure["value"].iloc[1:])
        ax[axis].plot(param_tissue_measure["timepoint"].iloc[1:], param_tissue_measure["value"].iloc[1:],linewidth=5)
        ax[axis].plot(param_tissue_measure["timepoint"].iloc[1:], uncert_bd1, '--', color='orange', linewidth=2)
        ax[axis].plot(param_tissue_measure["timepoint"].iloc[1:], uncert_bd2, '--', color='orange', linewidth=2)
        
        # if preRT2val >= preRT1val:
        #     ax[pt-1].fill_between(param_tissue_measure["timepoint"], uncert_bd2[1],  uncert_bd1[0], alpha=0.2, color='orange')
        # else:
        #     ax[pt-1].fill_between(param_tissue_measure["timepoint"], uncert_bd2[0],  uncert_bd1[1], alpha=0.2, color='orange')
        # ax[pt-1].set_title("Patient {}".format(pt), size=22)
        # ax[pt-1].set_ylabel("ADC ($mm^2/s$)", size=18)
        # ax[pt-1].tick_params(axis='x', labelsize=18)
        # ax[pt-1].tick_params(axis='y', labelsize=18)
        ax[axis].fill_between(param_tissue_measure["timepoint"].iloc[1:], uncert_bd2[0],  uncert_bd1[1], alpha=0.2, color='orange')
        ax[axis].set_title("Patient {}".format(pt), size=22)
        # ax[axis].set_ylabel("ADC ($mm^2/s$)", size=18)
        ax[axis].tick_params(axis='x', labelsize=18)
        ax[axis].tick_params(axis='y', labelsize=18)
        
        axis += 1


    plt.tight_layout()
    plt.show()

cvs = [2.98, 	1.11,	1.11,	12.55,	1.21,	1.92,	1.20,	2.35,	0.98,	0.72]

param = 'adc'
tissue = 'tumour'
measure = 'range'
plot_data(data_dir, param, tissue, measure, cvs)

#%%

def disp_ADC_map(patient_maps_dir):
    '''
    Display ADC map for a given patient at different timepoints. 

    Returns
    -------
    None.

    '''
    return 



in_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\SiBiRT2_Data'
prostatemasks_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\6_segmentation_to_follow_dispfield'
registrations_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\2_registration_follow_to_base'

map_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\Pipeline\Derived_SI-BiRT(2)_data'
tumourmasks_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\5_segmentation_baseline_tumour_b1600_autothresh"
coreg_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\3_registration_base_to_follow_dispfield"



ptx = []    # Empty list to store patients
for root, dirs, files in walk(in_dir):
    for d in dirs:
        if '3497' in d:
            ptx.append(d)
ptx = np.unique(np.array(ptx))

p = ptx[0]
ptx_dir = join(in_dir, p)
timepoints = listdir(ptx_dir)
study_dates = [re.findall(r'\d+', tp)[0] for tp in timepoints]


timepoints = listdir(ptx_dir)
timepoint_names = [(tp[9:]).replace("_","") for tp in timepoints]
study_dates = [re.findall(r'\d+', tp)[0] for tp in timepoints]

# Get tumour mask for current patient and timepoint
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
    


for tp in range(len(timepoints)):
    tp = 0
    data_dir = join(ptx_dir, timepoints[tp])

    # print(data_dir)
    ptx_maps_dir = join(map_dir, p, timepoints[tp])
    
    # Read in maps
    
    adcmap_prostate, rmse_map_image, aic_map_image = read_in_adcmap(ptx_maps_dir) 
    
    # D_seg_map, Dstar_seg_map, f_seg_map = read_in_seg_ivim_maps(ptx_maps_dir)
    # D_seg_b_thres_map, Dstar_seg_b_thres_map, f_seg_b_thres_map, bval_thres_map_seg = read_in_seg_bval_thres_maps(ptx_maps_dir)
    
    # D_overseg_map, Dstar_overseg_map, f_overseg_map = read_in_overseg_maps(ptx_maps_dir)
    # D_overseg_b_thres_map, Dstar_overseg_b_thres_map, f_overseg_b_thres_map, bval_thres_map_overseg = read_in_overseg_bval_thres_maps(ptx_maps_dir)

    # Read in data
    dw_image1, bvals1, dw_image2, bvals2, adc_image, adcimage_2, t2w3d_image = read_in_data(data_dir)
    adcmap = sitk.GetArrayFromImage(adc_image) 
    dw_array = sitk.GetArrayFromImage(dw_image1) 
    # adc_image = sitk.GetImageFromArray(dwi_afterdenoising)

    # # Denoise signals ADC
    dwi_fordenoising = np.moveaxis(adcmap,(0,1,2),(2,0,1))  # need (rows (y),columns (x),z,b) to move axis for nlmeans filter
    sigma = estimate_sigma(dwi_fordenoising, N=16)
    denoised_S = nlmeans(dwi_fordenoising, sigma=sigma, mask=None, patch_radius=1, block_radius=2, rician=True)
    dwi_afterdenoising = np.moveaxis(denoised_S, (0,1,2),(1,2,0)) # (b,z,y,x)
    # adc_image = sitk.GetImageFromArray(dwi_afterdenoising)
    
    # # Denoise signals DWI
    # dwi_fordenoising = np.moveaxis(adcmap,(0,1,2,3),(3,2,0,1))  # need to move axis for nlmeans filter
    # sigma = estimate_sigma(dwi_fordenoising, N=16)
    # denoised_S = nlmeans(dwi_fordenoising, sigma=sigma, mask=None, patch_radius=1, block_radius=2, rician=True)
    # dwi_afterdenoising = np.moveaxis(denoised_S, (0,1,2,3),(2,3,1,0))   # move axis back

        
    # If not preRT1 timepoint, then also apply tfm to current timepoint
    if study_dates[tp] != study_dates[0]:
        coreg_file = study_dates[0]+'_to_'+study_dates[tp]+'.tfm'
        tfm = sitk.ReadTransform(join(coreg_dir,p,coreg_file))
        tumour_mask_3dt2w_tp = sitk.Resample(tumour_mask_3dt2w_eroded, t2w3d_image, tfm, sitk.sitkNearestNeighbor, 0.0, tumour_mask_3dt2w_eroded.GetPixelID())
        benign_mask_3dt2w_tp = sitk.Resample(benign_mask_3dt2w_eroded, t2w3d_image, tfm, sitk.sitkNearestNeighbor, 0.0, benign_mask_3dt2w_eroded.GetPixelID())
    else:
        tumour_mask_3dt2w_tp = tumour_mask_3dt2w_eroded
        benign_mask_3dt2w_tp = benign_mask_3dt2w_eroded
    
    # tumour_mask_3dt2w_tp = tumour_mask_3dt2w_eroded
    # benign_mask_3dt2w_tp = benign_mask_3dt2w_eroded
    
    # Transform mask to DWI space
    tfm_file = timepoints[tp]+'_dwi_to_3dt2w.tfm'
    tfm = sitk.ReadTransform(join(registrations_dir,p,tfm_file))
    tfm_inv = tfm.GetInverse()
    tumour_mask_dwi = sitk.Resample(tumour_mask_3dt2w_tp, adc_image, tfm_inv, sitk.sitkNearestNeighbor, 0.0, tumour_mask_3dt2w_tp.GetPixelID())
    benign_mask_dwi = sitk.Resample(benign_mask_3dt2w_tp, adc_image, tfm_inv, sitk.sitkNearestNeighbor, 0.0, benign_mask_3dt2w_tp.GetPixelID())
    
       
    # flip_adcmap = [np.flipud(dwi_afterdenoising[sl,:,:]) for sl in range(dwi_afterdenoising.shape[0])]
    # rotate = [ndimage.rotate(dwi_afterdenoising[sl,:], 180) for sl in range(dwi_afterdenoising.shape[0])]
    # adc_denoised = sitk.GetImageFromArray(dwi_afterdenoising)
    # adc_denoised.CopyInformation(adc_image)
    # dwi_noisy = sitk.GetImageFromArray(dw1_array[0,:])
    # dwi_noisy.CopyInformation(adc_image)

    # dwi_denoised = sitk.GetImageFromArray(dwi_afterdenoising[0,:])
    dwi_denoised = sitk.GetImageFromArray(dwi_afterdenoising)
    dwi_denoised.CopyInformation(adc_image)
    

# cmap = plt.get_cmap('gray')
# plt.imshow(flip_adcmap, cmap=cmap)
# cmap2 = plt.get_cmap('magma')
# plt.imshow(prostate_adc[7], cmap=cmap2, alpha=.7), plt.colorbar()
# plt.show()
tumour_benign_mask = {'benign': benign_mask_dwi, 'tumour': tumour_mask_dwi, 'prostate': prostate_mask_dwi}

vis = ImageVisualiser(dwi_denoised, cut = (7, 25, 50), window=(200,2000))

# vis = ImageVisualiser(adc_denoised, cut = (7, 25, 50), window=(200,2000))
# vis.add_scalar_overlay(adcmap_prostate, colormap=plt.get_cmap('magma'))
# vis.add_contour(tumour_mask_dwi, linewidth=3, color='blue')

vis.add_contour(tumour_benign_mask, linewidth=3)
vis.show()
    





