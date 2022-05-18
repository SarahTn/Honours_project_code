

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:16:59 2022

@author: ST
"""
from os import walk, listdir
from os.path import join
import sys
import numpy as np
from read_in_data_pl import read_in_data
import re
import SimpleITK as sitk
import time
import matplotlib.pyplot as plt
from platipy.imaging import ImageVisualiser
from scipy.stats import skew, kurtosis, entropy
sys.path.append('../')

# Import files
from config.definitions import ROOT_DIR
from mapping_code.adc_mapping_pl import save_adc_map
from read_in_maps import read_in_adcmap, read_in_seg_ivim_maps, read_in_seg_bval_thres_maps, read_in_overseg_maps, read_in_overseg_bval_thres_maps, read_in_kurt_maps, read_in_linearied_kurt_maps

#%%
def get_stats(arr):
    maximum = np.max(arr)
    minimum = np.min(arr)
    Range = maximum - minimum
    mean = np.mean(arr)
    variance = np.var(arr)
    stdev = np.std(arr)
    median = np.median(arr)
    tenth_perc = np.percentile(arr, 10)
    ninetieth_perc = np.percentile(arr, 90)
    iqr = np.percentile(arr,75) - np.percentile(arr,25)
    skewness = skew(arr)
    kurt = kurtosis(arr)
    heterogeneity = 100*iqr/median
    ent = entropy(arr)
    root_mean_sqrd = np.sqrt(np.mean(arr**2))
    mean_abs_dev = np.sum(abs(arr - mean)/len(arr))
    
    ls = [ele >= tenth_perc and ele <= ninetieth_perc for ele in arr]
    subset_10_to_90_perc = arr[ls]
    robust_mean_abs_dev = np.sum(abs(subset_10_to_90_perc - mean)/len(arr))
    
    return(maximum, minimum, Range, mean, variance, stdev, median, tenth_perc, ninetieth_perc, iqr, skewness, kurt, heterogeneity, ent, root_mean_sqrd, mean_abs_dev, robust_mean_abs_dev)



#%%
tic = time.perf_counter()


# Input directories
in_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\SiBiRT2_Data'
map_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\Pipeline\Derived_SI-BiRT(2)_data'
tumourmasks_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\5_segmentation_baseline_tumour_b1600_autothresh"
registrations_dir = r'C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\2_registration_follow_to_base'
coreg_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\sibirt2(1)\3_registration_base_to_follow_dispfield"


ptx = []    # Empty list to store patients
for root, dirs, files in walk(map_dir):
    for d in dirs:
        if '3497' in d:
            ptx.append(d)
ptx = np.unique(np.array(ptx))
count = 0

results_table = []
columns = ['Patient', 'Timepoint','Tissue','Parameter','Measure','Value']
measures = ['Maximum', 'Minimum', 'Range', 'Mean', 'Variance', 'Stdev', 'Median', '10th Percentile', '90th Percentile', 'IQR', 'Skewness', 'Kurtosis', 'Heterogeneity', 'Entropy', 'Root Mean Squared', 'Mean AbsDev', 'Robust Mean AbsDev']


for p in ptx:
    ptx_data_dir = join(in_dir, p)
    timepoints = listdir(ptx_data_dir)
    timepoint_names = [(tp[9:]).replace("_","") for tp in timepoints]
    study_dates = [re.findall(r'\d+', tp)[0] for tp in timepoints]
    for tp in range(len(timepoints)):
        data_dir = join(ptx_data_dir, timepoints[tp])
        ptx_maps_dir = join(map_dir, p, timepoints[tp])
        
        # Read in parametric maps 
        adcmap, rmse_map_adc, aic_map_adc = read_in_adcmap(ptx_maps_dir) 
        
        D_seg_map, Dstar_seg_map, f_seg_map, rmse_seg_map, aic_seg_map = read_in_seg_ivim_maps(ptx_maps_dir)
        D_seg_b_thres_map, Dstar_seg_b_thres_map, f_seg_b_thres_map, bval_thres_map_seg, rmse_seg_b_thres_map, aic_seg_b_thres_map = read_in_seg_bval_thres_maps(ptx_maps_dir)
        
        D_overseg_map, Dstar_overseg_map, f_overseg_map, rmse_overseg_map, aic_overseg_map = read_in_overseg_maps(ptx_maps_dir)
        D_overseg_b_thres_map, Dstar_overseg_b_thres_map, f_overseg_b_thres_map, bval_thres_map_overseg, rmse_overseg_b_thres_map, aic_overseg_b_thres_map = read_in_overseg_bval_thres_maps(ptx_maps_dir)
        
        Dapp_kurt_map, Kapp_kurt_map, rmse_kurt_map, aic_kurt_map = read_in_kurt_maps(ptx_maps_dir)
        Dapp_lin_kurt_map, Kapp_lin_kurt_map, rmse_lin_kurt_map, aic_lin_kurt_map = read_in_linearied_kurt_maps(ptx_maps_dir)
        
        parametric_maps = [adcmap, rmse_map_adc, aic_map_adc, D_seg_map, Dstar_seg_map, f_seg_map, rmse_seg_map, aic_seg_map, D_seg_b_thres_map, Dstar_seg_b_thres_map, f_seg_b_thres_map, bval_thres_map_seg, rmse_seg_b_thres_map, aic_seg_b_thres_map, D_overseg_map, Dstar_overseg_map, f_overseg_map, rmse_overseg_map, aic_overseg_map, D_overseg_b_thres_map, Dstar_overseg_b_thres_map, f_overseg_b_thres_map, bval_thres_map_overseg, rmse_overseg_b_thres_map, aic_overseg_b_thres_map, Dapp_kurt_map, Kapp_kurt_map, rmse_kurt_map, aic_kurt_map, Dapp_lin_kurt_map, Kapp_lin_kurt_map, rmse_lin_kurt_map, aic_lin_kurt_map]
        
        parameter_names = ['ADC', 'RMSE adc', 'AIC adc', 'D seg', 'D* seg', 'f seg', 'RMSE seg', 'AIC seg', 'D seg b-thres', 'D* seg b-thres', 'f seg b-thres', 'b-thres seg', 'RMSE seg b-thres', 'AIC seg b-thres', 'D overseg', 'D* overseg', 'f overseg', 'RMSE overseg', 'AIC overseg', 'D overseg b-thres', 'D* overseg b-thres', 'f overseg b-thres', 'b-thres overseg', 'RMSE overseg b-thres', 'AIC overseg b-thres', 'D kurt', 'K kurt', 'RMSE kurt', 'AIC kurt', 'D lin kurt', 'K lin kurt', 'RMSE lin kurt', 'AIC lin kurt']
                
        # Read in data
        dw_image1, bvals1, dw_image2, bvals2, adc_image, adcimage_2, t2w3d_image = read_in_data(data_dir)
        
                
        # Get tumour mask for current patient and timepoint 0 
        tumourmask_file = study_dates[0]+'_tumour_pred_b1600_otsu.nii.gz'
        benignmask_file = study_dates[0]+'_background_pred_b1600_otsu.nii.gz'
        tumour_mask_3dt2w = sitk.ReadImage(join(tumourmasks_dir,p,tumourmask_file))
        benign_mask_3dt2w = sitk.ReadImage(join(tumourmasks_dir,p,benignmask_file))
        
        # Apply fill holes filter
        # fill_holesFilter = sitk.BinaryFillholeImageFilter()
        # fill_holesFilter.SetForegroundValue( 1 )
        # fill_holesFilter.SetFullyConnected(True)
        # tumour_mask_3dt2w_fillhole = fill_holesFilter.Execute(tumour_mask_3dt2w)
        # benign_mask_3dt2w_fillhole = fill_holesFilter.Execute(benign_mask_3dt2w)
        
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
        
        # If not preRT1 timepoint, then also apply tfm to current timepoint
        if study_dates[tp] != study_dates[0]:
            coreg_file = study_dates[0]+'_to_'+study_dates[tp]+'.tfm'
            tfm = sitk.ReadTransform(join(coreg_dir,p,coreg_file))
            tumour_mask_3dt2w_tp = sitk.Resample(tumour_mask_3dt2w_eroded, t2w3d_image, tfm, sitk.sitkNearestNeighbor, 0.0, tumour_mask_3dt2w_eroded.GetPixelID())
            benign_mask_3dt2w_tp = sitk.Resample(benign_mask_3dt2w_eroded, t2w3d_image, tfm, sitk.sitkNearestNeighbor, 0.0, benign_mask_3dt2w_eroded.GetPixelID())
        else:
            tumour_mask_3dt2w_tp = tumour_mask_3dt2w_eroded
            benign_mask_3dt2w_tp = benign_mask_3dt2w_eroded
        
        # Transform mask to DWI space
        tfm_file = timepoints[tp]+'_dwi_to_3dt2w.tfm'
        tfm = sitk.ReadTransform(join(registrations_dir,p,tfm_file))
        tfm_inv = tfm.GetInverse()
        tumour_mask_dwi = sitk.Resample(tumour_mask_3dt2w_tp, adc_image, tfm_inv, sitk.sitkNearestNeighbor, 0.0, tumour_mask_3dt2w_tp.GetPixelID())
        benign_mask_dwi = sitk.Resample(benign_mask_3dt2w_tp, adc_image, tfm_inv, sitk.sitkNearestNeighbor, 0.0, benign_mask_3dt2w_tp.GetPixelID())
        
        

        tumour_array = sitk.GetArrayFromImage(tumour_mask_dwi)
        benign_array = sitk.GetArrayFromImage(benign_mask_dwi)
        
        # Get voxels within the masks
        for m in range(len(parametric_maps)):
            map_arr = sitk.GetArrayFromImage(parametric_maps[m])
            tumour_label = sitk.GetArrayFromImage(tumour_mask_dwi)
            benign_label = sitk.GetArrayFromImage(benign_mask_dwi)
            
            tumour_vox = np.extract(tumour_label, map_arr)
            benign_vox = np.extract(benign_label, map_arr)        
            
            # Call function which extracts mean, 'stdev', median, IQR, heterogeneity, skewness and writes into csv file using pandas
            tumour_stats = get_stats(tumour_vox) # column array
            pt_info = np.tile([p[5:], timepoint_names[tp], 'Tumour', '{}'.format(parameter_names[m])],(len(measures),1))
            data_add = np.hstack((pt_info, np.reshape(measures,(len(measures),1)), np.reshape(tumour_stats,(len(measures),1))))
            results_table.append(data_add)
            
            # Kurtosis fitting performed in tumour only due to noise floor considerations
            if 'kurt' not in parameter_names[m]:
                benign_stats = get_stats(benign_vox) # column array
                pt_info = np.tile([p[5:], timepoint_names[tp], 'Benign', '{}'.format(parameter_names[m])],(len(measures),1))
                data_add = np.hstack((pt_info, np.reshape(measures,(len(measures),1)), np.reshape(benign_stats,(len(measures),1))))
            results_table.append(data_add)
            
        
# plt histograms of tumour and benign 
# fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
# fig.suptitle('Apparent Diffusion Coefficient Histograms for Patient 1, Timepoint 2')

# # We can set the number of bins with the *bins* keyword argument
# n_bins = 25
# axs[0].hist(tumour_vox, bins=n_bins, color='r')
# axs[0].title.set_text('Tumour Tissue')
# axs[1].hist(benign_vox, bins=n_bins, color='tab:purple')
# axs[1].title.set_text('Benign Tissue')
# fig.text(0.5, 0.04, 'ADC $mm^2/s$', ha='center')
# fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical')


# plt.hist(tumour_vox, bins=n_bins, color='blue')
# plt.xlabel('ADC $10^{-6} mm^2/s$', fontsize=14)
# plt.ylabel('Count', fontsize=14)
# plt.title('Apparent Diffusion Coefficient Tumour Tissue Histogram', fontsize=14)



        ###
        # Pt ID, timepoint, tissue, parameter, type, value
        
### Create separate pipeline for kurtosis mapping in the tumour regions only - append to above csv file. 
  # Pt ID, timepoint, tissue (tumour only), parameter, type, value
        
#### ------------------------------------------- do everything else after if you have time        
        
        # Call function to perform radiomics using pyradiomics
        
        
        # ADC mapping
        # bvals_for_fit = [50, 400, 800]
        # save_map = save_adc_map(data_dir, ROOT_DIR, bvals_for_fit, vox)
        
        # # ------ IVIM mapping ------
        # # Segmented fit
        # save_seg_ivim_map(data_dir, ROOT_DIR, bval_cutoff, vox)
        
        # # Over-segmented fit
        # save_overseg_ivim_map(data_dir, ROOT_DIR, bval_cutoff, vox)
        
        # # Adaptive threshold algorithm 
        # save_adapt_thres_ivim_map(data_dir, ROOT_DIR, bval_threshold_ls, vox)
                
    #     count += 1
        
    #     if count == 4:
    #         break
        
    # if count == 4:
    #     break
    
    


# Save table as .csv file
import pandas as pd
out_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\Pipeline\Quantitative_Analysis"
df = pd.DataFrame(np.concatenate(results_table, axis=0), index = None, columns = columns)
xlsfile = join(out_dir, 'stats_results_final.xlsx')
with pd.ExcelWriter(xlsfile) as writer:
    df.to_excel(writer, index=None, columns=columns)

        
toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
print(f"Downloaded the tutorial in {(toc - tic)/60:0.1f} minutes")  

  
        
  
    
#%%
# # flip_adcmap = [np.flipud(adcmap[sl,:,:]) for sl in range(adcmap.shape[0])]  # flip vertically 
vis = ImageVisualiser(adcmap, cut=(9,25,50), window=(0,3000))
# vis = ImageVisualiser(tumour_mask_dwi, window=(0,1))
# vis.add_scalar_overlay(tumour_mask_dwi, colormap=plt.cm.get_cmap('viridis'))
# vis.add_scalar_overlay(benign_mask_dwi)
vis.add_contour(benign_mask_dwi)
vis.add_contour(prostate_mask_dwi)
vis.show()

# vis_seg = ImageVisualiser(tumour_mask_dwi, cut=(7,25,50), window=(0,1))
# vis_seg.show()

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