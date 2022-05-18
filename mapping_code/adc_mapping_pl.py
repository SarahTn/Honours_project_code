# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:15:37 2022

@author: ST
"""
# Import packages
import SimpleITK as sitk
import sys
import os
from os.path import join, dirname, basename
import numpy as np
from lmfit import Model
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import ndimage
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from platipy.imaging import ImageVisualiser
from pathlib import Path


# Import from files
sys.path.append('../../')
from config.definitions import ROOT_DIR
from Pipeline.read_in_data_pl import read_in_data
import Pipeline.model_dictionary_pl as models 
from Pipeline.apply_prostate_mask import prostate_mask_to_dwi_space



#%% Perform ADC mapping and save resulting map
def save_adc_map(in_dir, ROOT_DIR, bvals_for_fit, vox):
# def save_adc_map(dw_image1, adc_image1, bvals1, out_path, bvals_for_fit, vox):
    '''
    bval_threshold_ls: list of b-value thresholds
    vox: tuple of voxels inside the prostate
    '''    
    
    # Call next function
    adcmap, rmse_map, aic_map = func_adcmapping(in_dir, bvals_for_fit, vox)
    # adcmap = func_adcmapping(dw_image1, bvals1, bvals_for_fit, vox)
    out_dir = join(ROOT_DIR, 'Pipeline', 'Derived_SI-BiRT(2)_data')
    
    # flip maps along vertical axis so maps are the correct way round
    # flip_adcmap = [np.flipud(adcmap[sl,:,:]) for sl in range(adcmap.shape[0])]  # flip vertically 
    # adcimage = sitk.GetImageFromArray(flip_adcmap)
    adcimage = sitk.GetImageFromArray(adcmap)
    dw_image1, bvals1, dw_image2, bvals2, adc_image1, adcimage_2, t2w3d_image = read_in_data(in_dir)
    adcimage.CopyInformation(adc_image1)
    
    rmse_map_image = sitk.GetImageFromArray(rmse_map)
    rmse_map_image.CopyInformation(adc_image1)
    
    aic_map_image = sitk.GetImageFromArray(aic_map)
    aic_map_image.CopyInformation(adc_image1)
    
    # # Get patient and studydate for out_path
    patient = basename(dirname(in_dir))
    studydate = basename(in_dir)
    
    out_path = join(out_dir, patient, studydate)
    Path(out_path).mkdir(parents=True, exist_ok=True)  # Don't think this is working properly. It created a folder for RT2 in preRT2 for patient after running Pipeline file over all data.
    
    # Save map
    sitk.WriteImage(adcimage,join(out_path,'adcmap_bvals_50_400_800.nii.gz'))
    sitk.WriteImage(rmse_map_image,join(out_path,'rmse_map_adc_bvals_50_400_800.nii.gz'))
    sitk.WriteImage(aic_map_image,join(out_path,'aic_map_adc_bvals_50_400_800.nii.gz'))

    
    # Display out path
    # print(out_path)

#%%
def func_adcmapping(in_dir, bvals_for_fit, vox):
# def func_adcmapping(dw_image1, bvals1, bvals_for_fit, vox):    
    # Read in data
    dw_image1, bvals1, dw_image2, bvals2, adc_image1, adcimage_2, t2w3d_image = read_in_data(in_dir)
    array1 = sitk.GetArrayFromImage(dw_image1)

    # ----- Prepare data for fit ------
    # Select out data corresponding to the b-values in bvals_for_fit
    indices = [list(bvals1).index(bval) for bval in bvals_for_fit]  
    select_S = np.array([array1[index, :] for index in indices]) 
    
    # Denoise signals
    dwi_fordenoising = np.moveaxis(select_S,(0,1,2,3),(3,2,0,1))  # need to move axis for nlmeans filter
    sigma = estimate_sigma(dwi_fordenoising, N=16)
    denoised_S = nlmeans(dwi_fordenoising, sigma=sigma, mask=None, patch_radius=1, block_radius=2, rician=True)
    dwi_afterdenoising = np.moveaxis(denoised_S, (0,1,2,3),(2,3,1,0))   # move axis back
    
    # Fit map
    adcmap, rmse_map, aic_map = fit_ADC_in_parallel(dwi_afterdenoising, bvals_for_fit, vox)
    
    return(adcmap, rmse_map, aic_map)

#%%
def fit_ADC_in_parallel(S, bvals_for_fit, vox):
    ''' 
    Fit multiple slices in parallel using joblib's 'Parallel' and delayed'.
    ''' 
    
    # Setup model using lmfit  
    fit_model = Model(models.linear2, nan_policy='propagate')
    start_ADC = 1/np.mean(bvals_for_fit)    # mm^2/s --> ADC is proportional to 1/b
    
    dim = S.shape
    slices = np.unique(vox[0])
    vox_arr = np.array(vox)

    out = [Parallel(n_jobs=-1, verbose=20)(delayed(fit_slice_ADC)(S[:,sl,:,:], bvals_for_fit, np.squeeze(vox_arr[1:, list(np.where(vox[0] == sl))]), fit_model, start_ADC) for sl in slices)]
    
    out = np.squeeze(np.array(out))

    # Ensure final map is still 18 slices even though prostate mask only exits for ~12 slices
    adcmap = np.zeros(shape=dim[1:])  # don't want b values
    rmse_map = np.zeros(shape=dim[1:])
    aic_map = np.zeros(shape=dim[1:])
    
    for s in range(len(slices)):
        adcmap[slices[s],:,:] = out[s,0,:,:]
        rmse_map[slices[s],:,:] = out[s,1,:,:]
        aic_map[slices[s],:,:] = out[s,2,:,:]

    return(adcmap, rmse_map, aic_map)

#%%
def fit_slice_ADC(S, bvals_for_fit, vox, fit_model, start_ADC):
    '''
    For a given slice, fit the model to each pixel within prostate mask.
    '''
    
    S = np.squeeze(S)
    dim = S.shape
    # Create empty arrays to store x,y but in one dimension
    adcslice = np.zeros(dim[1]*dim[2])
    rmse_slice = np.zeros(dim[1]*dim[2])
    aic_slice = np.zeros(dim[1]*dim[2])
    
    # Create empty array to fill with signals corresponding to each location in a given slice. Shape:  (bvals, x*y)
    S_z = np.zeros(shape=(dim[0],dim[1]*dim[2]))    
    
    # For each b bvalue, store the signal at each location in adcslice
    for k in range(dim[0]):
        S_z[k,:] = np.reshape(S[k,:,:],(1,dim[1]*dim[2]))
    
    # For each pixel in prostate, compute ADC
    vox_idx = np.ravel_multi_index((vox[0,:],vox[1,:]), (dim[1],dim[2]))
    for v in vox_idx:
          adcslice[v], rmse_slice[v], aic_slice[v] = fit_voxels_ADC(S_z[:,v], bvals_for_fit, fit_model, start_ADC)
    
    return np.reshape(adcslice,(dim[1],dim[2])), np.reshape(rmse_slice,(dim[1],dim[2])), np.reshape(aic_slice,(dim[1],dim[2]))

#%%
def fit_voxels_ADC(signals, bvals, fit_linear_model, init_ADC):
    '''
    Fit linearised monoexponential model return an estimate for ADC:
        ln(S(B)) = - ADC*b + ln(S)   --->   y = -m*x + b    where m = ADC, b = ln(S0)
    
    Parameters
    ----------
    signals: 
        signals corresponding to selected b values chosen for fit. 
    bvals:
        selected bvals for fit.
    fit_linear_model:
        create an instance of linear model 
    Returns
    -------
    ADC:
        estimate for ADC
    '''
    init_S0 = signals[0]
    ln_signals = np.log(signals)
      
    # Add parameters to model
    params = fit_linear_model.make_params()
    params.add('m', value = init_ADC)
    params.add('b', value = np.log(init_S0))
    
    try:     
        # result = fit_linear_model.fit(ln_signals[indx], x=bvals[indx], params=params)
        result = fit_linear_model.fit(ln_signals, x=bvals, params=params)    

    except ValueError:
        return(0,0,0)
    
    ADC = result.best_values['m']
    # ln_S0 = result.best_values['b']
    # CI = result.conf_interval(sigmas=[2])
    
    aic = result.aic
    residuals = result.residual
    rmse = (np.sum(residuals**2)/len(residuals))**(1/2) 
    
    # S0_out = np.exp(ln_S0)
    ADC_out = ADC*1e6   # 10^-6 * mm^2/s 

    # filter out bad values
    if ADC_out > 2500 or ADC_out < 300:
        return(0,0,0)    
    
    # return ADC_out, S0_out, result, CI
    return ADC_out, rmse, aic

    
#%% Test func_adcmapping

# # Read in data
# patient = '3497-1'
# treatment_date = '20200721_preRT_1'
# wk0_date = '20200721'
# in_dir = join(ROOT_DIR, 'SiBiRT2_Data', patient, treatment_date)
# dw_image1, bvals1, dw_image2, bvals2, adc_image, adcimage_2, t2w3d_image = read_in_data(in_dir)

# # Get prostate mask in dwi space

# prostate_mask_dwi, vox = prostate_mask_to_dwi_space(ROOT_DIR, patient, treatment_date, wk0_date, t2w3d_image, adc_image)

# # # image_visualiser = ImageVisualiser(adc_image, window=(200,2000))
# # # image_visualiser.add_contour(prostate_mask_dwi)
# # # fig1 = image_visualiser
  
# # ADC mapping test
# bvals_for_fit = [50, 400, 800]
# # save_adc_map(in_dir, ROOT_DIR, bvals_for_fit, vox)

# adcmap, rmse_map, aic_map = func_adcmapping(in_dir, bvals_for_fit, vox)

# cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(aic_map[8], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('AIC map ADC')

# # cmap = plt.get_cmap('viridis')
# # plt.rcParams.update({'font.size': 13})
# # plt.imshow(adcmap[8], cmap=cmap)
# # plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# # plt.title('ADC map')

# cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(rmse_map[8], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('RMSE map ADC')

# # # flip_adcmap = np.flipud(adcmap[7,:,:])  # Only needed for display -> sitk handles this when writing image to file 

# # Plot map
# flip_adcmap = [np.flipud(adcmap[sl,:,:]) for sl in range(adcmap.shape[0])]
# cmap = plt.get_cmap('magma')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(flip_adcmap[7], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('ADC map')

#%% Test writting map to file

# out_dir = join(ROOT_DIR, 'Pipeline', 'Derived_SI-BiRT(2)_data')
# # rotate = [ndimage.rotate(adcmap[sl,:], 180) for sl in range(adcmap.shape[0])]
# # rot_adcmap = np.squeeze(np.array(rotate))
# flip_adcmap = [np.flipud(adcmap[sl,:,:]) for sl in range(adcmap.shape[0])]
# adcimage = sitk.GetImageFromArray(flip_adcmap)

# # Save map
# patient = basename(dirname(in_dir))
# studydate = basename(in_dir)

# out_path = join(out_dir, patient, studydate,'adcmap_50_400_800.nii.gz')
# os.makedirs(dirname(out_path), exist_ok=True)

# sitk.WriteImage(adcimage, out_path)
# print(out_path)

#%% Program for a single image slice

# dw_image1, bvals1, dw_image2, bvals2, adc_image1, adcimage_2, t2w3d_image = read_in_data(in_dir)
# array1 = sitk.GetArrayFromImage(dw_image1)

# # prepare data for fit
# # Select out data corresponding to the b-values in bvals_for_fit
# indices = [list(bvals1).index(bval) for bval in bvals_for_fit]  
# select_S = np.array([array1[index, :] for index in indices]) 

# # Denoise signals
# dwi_fordenoising = np.moveaxis(select_S,(0,1,2,3),(3,2,0,1))  # more axis for nlmeans filter
# sigma = estimate_sigma(dwi_fordenoising, N=16)
# denoised_S = nlmeans(dwi_fordenoising, sigma=sigma, mask=None, patch_radius=1, block_radius=2, rician=True)
# dwi_afterdenoising = np.moveaxis(denoised_S, (0,1,2,3),(2,3,1,0))


# fit_model = Model(models.linear2, nan_policy='propagate')
# start_ADC = 1/np.mean(bvals1)

# S = dwi_afterdenoising[:,9,:,:]
# S = np.squeeze(S)
# dim = S.shape
# adcslice = np.zeros(dim[1]*dim[2])
# S_z = np.zeros(shape=(dim[0],dim[1]*dim[2]))    # (bvals, x*y)

# # For each b bvalue, store the signal at each location in adcslice
# for k in range(dim[0]):
#     S_z[k,:] = np.reshape(S[k,:,:],(1,dim[1]*dim[2]))

# # For each pixel (element in adcslice), compute ADC
# vox_idx = np.ravel_multi_index((vox[1],vox[2]), (dim[1],dim[2]))
# for v in vox_idx:
#       adcslice[v] = fit_voxels_ADC(S_z[:,v], bvals_for_fit, fit_model, start_ADC)
      
     
# adcmap = np.reshape(adcslice,(dim[1],dim[2]))

# rot_adcmap = ndimage.rotate(adcmap[7,:,:], 180) 

# rotate = [ndimage.rotate(adcmap[:,sl,:], 180) for sl in range(adcmap.shape[0])]
# rot_adcmap = np.squeeze(np.array(rotate))

# # Plot map
# cmap = plt.get_cmap('magma')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(rot_adcmap[:,9,:], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('ADC map')
