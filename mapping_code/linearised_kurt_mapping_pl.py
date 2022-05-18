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
sys.path.append('../')
from read_in_data_pl import read_in_data
import model_dictionary_pl as models 
from apply_prostate_mask import prostate_mask_to_dwi_space
from config.definitions import ROOT_DIR




#%% Perform kurt mapping and save resulting map
def save_lin_kurt_map(in_dir, ROOT_DIR, bval_cutoff, vox):
    
    # Call next function
    Dappmap, Kappmap, rmse_map, aic_map = func_lin_kurtmapping(in_dir, bval_cutoff, vox)
    out_dir = join(ROOT_DIR, 'Pipeline', 'Derived_SI-BiRT(2)_data')
    
    dw_image1, bvals1, dw_image2, bvals2, adc_image1, adcimage_2, t2w3d_image = read_in_data(in_dir)

    Dapp_map_image = sitk.GetImageFromArray(Dappmap)
    Dapp_map_image.CopyInformation(adc_image1)
    
    Kappmap_map_image = sitk.GetImageFromArray(Kappmap)
    Kappmap_map_image.CopyInformation(adc_image1)
    
    rmse_map_image = sitk.GetImageFromArray(rmse_map)
    rmse_map_image.CopyInformation(adc_image1)
    
    aic_map_image = sitk.GetImageFromArray(aic_map)
    aic_map_image.CopyInformation(adc_image1)
    
    # Get patient and studydate for out_path
    patient = basename(dirname(in_dir))
    studydate = basename(in_dir)
    
    # Get out paths
    out_path = join(out_dir, patient, studydate)
    Path(out_path).mkdir(parents=True, exist_ok=True)
       
    # Save maps
    sitk.WriteImage(Dapp_map_image, join(out_path,'Dapp_map_lin_kurt_bval_cutoff_{}.nii.gz'.format(bval_cutoff)))
    sitk.WriteImage(Kappmap_map_image, join(out_path, 'Kapp_map_lin_kurt_bval_cutoff_{}.nii.gz'.format(bval_cutoff)))
    sitk.WriteImage(rmse_map_image, join(out_path, 'rmse_map_lin_kurt_bval_cutoff_{}.nii.gz'.format(bval_cutoff)))
    sitk.WriteImage(aic_map_image, join(out_path, 'aic_map_lin_kurt_bval_cutoff_{}.nii.gz'.format(bval_cutoff)))
    print(out_path)

#%%
def func_lin_kurtmapping(in_dir, bval_cutoff, vox):
    
    # Read in data
    dw_image1, bvals1, dw_image2, bvals2, kurt_image1, kurtimage_2, t2w3d_image = read_in_data(in_dir)
    
    array1 = sitk.GetArrayFromImage(dw_image1)
    array2 = sitk.GetArrayFromImage(dw_image2)  
    
    array_tot = np.concatenate((array1, array2))
    bvals_tot = np.concatenate((bvals1, bvals2))

    # ----- Prepare data for fit ------

    
    # Denoise signals
    dwi_fordenoising = np.moveaxis(array_tot,(0,1,2,3),(3,2,0,1))  # need to move axis for nlmeans filter
    sigma = estimate_sigma(dwi_fordenoising, N=16)
    denoised_S = nlmeans(dwi_fordenoising, sigma=sigma, mask=None, patch_radius=1, block_radius=2, rician=True)
    dwi_afterdenoising = np.moveaxis(denoised_S, (0,1,2,3),(2,3,1,0))   # move axis back
    
    Dappmap, Kappmap, rmse_map, aic_map = fit_kurt_in_parallel(dwi_afterdenoising, bvals_tot, bval_cutoff, vox)
    
    return Dappmap, Kappmap, rmse_map, aic_map

#%%
def fit_kurt_in_parallel(S, bvals_tot, bval_cutoff, vox):
    ''' 
    Fit multiple slices in parallel
    ''' 
    
    # Setup model using lmfit  
    fit_model = Model(models.lin_kurtosis, nan_policy='propagate')
    start_Dapp = 1/np.mean(bvals_tot)    # mm^2/s --> Dapp is proportional to 1/b
    
    dim = S.shape
    slices = np.unique(vox[0])
    vox_arr = np.array(vox)
    out = [Parallel(n_jobs=-1, verbose=20)(delayed(fit_slice_kurt)(S[:,sl,:,:], bvals_tot, bval_cutoff,  np.squeeze(vox_arr[1:, list(np.where(vox[0] == sl))]), fit_model, start_Dapp) for sl in slices)]
    
    out = np.squeeze(np.array(out))

    # Ensure final map is still 18 slices even though prostate mask only exits for ~12 slices
    Dappmap = np.zeros(shape=dim[1:])  # don't want b values
    Kappmap = np.zeros(shape=dim[1:])
    rmse_map = np.zeros(shape=dim[1:])
    aic_map = np.zeros(shape=dim[1:])
    
    for s in range(len(slices)):
        Dappmap[slices[s],:,:] = out[s,0,:,:]
        Kappmap[slices[s],:,:] = out[s,1,:,:]
        rmse_map[slices[s],:,:] = out[s,2,:,:]
        aic_map[slices[s],:,:] = out[s,3,:,:]
        
    return Dappmap, Kappmap, rmse_map, aic_map

#%%
def fit_slice_kurt(S, bvals_tot, bval_cutoff, vox, fit_model, start_Dapp):
    '''
    For a given slice, fit the model to each pixel within prostate mask.
    '''
    
    S = np.squeeze(S)
    dim = S.shape
    # Create empty arrays to store x,y but in one dimension
    Dappslice = np.zeros(dim[1]*dim[2])    
    Kappslice = np.zeros(dim[1]*dim[2])
    rmse_slice = np.zeros(dim[1]*dim[2])
    aic_slice = np.zeros(dim[1]*dim[2])
    
    S_z = np.zeros(shape=(dim[0],dim[1]*dim[2]))    # (bvals, x*y)
    
    # For each b bvalue, store the signal at each location in kurtslice
    for k in range(dim[0]):
        S_z[k,:] = np.reshape(S[k,:,:],(1,dim[1]*dim[2]))
    
    # For each pixel (element in kurtslice), compute kurt
    vox_idx = np.ravel_multi_index((vox[0,:],vox[1,:]), (dim[1],dim[2]))
    for v in vox_idx:
          Dappslice[v], Kappslice[v], rmse_slice[v], aic_slice[v] = fit_voxels_linear_kurt(S_z[:,v], bvals_tot, bval_cutoff, fit_model, start_Dapp)
    
    return np.reshape(Dappslice,(dim[1],dim[2])), np.reshape(Kappslice,(dim[1],dim[2])), np.reshape(rmse_slice,(dim[1],dim[2])), np.reshape(aic_slice,(dim[1],dim[2]))

#%%
def fit_voxels_linear_kurt(S, bvals_tot, bval_cutoff, fit_kurtosis_model, init_Dapp):
    '''
    - corrected_sigs: signals with correction factor to account for different TEs for each b-value dataset.
    - Only fit within tumour to avoid fitting signals below the noise floor.
    '''
    
    
    TE1 = 60     # echo time for dataset 1 (ms)
    TE2 = 69    # echo time for dataset 2 (ms)
    signals1 = S[:9]
    signals2 = S[9:]
    bvals1 = bvals_tot[:9]
    bvals2 = bvals_tot[9:]
    T2 = -(TE1 - TE2)/(np.log(signals1[0]/signals2[0]))     # ms 
    corr_factor1 = np.exp(- TE1/T2)     # correction factor for dataset 1 (ms)
    corr_factor2 = np.exp(- TE2/T2)     # correction factor for dataset 2 (ms)

      
    signals1_corr = signals1/corr_factor1
    signals2_corr = signals2/corr_factor2
    ordered_bvals = np.concatenate(([bvals1[0], bvals2[0]], bvals1[1:], bvals2[1:]))
    signals_corr = np.concatenate(([signals1_corr[0], signals2_corr[0]], signals1_corr[1:], signals2_corr[1:]))
    
    # bvals = np.unique(bvals_tot)
    ind1 = np.where(ordered_bvals >= bval_cutoff)
    # ind1 = np.where(bvals_tot >= bval_cutoff)
    ln_signals = np.log(signals_corr)
    
    
    # Add parameters to model
    params = fit_kurtosis_model.make_params()
    params.add('Dapp', value = init_Dapp)
    params.add('Kapp', value = 0)
    params.add('lnS0', value = ln_signals[0])


    try:     
        result = fit_kurtosis_model.fit(ln_signals[ind1], x=ordered_bvals[ind1], params=params)    

    except ValueError:
        return 0,0
    
    # S0 = result.best_values['S0']
    Dapp = result.best_values['Dapp']
    Kapp = result.best_values['Kapp']
    
    aic = result.aic
    residuals = result.residual
    rmse = (np.sum(residuals**2)/len(residuals))**(1/2) 

    if rmse > 50:
        return(0,0,0,0) 

    Dapp_out = Dapp*1e6      # 10^-6 * mm^2/s 
    if Dapp_out > 2500 or Dapp_out < 500:
        Dapp_out = 0   
    
    if Kapp < 0 or Kapp > 3:
        Kapp = 0

    # return Dapp_out, Kapp, S0, result
    return Dapp_out, Kapp, rmse, aic


    
#%% Test func_kurtmapping

# # Read in data
# patient = '3497-1'
# treatment_date = '20200804_preRT_2'
# wk0_date = '20200721'
# in_dir = join(ROOT_DIR, 'SiBiRT2_Data', patient, treatment_date)
# dw_image1, bvals1, dw_image2, bvals2, adc_image, adcimage_2, t2w3d_image = read_in_data(in_dir)

# # Get prostate mask in dwi space
# prostate_mask_dwi, vox_tuple = prostate_mask_to_dwi_space(ROOT_DIR, patient, treatment_date, wk0_date, t2w3d_image, adc_image)

# # # image_visualiser = ImageVisualiser(adc_image, window=(200,2000))
# # # image_visualiser.add_contour(prostate_mask_dwi)
# # # fig1 = image_visualiser
  
# # # Kurtosis mapping test
# bval_cutoff = 400
# Dappmap, Kappmap, rmse_map, aic_map  = func_kurtmapping(in_dir, bval_cutoff, vox_tuple)


# cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(rmse_map[8], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('RMSE map Overseg IVIM')

# cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(Dappmap[8], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('Dapp map Overseg IVIM')

#%% Test writting map to file

# out_dir = join(ROOT_DIR, 'Pipeline', 'Derived_SI-BiRT(2)_data')
# # rotate = [ndimage.rotate(kurtmap[sl,:], 180) for sl in range(kurtmap.shape[0])]
# # rot_kurtmap = np.squeeze(np.array(rotate))
# flip_kurtmap = [np.flipud(kurtmap[sl,:,:]) for sl in range(kurtmap.shape[0])]
# kurtimage = sitk.GetImageFromArray(flip_kurtmap)

# # Save map
# patient = basename(dirname(in_dir))
# studydate = basename(in_dir)

# out_path = join(out_dir, patient, studydate,'kurtmap_50_400_800.nii.gz')
# os.makedirs(dirname(out_path), exist_ok=True)

# sitk.WriteImage(kurtimage, out_path)
# print(out_path)

#%% Program for a single image slice

# dw_image1, bvals1, dw_image2, bvals2, kurt_image1, kurtimage_2, t2w3d_image = read_in_data(in_dir)
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
# start_kurt = 1/np.mean(bvals1)

# S = dwi_afterdenoising[:,9,:,:]
# S = np.squeeze(S)
# dim = S.shape
# kurtslice = np.zeros(dim[1]*dim[2])
# S_z = np.zeros(shape=(dim[0],dim[1]*dim[2]))    # (bvals, x*y)

# # For each b bvalue, store the signal at each location in kurtslice
# for k in range(dim[0]):
#     S_z[k,:] = np.reshape(S[k,:,:],(1,dim[1]*dim[2]))

# # For each pixel (element in kurtslice), compute kurt
# vox_idx = np.ravel_multi_index((vox[1],vox[2]), (dim[1],dim[2]))
# for v in vox_idx:
#       kurtslice[v] = fit_voxels_kurt(S_z[:,v], bvals_for_fit, fit_model, start_kurt)
      
     
# kurtmap = np.reshape(kurtslice,(dim[1],dim[2]))

# rot_kurtmap = ndimage.rotate(kurtmap[7,:,:], 180) 

# rotate = [ndimage.rotate(kurtmap[:,sl,:], 180) for sl in range(kurtmap.shape[0])]
# rot_kurtmap = np.squeeze(np.array(rotate))

# # Plot map
# cmap = plt.get_cmap('magma')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(rot_kurtmap[:,9,:], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('kurt map')
