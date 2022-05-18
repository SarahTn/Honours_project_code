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

#%% Perform IVIM mapping and save resulting map
def save_overseg_ivim_maps(in_dir, ROOT_DIR, bval_cutoff, vox):
    '''
    bval_cutoff: b-value threshold for fitting
    vox: tuple of voxels inside the prostate
    '''       
    
    D_map, Dstar_map, f_map, rmse_map, aic_map = func_overseg_ivim_mapping(in_dir, bval_cutoff, vox)
    out_dir = join(ROOT_DIR, 'Pipeline', 'Derived_SI-BiRT(2)_data')
    
    dw_image1, bvals1, dw_image2, bvals2, adc_image1, adcimage_2, t2w3d_image = read_in_data(in_dir)
    
    # flip maps along vertical axis so maps are the correct way round
    # flip_D_map = [np.flipud(D_map[sl,:,:]) for sl in range(D_map.shape[0])]
    # D_map_image = sitk.GetImageFromArray(flip_D_map)
    D_map_image = sitk.GetImageFromArray(D_map)
    D_map_image.CopyInformation(adc_image1)
    
    # flip_Dstar_map = [np.flipud(Dstar_map[sl,:,:]) for sl in range(Dstar_map.shape[0])]
    # Dstar_map_image = sitk.GetImageFromArray(flip_Dstar_map)
    Dstar_map_image = sitk.GetImageFromArray(Dstar_map)
    Dstar_map_image.CopyInformation(adc_image1)
    
    # flip_f_map = [np.flipud(f_map[sl,:,:]) for sl in range(f_map.shape[0])]
    # f_map_image = sitk.GetImageFromArray(flip_f_map)
    f_map_image = sitk.GetImageFromArray(f_map)
    f_map_image.CopyInformation(adc_image1)
    
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
    sitk.WriteImage(D_map_image, join(out_path,'D_map_overseg_ivim_bval_cutoff_{}.nii.gz'.format(bval_cutoff)))
    sitk.WriteImage(Dstar_map_image, join(out_path, 'Dstar_map_overseg_ivim_bval_cutoff_{}.nii.gz'.format(bval_cutoff)))
    sitk.WriteImage(f_map_image, join(out_path,'f_map_overseg_ivim_bval_cutoff_{}.nii.gz'.format(bval_cutoff)))
    sitk.WriteImage(rmse_map_image, join(out_path,'rmse_map_overseg_ivim_bval_cutoff_{}.nii.gz'.format(bval_cutoff)))
    sitk.WriteImage(aic_map_image, join(out_path,'aic_map_overseg_ivim_bval_cutoff_{}.nii.gz'.format(bval_cutoff)))
    
    # Display out paths
    print(out_path)



#%%
def func_overseg_ivim_mapping(in_dir, bval_cutoff, vox):
      
    dw_image1, bvals1, dw_image2, bvals2, adc_image1, adcimage_2, t2w3d_image = read_in_data(in_dir)
    array1 = sitk.GetArrayFromImage(dw_image1)

    # ----- Prepare data for fit ------
    # Denoise signals
    dwi_fordenoising = np.moveaxis(array1,(0,1,2,3),(3,2,0,1))  # need to move axis for nlmeans filter
    sigma = estimate_sigma(dwi_fordenoising, N=16)
    denoised_S = nlmeans(dwi_fordenoising, sigma=sigma, mask=None, patch_radius=1, block_radius=2, rician=True)
    dwi_afterdenoising = np.moveaxis(denoised_S, (0,1,2,3),(2,3,1,0))   # move axis back
    
    # Fit maps
    D_map, Dstar_map, f_map, rmse_map, aic_map = fit_overseg_ivim(dwi_afterdenoising, bvals1, bval_cutoff, vox)
    
    return D_map, Dstar_map, f_map, rmse_map, aic_map

#%%
def fit_overseg_ivim(S, bvals, bval_cutoff, vox):
    ''' 
    Fit multiple slices in parallel using joblib's 'Parallel' and delayed'.
    '''
    
    # Setup model  
    D_model = Model(models.linear2, nan_policy='propagate')     # monoexponential model
    ivim_model = Model(models.ivim, nan_policy='propagate')     # IVIM model
    init_D = 1/np.mean(bvals)    # mm^2/s --> D is proportional to 1/b
    
    dim = S.shape
    slices = np.unique(vox[0])
    vox_arr = np.array(vox)
    out = [Parallel(n_jobs=-1, verbose=20)(delayed(fit_slice_overseg_ivim)(S[:,sl,:,:], bvals, bval_cutoff, np.squeeze(vox_arr[1:, list(np.where(vox[0] == sl))]), D_model, ivim_model, init_D) for sl in slices)]
    
    out = np.squeeze(np.array(out))     # out.shape is (slices, parameter map, row, columns), e.g. (12,3,56,100)
        
    # Ensure final map is still 18 slices even though prostate mask only exits for ~12 slices
    D_map = np.zeros(shape=dim[1:])  # don't want b values
    Dstar_map = np.zeros(shape=dim[1:])
    f_map = np.zeros(shape=dim[1:])
    rmse_map = np.zeros(shape=dim[1:])
    aic_map = np.zeros(shape=dim[1:])

    for s in range(len(slices)):
        D_map[slices[s],:,:] = out[s,0,:,:]
        Dstar_map[slices[s],:,:] = out[s,1,:,:]
        f_map[slices[s],:,:] = out[s,2,:,:]
        rmse_map[slices[s],:,:] = out[s,3,:,:]
        aic_map[slices[s],:,:] = out[s,4,:,:]
        
    return D_map, Dstar_map, f_map, rmse_map, aic_map

#%%
def fit_slice_overseg_ivim(S, bvals, bval_cutoff, vox, D_model, ivim_model, init_D):
    
    S = np.squeeze(S)
    dim = S.shape
    # Create empty arrays to store x,y but in one dimension
    Dslice = np.zeros(dim[1]*dim[2])    
    Dstarslice = np.zeros(dim[1]*dim[2])
    fslice = np.zeros(dim[1]*dim[2])
    rmse_slice = np.zeros(dim[1]*dim[2])
    aic_slice = np.zeros(dim[1]*dim[2])
    
    # Create empty array to fill with signals corresponding to each location in a given slice. Shape:  (bvals, x*y)
    S_z = np.zeros(shape=(dim[0],dim[1]*dim[2]))
    
    # For each b-value, store the signal at each location in slice
    for k in range(dim[0]):
        S_z[k,:] = np.reshape(S[k,:,:],(1,dim[1]*dim[2]))
    
    # For each pixel in prostate, compute parameters
    vox_idx = np.ravel_multi_index((vox[0,:],vox[1,:]), (dim[1],dim[2]))
    for v in vox_idx:
          Dslice[v], Dstarslice[v], fslice[v], rmse_slice[v], aic_slice[v] = fit_voxels_overseg_ivim(S_z[:,v], bvals, D_model, ivim_model, bval_cutoff, init_D)
    
    return np.reshape(Dslice,(dim[1],dim[2])), np.reshape(Dstarslice,(dim[1],dim[2])), np.reshape(fslice,(dim[1],dim[2])), np.reshape(rmse_slice,(dim[1],dim[2])), np.reshape(aic_slice,(dim[1],dim[2]))

#%%
def fit_voxels_overseg_ivim(signals, bvals, D_model, ivim_model, bval_cutoff, init_D):
    '''
    "Oversegmented" (Merisaari 2017): 
        Estimate D using linerised monoexp model. Estimate f using (f = 1 - intercept/S0). 
        Then fix D and f and fit for S0 and Dstar using full IVIM model.
    '''
     
    ind = np.where(bvals >= bval_cutoff)
    select_S = np.array(signals)[ind]
    ln_select_S = np.log(select_S)
    
    # STEP 1
    params1 = D_model.make_params()
    params1.add('m', value = init_D)
    params1.add('b', value = np.log(select_S[0]))
    
    params2 = ivim_model.make_params()
    params2.add('Dstar', value = 100*init_D)
    
    # First fit for D and intercept
    try:
        result1 = D_model.fit(ln_select_S, x=bvals[ind], params=params1)
        
    except ValueError:
        return(0,0,0,0,0)  
    
    D = result1.best_values['m']
    intercept = result1.best_values['b']
    # CI_1 = result1.conf_interval(sigmas=[2])
    
    # Estimate f
    f = 1 - np.exp(intercept)/signals[0]
    
    # STEP 2: fit for Dstar and S0, fixing D and f
    params2.add('D', value = D, vary=False)
    params2.add('f', value = f, vary=False)
    params2.add('S0', value = signals[0])  
    
    try: 
        result2 = ivim_model.fit(signals, x=bvals, params=params2, method='leastsq')
        
    except ValueError:
        return(0,0,0,0,0)        

    Dstar = result2.best_values['Dstar']
    
    # S0 = result2.best_values['S0']
    aic = result2.aic
    residuals = result2.residual
    rmse = (np.sum(residuals**2)/len(residuals))**(1/2) 
    # chi_sqr = result2.aic
       
    # Filter out bad values
    if rmse > 50:
        return(0,0,0,0,0) 
    
    D_out = D*1e6
    if D_out > 2500 or D_out < 500:
        D_out = 0
    
    Dstar_out = Dstar
    if Dstar > 0 and Dstar < 0.05:
        Dstar_out = Dstar*1000000 
    if Dstar_out > 25000:
        Dstar_out = 0
        
    f_out = f
    if f > 0 and f < 1:
        f_out = f*100
    if f_out < 0 or f_out > 30:
        f_out = 0
        
    return D_out, Dstar_out, f_out, rmse, aic 
    
    
#%% Test func_overseg_ivim_mapping 

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
  
# # # IVIM mapping test
# bval_cutoff = 200
# D_map, Dstar_map, f_map, rmse_map, aic_map  = func_overseg_ivim_mapping(in_dir, bval_cutoff, vox_tuple)


# cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(aic_map[8], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('AIC map Overseg IVIM')

# cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(rmse_map[8], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('RMSE map Overseg IVIM')

# cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(Dstar_map[8], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('Dstar map Overseg IVIM')



# # Plot D map
# plt.figure()
# flip_D_map = [np.flipud(D_map[sl,:,:]) for sl in range(D_map.shape[0])]
# cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(flip_D_map[7], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('D map overseg b-cutoff {}'.format(bval_cutoff))

# # Plot Dstar map
# plt.figure()
# flip_Dstar_map = [np.flipud(Dstar_map[sl,:,:]) for sl in range(Dstar_map.shape[0])]
# cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(flip_Dstar_map[7], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('Dstar map overseg b-cutoff {}'.format(bval_cutoff))

# # Plot f map
# plt.figure()
# flip_f_map = [np.flipud(f_map[sl,:,:]) for sl in range(f_map.shape[0])]
# cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(flip_f_map[7], cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('f map overseg b-cutoff {}'.format(bval_cutoff))

#%% Test writting map to file
 
# out_dir = join(ROOT_DIR, 'Pipeline', 'Derived_SI-BiRT(2)_data')
# # rotate = [ndimage.rotate(adcmap[sl,:], 180) for sl in range(adcmap.shape[0])]
# # rot_adcmap = np.squeeze(np.array(rotate))
# flip_ivim_map = [np.flipud(ivim_map[sl,:,:]) for sl in range(ivim_map.shape[0])]
# ivim_image = sitk.GetImageFromArray(flip_ivim_map)

# # Save map
# patient = basename(dirname(in_dir))
# studydate = basename(in_dir)

# out_path = join(out_dir, patient, studydate,'Dstar_map_overseg_b_cutoff_{}.nii.gz'.format(bval_cutoff))
# os.makedirs(dirname(out_path), exist_ok=True)

# sitk.WriteImage(ivim_image, out_path)
# print(out_path)


#%% Program for a single image slice

# dw_image1, bvals1, dw_image2, bvals2, ivim_image1, adcimage_2, t2w3d_image = read_in_data(in_dir)
# array1 = sitk.GetArrayFromImage(dw_image1)

# # ----- Prepare data for fit ------
# # Denoise signals
# dwi_fordenoising = np.moveaxis(array1,(0,1,2,3),(3,2,0,1))  # need to move axis for nlmeans filter

# sigma = estimate_sigma(dwi_fordenoising, N=16)
# denoised_S = nlmeans(dwi_fordenoising, sigma=sigma, mask=None, patch_radius=1, block_radius=2, rician=True)
# dwi_afterdenoising = np.moveaxis(denoised_S, (0,1,2,3),(2,3,1,0)) 


# # Setup model  
# D_model = Model(models.linear2, nan_policy='propagate')
# ivim_model = Model(models.ivim, nan_policy='propagate')
# init_D = 1/np.mean(bvals1)    # mm^2/s --> D is proportional to 1/b 

# sl = 8
# sl_signals = dwi_afterdenoising[:,sl,:,:]

# S = np.squeeze(sl_signals)
# dim = S.shape
# ivim_slice = np.zeros(dim[1]*dim[2])
# S_z = np.zeros(shape=(dim[0],dim[1]*dim[2]))    # (bvals, x*y)

# dim = S.shape
# slices = np.unique(vox_tuple[0])
# vox_arr = np.array(vox_tuple)
# vox = np.squeeze(vox_arr[1:, list(np.where(vox_tuple[0] == sl))])

# # For each b bvalue, store the signal at each location in slice
# for k in range(dim[0]):
#     S_z[k,:] = np.reshape(S[k,:,:],(1,dim[1]*dim[2]))

# # For each pixel (element in slice), compute parameter
# vox_idx = np.ravel_multi_index((vox[0,:],vox[1,:]), (dim[1],dim[2]))
# for v in vox_idx:
#       ivim_slice[v] = fit_voxels_overseg_ivim(S_z[:,v], bvals1, D_model, ivim_model, bval_cutoff, init_D, thres_alg=False)

# # Plot map
# ivim_map = np.reshape(ivim_slice,(dim[1],dim[2]))
# # flip_ivim_map = [np.flipud(ivim_map[sl,:,:]) for sl in range(ivim_map.shape[0])]
# cmap = plt.get_cmap('magma')
# plt.rcParams.update({'font.size': 13})
# plt.imshow(ivim_map, cmap=cmap)
# plt.colorbar(label='$10^{-6}$ $mm^2/s$')
# plt.title('Dstar map')



