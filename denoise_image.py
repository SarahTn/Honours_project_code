# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:48:39 2022

@author: slizz
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
from config.definitions import ROOT_DIR
from Pipeline.read_in_data_pl import read_in_data
import Pipeline.model_dictionary_pl as models 
from Pipeline.apply_prostate_mask import prostate_mask_to_dwi_space

#%%  
# Read in data
patient = '3497-1'
treatment_date = '20200721_preRT_1'
wk0_date = '20200721'
in_dir = join(ROOT_DIR, 'SiBiRT2_Data', patient, treatment_date)
dw_image1, bvals1, dw_image2, bvals2, adc_image, adcimage_2, t2w3d_image = read_in_data(in_dir)

array1 = sitk.GetArrayFromImage(dw_image1)

# ----- Prepare data for fit ------
# # Select out data corresponding to the b-values in bvals_for_fit
# indices = [list(bvals1).index(bval) for bval in bvals_for_fit]  
select_S = array1[5, :]  
# select_S = array1 


# Denoise signals
dwi_fordenoising = np.moveaxis(select_S,(0,1,2),(2,1,0))  # need to move axis for nlmeans filter
sigma = estimate_sigma(dwi_fordenoising, N=16)
denoised_S = nlmeans(dwi_fordenoising, sigma=sigma, mask=None, patch_radius=1, block_radius=2, rician=True)
dwi_afterdenoising = np.moveaxis(denoised_S, (0,1,2),(2,1,0)) 

#%%
cmap = plt.get_cmap('gray')
plt.rcParams.update({'font.size': 13})
plt.imshow(dwi_afterdenoising[7], cmap=cmap)
plt.colorbar(label='$10^{-6}$ $mm^2/s$')
plt.title('DWI')


#%%
dwi_im = sitk.GetImageFromArray(dwi_afterdenoising)
vis = ImageVisualiser(dwi_im, window=(0,1000))
vis.show()