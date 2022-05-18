# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:18:01 2022

@author: Sarah
"""

import sys
import os
sys.path.append('../../')
import SimpleITK as sitk
import numpy as np

def read_in_data(in_dir):
    '''
    Parameters
    ----------
    in_dir : filepath
        Folder filepath (in_dir) which contains imaging data. 
        
    Returns
    -------
    dw_image_1 : sitk image 1
        Trace DW image dataset 1 (bvals 0-800 s/mm^2).
    bvals_1 : numpy array
        B values corresponding to dataset 1.
    dw_image_2 : sitk image 2
        Trace DW image dataset 2 (bvals 0, 1600-2400 s/mm^2).
    bvals_2 : numpy array
        B values corresponding to dataset 2.
    '''
    all_dirs = os.listdir(in_dir)
        
    dwi_1_dirs = [i for i in all_dirs if "800_tra" in i and 'nii.gz' in i]
    bvals_1_dirs = [i for i in all_dirs if "800_tra" in i and "bval" in i]
    
    dwi_2_dirs = [i for i in all_dirs if "2400_tra_tra" in i and 'nii.gz' in i]    
    bvals_2_dirs = [i for i in all_dirs if "2400_tra_tra" in i and "bval" in i]
    
    adc_1_dirs = [i for i in all_dirs if "800" in i and "adc" in i and 'nii.gz' in i] 
    adc_2_dirs = [i for i in all_dirs if "2400" in i and "adc" in i and 'nii.gz' in i]  
    
    t2w3d_dirs = [i for i in all_dirs if "t2" in i and "sag" in i and 'nii.gz' in i]  
    
    bvals1 = np.loadtxt(os.path.join(in_dir, bvals_1_dirs[0]))
    bvals2 = np.loadtxt(os.path.join(in_dir,  bvals_2_dirs[0]))
    
    dw_image1 = sitk.ReadImage(os.path.join(in_dir, dwi_1_dirs[0]))   
    dw_image2 = sitk.ReadImage(os.path.join(in_dir, dwi_2_dirs[0]))
    
    adc_image1 = sitk.ReadImage(os.path.join(in_dir, adc_1_dirs[0]))   
    adc_image2 = sitk.ReadImage(os.path.join(in_dir, adc_2_dirs[0])) 

    t2w3d_image = sitk.ReadImage(os.path.join(in_dir, t2w3d_dirs[0])) 

    return dw_image1, bvals1, dw_image2, bvals2, adc_image1, adc_image2, t2w3d_image 



