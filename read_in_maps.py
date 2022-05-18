# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:14:04 2022

@author: ST
"""
from os.path import join
import SimpleITK as sitk

def read_in_adcmap(map_dir):
    '''
    map_dir: directory where maps for patient have been saved.
    '''
    
    # adcmap_dir = join(map_dir, 'adcmap_50_400_800.nii.gz')
    adcmap_dir = join(map_dir, 'adcmap_bvals_50_400_800.nii.gz')
    adcmap_image = sitk.ReadImage(adcmap_dir)
    
    rmse_map_dir = join(map_dir, 'rmse_map_adc_bvals_50_400_800.nii.gz')
    rmse_map_image = sitk.ReadImage(rmse_map_dir)
    
    aic_map_dir = join(map_dir, 'aic_map_adc_bvals_50_400_800.nii.gz')
    aic_map_image = sitk.ReadImage(aic_map_dir)
    
    return adcmap_image, rmse_map_image, aic_map_image
    

def read_in_seg_ivim_maps(map_dir): 
         
    D_seg_dir = join(map_dir, 'D_map_seg_ivim_bval_cutoff_200.nii.gz')
    D_seg_map = sitk.ReadImage(D_seg_dir)
    
    Dstar_seg_dir = join(map_dir, 'Dstar_map_seg_ivim_bval_cutoff_200.nii.gz')
    Dstar_seg_map = sitk.ReadImage(Dstar_seg_dir)
    
    f_seg_dir = join(map_dir, 'f_map_seg_ivim_bval_cutoff_200.nii.gz')
    f_seg_map = sitk.ReadImage(f_seg_dir)
    
    rmse_seg_dir = join(map_dir, 'rmse_map_seg_ivim_bval_cutoff_200.nii.gz')
    rmse_seg_map = sitk.ReadImage(rmse_seg_dir)
    
    aic_seg_dir = join(map_dir, 'aic_map_seg_ivim_bval_cutoff_200.nii.gz')
    aic_seg_map = sitk.ReadImage(aic_seg_dir)   
    
    return D_seg_map, Dstar_seg_map, f_seg_map, rmse_seg_map, aic_seg_map


def read_in_seg_bval_thres_maps(map_dir):
    D_seg_b_thres_dir = join(map_dir, 'D_map_seg_ivim_bval_thres_alg.nii.gz')
    D_seg_b_thres_map = sitk.ReadImage(D_seg_b_thres_dir)
    
    Dstar_seg_b_thres_dir = join(map_dir, 'Dstar_map_seg_ivim_bval_thres_alg.nii.gz')
    Dstar_seg_b_thres_map = sitk.ReadImage(Dstar_seg_b_thres_dir)
    
    f_seg_b_thres_dir = join(map_dir, 'f_map_seg_ivim_bval_thres_alg.nii.gz')
    f_seg_b_thres_map = sitk.ReadImage(f_seg_b_thres_dir)
    
    bval_thres_map_seg_dir = join(map_dir, 'bval_thres_map_seg_ivim.nii.gz')
    bval_thres_map_seg = sitk.ReadImage(bval_thres_map_seg_dir)
    
    rmse_seg_b_thres_dir = join(map_dir, 'rmse_map_seg_ivim_bval_thres_alg.nii.gz')
    rmse_seg_b_thres_map = sitk.ReadImage(rmse_seg_b_thres_dir)
    
    aic_seg_b_thres_dir = join(map_dir, 'aic_map_seg_ivim_bval_thres_alg.nii.gz')
    aic_seg_b_thres_map = sitk.ReadImage(aic_seg_b_thres_dir)   
    
    
    return D_seg_b_thres_map, Dstar_seg_b_thres_map, f_seg_b_thres_map, bval_thres_map_seg, rmse_seg_b_thres_map, aic_seg_b_thres_map

def read_in_overseg_maps(map_dir): 
         
    D_overseg_dir = join(map_dir, 'D_map_overseg_ivim_bval_cutoff_200.nii.gz')
    D_overseg_map = sitk.ReadImage(D_overseg_dir)
    
    Dstar_overseg_dir = join(map_dir, 'Dstar_map_overseg_ivim_bval_cutoff_200.nii.gz')
    Dstar_overseg_map = sitk.ReadImage(Dstar_overseg_dir)
    
    f_overseg_dir = join(map_dir, 'f_map_overseg_ivim_bval_cutoff_200.nii.gz')
    f_overseg_map = sitk.ReadImage(f_overseg_dir)
    
    rmse_overseg_dir = join(map_dir, 'rmse_map_overseg_ivim_bval_cutoff_200.nii.gz')
    rmse_overseg_map = sitk.ReadImage(rmse_overseg_dir)
    
    aic_overseg_dir = join(map_dir, 'aic_map_overseg_ivim_bval_cutoff_200.nii.gz')
    aic_overseg_map = sitk.ReadImage(aic_overseg_dir)   
    
    return D_overseg_map, Dstar_overseg_map, f_overseg_map, rmse_overseg_map, aic_overseg_map



def read_in_overseg_bval_thres_maps(map_dir):
    D_overseg_b_thres_dir = join(map_dir, 'D_map_overseg_ivim_bval_thres_alg.nii.gz')
    D_overseg_b_thres_map = sitk.ReadImage(D_overseg_b_thres_dir)
    
    Dstar_overseg_b_thres_dir = join(map_dir, 'Dstar_map_overseg_ivim_bval_thres_alg.nii.gz')
    Dstar_overseg_b_thres_map = sitk.ReadImage(Dstar_overseg_b_thres_dir)
    
    f_overseg_b_thres_dir = join(map_dir, 'f_map_overseg_ivim_bval_thres_alg.nii.gz')
    f_overseg_b_thres_map = sitk.ReadImage(f_overseg_b_thres_dir)
    
    bval_thres_map_overseg_dir = join(map_dir, 'bval_thres_map_overseg_ivim.nii.gz')
    bval_thres_map_overseg = sitk.ReadImage(bval_thres_map_overseg_dir)
    
    rmse_overseg_b_thres_dir = join(map_dir, 'rmse_map_overseg_ivim_bval_thres_alg.nii.gz')
    rmse_overseg_b_thres_map = sitk.ReadImage(rmse_overseg_b_thres_dir)
    
    aic_overseg_b_thres_dir = join(map_dir, 'aic_map_overseg_ivim_bval_thres_alg.nii.gz')
    aic_overseg_b_thres_map = sitk.ReadImage(aic_overseg_b_thres_dir)
    
    return D_overseg_b_thres_map, Dstar_overseg_b_thres_map, f_overseg_b_thres_map, bval_thres_map_overseg, rmse_overseg_b_thres_map, aic_overseg_b_thres_map 
    
    
def read_in_kurt_maps(map_dir): 
         
    Dapp_dir = join(map_dir, 'Dapp_map_kurtosis_bval_cutoff_400.nii.gz')
    Dapp_map = sitk.ReadImage(Dapp_dir)
    
    Kapp_dir = join(map_dir, 'Kapp_map_kurtosis_bval_cutoff_400.nii.gz')
    Kapp_map = sitk.ReadImage(Kapp_dir)
    
    rmse_kurt_dir = join(map_dir, 'rmse_map_kurtosis_bval_cutoff_400.nii.gz')
    rmse_kurt_map = sitk.ReadImage(rmse_kurt_dir)
    
    aic_kurt_dir = join(map_dir, 'aic_map_kurtosis_bval_cutoff_400.nii.gz')
    aic_kurt_map = sitk.ReadImage(aic_kurt_dir)
    
    return Dapp_map, Kapp_map, rmse_kurt_map, aic_kurt_map

def read_in_linearied_kurt_maps(map_dir): 
         
    Dapp_dir = join(map_dir, 'Dapp_map_lin_kurt_bval_cutoff_400.nii.gz')
    Dapp_map = sitk.ReadImage(Dapp_dir)
    
    Kapp_dir = join(map_dir, 'Kapp_map_lin_kurt_bval_cutoff_400.nii.gz')
    Kapp_map = sitk.ReadImage(Kapp_dir)
    
    rmse_kurt_dir = join(map_dir, 'rmse_map_lin_kurt_bval_cutoff_400.nii.gz')
    rmse_kurt_map = sitk.ReadImage(rmse_kurt_dir)
    
    aic_kurt_dir = join(map_dir, 'aic_map_lin_kurt_bval_cutoff_400.nii.gz')
    aic_kurt_map = sitk.ReadImage(aic_kurt_dir)
    
    return Dapp_map, Kapp_map, rmse_kurt_map, aic_kurt_map
    
    
    
    
    


    
    
    
    