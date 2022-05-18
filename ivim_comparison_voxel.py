# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:27:09 2022

@author: slizz
"""
# Import packages
import os
import sys
import numpy as np
import SimpleITK as sitk
from lmfit import Model
import matplotlib.pyplot as plt

# Import from files
sys.path.append('../')
from read_in_data_pl import read_in_data
import model_dictionary_pl as models 
from apply_prostate_mask import prostate_mask_to_dwi_space
from config.definitions import ROOT_DIR

#%%
def fit_voxels_seg_bval_thres(signals, bvals, D_model, ivim_model, bval_threshold_ls, init_D):
    '''
    Adaptive b-value threshold algorithm IVIM (Wurnig et al. 2015)
    '''
    chi_sqr_ls = []
    for bval_cutoff in bval_threshold_ls:
        
        try:
            D_out, Dstar_out, f_out, chi_sqr, rmse, aic, result = fit_voxels_seg_ivim(signals, bvals, D_model, ivim_model, bval_cutoff, init_D)
        
        except TypeError:
            return(0,0,0,0,0,0)
    
        chi_sqr_ls.append([chi_sqr])
      
    best_bval_cutoff = bval_threshold_ls[chi_sqr_ls.index(min(chi_sqr_ls))]     # pick b value threshold corresponding to the smallest chisqr

    try:    
        D_out, Dstar_out, f_out, chi_sqr, rmse, aic, result = fit_voxels_seg_ivim(signals, bvals, D_model, ivim_model, best_bval_cutoff, init_D)
    
    except TypeError:
        return(0,0,0,0,0,0)
    
    aic = result.aic
    residuals = result.residual
    rmse = (np.sum(residuals**2)/len(residuals))**(1/2) 
    
    if rmse > 50:
        return(0,0,0,0,0,0) 
    
    return D_out, Dstar_out, f_out, best_bval_cutoff, rmse, aic, result


#%%
def fit_voxels_seg_ivim(signals, bvals, D_model, ivim_model, bval_cutoff, init_D):
    '''
    "Segmented" (Merisaari 2017):
        Step 1: Use monoexp. equ. to estimate D.
        S(b) = S0 exp(-b*D) -> ln(S(b)) = -b*D + ln(S0) -> y = -m *x + b
        
        Step 2: Fix D in full IVIM model equation using estimate from step 1.
        Fit the S0, Dstar and f using S(b) = S0*((1-f)*exp(-D*b)+ f*exp(-Dstar*b))
    '''
    
    # STEP 1
    params1 = D_model.make_params()
    params1.add('m', value = init_D)
    params1.add('b', value = np.log(signals[0]))
    
    params2 = ivim_model.make_params()
    params2.add('Dstar', value = 100*init_D)
    
    ind1 = np.where(bvals >= bval_cutoff)
    ln_signals = np.log(signals)
    
    # First fit for D
    try:
        result1 = D_model.fit(ln_signals[ind1], x=bvals[ind1], params=params1)
        
    except ValueError:
        return(0,0,0,0,0)
    
    D = result1.best_values['m']
    intercept = result1.best_values['b']
    # CI_1 = result1.conf_interval(sigmas=[2])
    
    # Estimate f_init using intercept
    f_init = 1 - (np.exp(intercept))/signals[0]
    
    # Now fit for D*, f and S0. Fix D
    params2.add('D', value = D, vary=False)
    params2.add('f', value = f_init)
    params2.add('S0', value = signals[0])
    
    try: 
        result2 = ivim_model.fit(signals, x=bvals, params=params2, method='leastsq')
    
    except ValueError:
        return(0,0,0,0,0)
    
    Dstar = result2.best_values['Dstar']
    f = result2.best_values['f']
    # S0 = result2.best_values['S0']
    chi_sqr = result2.chisqr
    aic = result2.aic
    residuals = result2.residual
    rmse = (np.sum(residuals**2)/len(residuals))**(1/2) 
    
    # Filter out bad values   
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

    return D_out, Dstar_out, f_out, chi_sqr, rmse, aic, result2  
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
    chi_sqr = result2.chisqr
       
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
        
    return D_out, Dstar_out, f_out, chi_sqr, rmse, aic, result2   

#%%
def fit_voxels_overseg_bval_thres(signals, bvals, D_model, ivim_model, bval_threshold_ls, init_D):
    '''
    Adaptive b-value threshold algorithm IVIM (Wurnig et al. 2015)
    '''
    chi_sqr_ls = []
    for bval_cutoff in bval_threshold_ls:
        
        try:
            D_out, Dstar_out, f_out, chi_sqr, rmse, aic, result = fit_voxels_overseg_ivim(signals, bvals, D_model, ivim_model, bval_cutoff, init_D)
        
        except TypeError:
            return(0,0,0,0,0,0)
    
        chi_sqr_ls.append([chi_sqr])
      
    best_bval_cutoff = bval_threshold_ls[chi_sqr_ls.index(min(chi_sqr_ls))]     # pick b value threshold corresponding to the smallest chisqr

    try:    
        D_out, Dstar_out, f_out, chi_sqr, rmse, aic, result = fit_voxels_overseg_ivim(signals, bvals, D_model, ivim_model, best_bval_cutoff, init_D)
    
    except TypeError:
        return(0,0,0,0,0,0)
    
    aic = result.aic
    residuals = result.residual
    rmse = (np.sum(residuals**2)/len(residuals))**(1/2) 
    
    if rmse > 50:
        return(0,0,0,0,0,0)
    
    
    return D_out, Dstar_out, f_out, best_bval_cutoff, rmse, aic, result2  


#%%# ===== Get Data ====
in_dir = os.path.join(ROOT_DIR, 'SiBiRT2_Data', '3497-1', '20200721_preRT_1')
in_dir = os.path.join(ROOT_DIR, 'SiBiRT2_Data', '3497-1', '20200908_RT_2')


dw_image1, bvals1, dw_image2, bvals2, adc_image1, adc_image2, t2w3d_image  = read_in_data(in_dir)
array1 = sitk.GetArrayFromImage(dw_image1)

# Select voxels
# coords = [[48, 32, 8], [47, 32, 8], [51, 24, 8], [51, 36, 8]]
coords = [[48, 32, 8], [51, 36, 8]]

# coords = [[48, 32, 8], [51, 36, 8]]

# coords = [[51, 24, 8]] # benign
# coords = [[48, 32, 8]] # tumour
# coords = [[51, 36, 8]] # benign
# coords = [[47, 32, 8]] # tumour

# labels = ["Tumour", "Tumour", "Benign", "Benign"]
labels = ["Seg", "Seg bval thres", "Overseg", "Overseg bval thres"]



results = []
init_D = 0.001  # mm^2/s
bval_cutoff = 200
bval_threshold_ls = [100, 200, 300]
residuals_ls = []

for j in range(len(coords)):
    signals = []
    x = coords[j][0]
    y = coords[j][1]
    z = coords[j][2]
    # x = coords[0]
    # y = coords[1]
    # z = coords[2]
    
    # Extract signals 
    indices = [list(bvals1).index(bval) for bval in bvals1]  
    signals = [array1[index, z, y, x] for index in indices]
       
    linear_model = Model(models.linear2, nan_policy='propagate')
    ivim_model = Model(models.ivim, nan_policy='propagate')
    D_out1, Dstar_out1, f_out1, chi_sqr1, rmse1, aic1, result1 = fit_voxels_seg_ivim(signals, bvals1, linear_model, ivim_model, bval_cutoff, init_D)
    D_out2, Dstar_out2, f_out2, best_bval_cutoff2, rmse2, aic2, result2 = fit_voxels_seg_bval_thres(signals, bvals1, linear_model, ivim_model, bval_threshold_ls, init_D)
    D_out3, Dstar_out3, f_out3, chi_sqr3, rmse3, aic3, result3 = fit_voxels_overseg_ivim(signals, bvals1, linear_model, ivim_model, bval_cutoff, init_D)
    D_out4, Dstar_out4, f_out4, best_bval_cutoff4, rmse4, aic4, result4 = fit_voxels_overseg_bval_thres(signals, bvals1, linear_model, ivim_model, bval_threshold_ls, init_D)
# results.append([D, Dstar, f, S0, chi_sqr])

    print(best_bval_cutoff2)
    print(best_bval_cutoff4)
    
    # CI = result2.conf_interval(sigmas=[2])
    # print(result2.ci_report(ndigits=5)) 
    # print('RMSE fit {}: '.format(j+1), rmse)
    # print('AIC fit {}: '.format(j+1), aic)
    # print('-------------------------------------')
    
    # stats = result2.fit_report()
    # print(stats)
    
    # plt.scatter(bvals1, signals,c='black')
    # plt.plot(bvals1, result1.best_fit, '-', label='Segmented')
    # plt.plot(bvals1, result2.best_fit, '-', label='Segmented b-value thres. alg. ({})'.format(best_bval_cutoff2))
    # plt.plot(bvals1, result3.best_fit, '-', label='Over-segmented')
    # plt.plot(bvals1, result4.best_fit, '-', label='Over-segmented b-value thres. alg. ({})'.format(best_bval_cutoff4))
    plt.plot(bvals1, result1.residual, '-', label='Segmented')
    plt.plot(bvals1, result2.residual, '-', label='Segmented b-value thres. alg. ({})'.format(best_bval_cutoff2))
    plt.plot(bvals1, result3.residual, '-', label='Over-segmented')
    plt.plot(bvals1, result4.residual, '-', label='Over-segmented b-value thres. alg. ({})'.format(best_bval_cutoff4))

    

plt.xlabel("b-values $s/mm^2$")
# plt.ylabel(r"S(b)")
plt.ylabel(r"Residuals")

plt.title("DWI Signal Attenuation Versus b-value for the Four Different IVIM Fitting Methods")
plt.legend(prop={'size': 8})  
plt.show()

# print("[D, Dstar, f, S0, chi_sqr]")
# print(results)