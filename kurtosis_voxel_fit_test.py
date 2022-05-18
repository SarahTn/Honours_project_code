# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:08:16 2022

@author: ST
"""

import sys
import os
sys.path.append('../')
from config.definitions import ROOT_DIR
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from model_dictionary_pl import kurtosis, lin_kurtosis
from read_in_data_pl import read_in_data

#%%
def fit_voxels_kurtosis(S, bvals_tot, bval_cutoff, fit_kurtosis_model, init_Dapp):
    '''
    - corrected_sigs: signals with correction factor to account for different TEs for each b-value dataset.
    - Only fit within tumour to avoid fitting signals below the noise floor.
    '''
    ind1 = np.where(bvals_tot >= bval_cutoff)
    
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
    
   
    # Add parameters to model
    params = fit_kurtosis_model.make_params()
    params.add('Dapp', value = init_Dapp)
    params.add('Kapp', value = 0)
    params.add('S0', value = signals_corr[0])


    # try:     
        # result = fit_linear_model.fit(ln_signals[indx], x=bvals[indx], params=params)
    # result = fit_kurtosis_model.fit(corrected_sigs, x=combined_bvals, params=params)    
    result = fit_kurtosis_model.fit(signals_corr[ind1], x=ordered_bvals[ind1], params=params)    


    # except ValueError:
    #     return 0
    
    S0 = result.best_values['S0']
    Dapp = result.best_values['Dapp']
    Kapp = result.best_values['Kapp']

    Dapp_out = Dapp*1e6      # 10^-6 * mm^2/s 
    if Dapp_out > 2500 or Dapp_out < 500:
        Dapp_out = 0    
    
    if Kapp < 0:
        Kapp = 0

    # return Dapp_out, Kapp, S0, result
    return Dapp_out, Kapp, result, signals_corr


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
        # result = fit_linear_model.fit(ln_signals[indx], x=bvals[indx], params=params)
    # result = fit_kurtosis_model.fit(corrected_sigs, x=combined_bvals, params=params)    
        result = fit_kurtosis_model.fit(ln_signals[ind1], x=ordered_bvals[ind1], params=params)    
        # result = fit_kurtosis_model.fit(ln_signals[ind1], x=bvals_tot[ind1], params=params)    


    except ValueError:
        return 0,0
    
    # S0 = result.best_values['S0']
    Dapp = result.best_values['Dapp']
    Kapp = result.best_values['Kapp']

    Dapp_out = Dapp*1e6      # 10^-6 * mm^2/s 
    if Dapp_out > 2500 or Dapp_out < 500:
        Dapp_out = 0   
    
    if Kapp < 0 or Kapp > 2:
        Kapp = 0

    # return Dapp_out, Kapp, S0, result
    return Dapp_out, Kapp, result, signals_corr

   
#%%# ===== Get Data ====
in_dir = os.path.join(ROOT_DIR, 'SiBiRT2_Data', '3497-1', '20200721_preRT_1')

dw_image1, bvals1, dw_image2, bvals2, adc_image1, adc_image2, t2w3d_image  = read_in_data(in_dir)
array1 = sitk.GetArrayFromImage(dw_image1)
array2 = sitk.GetArrayFromImage(dw_image2)   

array_tot = np.concatenate((array1, array2))
bvals_tot = np.concatenate((bvals1, bvals2))


# Select voxels
coords = [[48, 32, 8], [47, 32, 8], [51, 24, 8], [51, 36, 8]]
labels = ["Tumour", "Tumour", "Benign", "Benign"]

# coords = [[48, 32, 8], [47, 32, 8], [51, 36, 8]]
# labels = ["Tumour", "Tumour", "Benign"]


results = []
init_Dapp = 0.001  # mm^2/s
bval_cutoff = 400

for j in range(len(coords)):
    signals = []
    x = coords[j][0]
    y = coords[j][1]
    z = coords[j][2]

    # Extract signals 
    indices = [list(bvals_tot).index(bval) for bval in bvals_tot]
    signals = [array_tot[index, z, y, x] for index in indices]
    
    ind1 = np.where(np.unique(bvals_tot) >= bval_cutoff)

    # kurtosis_model = Model(lin_kurtosis, nan_policy='propagate')
    kurtosis_model = Model(kurtosis, nan_policy='propagate')

    
    Dapp_out, K_app, result, signals_corr = fit_voxels_kurtosis(signals, bvals_tot, bval_cutoff, kurtosis_model, init_Dapp)

    # Dapp_out, K_app, result, signals_corr = fit_voxels_linear_kurt(signals, bvals_tot, bval_cutoff, kurtosis_model, init_Dapp)
    
    aic = result.aic
    residuals = result.residual
    rmse = (np.sum(residuals**2)/len(residuals))**(1/2) 
    print('RMSE: ', rmse)
   
    stats = result.fit_report()
    print(stats)
    bvals = np.unique(bvals_tot)

    # 
    # plt.scatter(bvals[ind1], np.log(signals_corr[ind1]))
    # plt.scatter(bvals[ind1], signals_corr[ind1])
    
    # plt.plot(bvals[ind1], result.best_fit, '-', label='{}'.format(labels[j]))


    plt.plot(np.linspace(400,2500, num=50), kurtosis(np.linspace(400,2500, num=50), result.best_values['Dapp'], result.best_values['Kapp'], result.best_values['S0']), label='{}'.format(labels[j]))

plt.xlabel("b-values $s/mm^2$")
plt.ylabel(r"$ln(S(b))$")
# plt.title("DWI signal attenuation on logarithmic scale for different b values")
plt.title("DWI Signal Attenuation for Different b-values")

plt.legend(prop={'size': 8})  

print(results)
