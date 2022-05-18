# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:46:13 2022

@author: ST
"""
import pandas as pd
# import seaborn as sns
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
# sns.set_theme(style="darkgrid")

# Read in data
data_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\Pipeline\Quantitative_Analysis\stats_results_final_raw.xlsx"
data = pd.read_excel(data_dir)
# data_test_dir = r"C:\Users\slizz\OneDrive - The University of Sydney (Students)\Honours Project\Pipeline\Quantitative_Analysis\galaxies.csv"
# data_test = pd.read_csv(data_test_dir)
# data_test.info()
# data_test.head()
# data.head()


#%%

fig, axes = plt.subplots(2,2,figsize=(20,16))

# fig.suptitle("Example ", size = 38)

ax = axes.ravel()
timepoints = ['2','3','4'] 

# parameter: ADC, feature: Range
# All patients
avg_vals1 = [2022.4, 1867.1, 2117.1]
std_err_vals1 = [119.9, 165.8, 132.9]

CV1 = 6.35
std_err_CV1 = 1.26
upper_bound1 = [avg_vals1[0] + (1.96*std_err_CV1/100) * avg_vals1[0] for tp in range(len(timepoints))]
lower_bound1 = [avg_vals1[0] - (1.96*std_err_CV1/100) * avg_vals1[0] for tp in range(len(timepoints))]
# upper_bound = avg_vals[0] + 
# (1.96*std_err_CV/100) * avg_vals[0]
# lower_bound = avg_vals[0] - (1.96*std_err_CV/100) * avg_vals[0]

# plt.plot(timepoints, avg_vals, linewidth=3)
tnrfont = {'fontname':'Times New Roman'}
ax[0].errorbar(timepoints, avg_vals1, yerr=std_err_vals1, ecolor='black', color='tab:blue', capsize=2, linewidth=3)
# sns.relplot(x=timepoints, y=avg_vals, ci='sd');
ax[0].plot(timepoints, upper_bound1, '--', color='orange', linewidth=2)
ax[0].plot(timepoints, lower_bound1, '--', color='orange', linewidth=2)
ax[0].fill_between(timepoints, upper_bound1[0],  lower_bound1[0], alpha=0.2, color='orange')
ax[0].set_ylabel('ADC ($10^{-6} mm^2/s$)', **tnrfont, fontsize=14)
ax[0].set_xlabel('DWI scan', **tnrfont, fontsize=14)
ax[0].set_title('Apparent Diffusion Coefficient Range in Tumour', **tnrfont, fontsize=14)


# parameter: Kurtosis, feature: Range
# All patients
avg_vals2 = [2.26, 2.10, 1.84]
std_err_vals2 = [0.15, 0.20, 0.30]

CV2 = 15.2
std_err_CV2 = 4.24
upper_bound2 = [avg_vals2[0] + (1.96*std_err_CV2/100) * avg_vals2[0] for tp in range(len(timepoints))]
lower_bound2 = [avg_vals2[0] - (1.96*std_err_CV2/100) * avg_vals2[0] for tp in range(len(timepoints))]


ax[1].plot(timepoints, avg_vals2, linewidth=3)
ax[1].errorbar(timepoints, avg_vals2, yerr=std_err_vals2, ecolor='black', color='tab:blue', capsize=2, linewidth=3)
# sns.relplot(x=timepoints, y=avg_vals, ci='sd');
ax[1].plot(timepoints, upper_bound2, '--', color='orange', linewidth=2)
ax[1].plot(timepoints, lower_bound2, '--', color='orange', linewidth=2)
ax[1].fill_between(timepoints, upper_bound2[0],  lower_bound2[0], alpha=0.2, color='orange')
ax[1].set_ylabel('Kurtosis',**tnrfont,  fontsize=14)
ax[1].set_xlabel('DWI scan', **tnrfont, fontsize=14)
ax[1].set_title('Kurtosis Range in Tumour', **tnrfont, fontsize=14)

# parameter: D overseg bval thres ivim, feature: stdev
# All patients

avg_vals3 = [1143.87, 1221.38, 1194.17]
std_err_vals3 = [31.42, 36.94, 40.63]

CV3 = 14.2
std_err_CV3 = 4.9
upper_bound3 = [avg_vals3[0] + (1.96*std_err_CV3/100) * avg_vals3[0] for tp in range(len(timepoints))]
lower_bound3 = [avg_vals3[0] - (1.96*std_err_CV3/100) * avg_vals3[0] for tp in range(len(timepoints))]


ax[2].plot(timepoints, avg_vals3, linewidth=3)
tnrfont = {'fontname':'Times New Roman'}
ax[2].errorbar(timepoints, avg_vals3, yerr=std_err_vals3, ecolor='black', color='tab:blue', capsize=2, linewidth=3)
# sns.relplot(x=timepoints, y=avg_vals, ci='sd');
ax[2].plot(timepoints, upper_bound3, '--', color='orange', linewidth=2)
ax[2].plot(timepoints, lower_bound3, '--', color='orange', linewidth=2)
ax[2].fill_between(timepoints, upper_bound3[0],  lower_bound3[0], alpha=0.2, color='orange')
ax[2].set_ylabel('Diffusion IVIM ($10^{-6} mm^2/s$)', **tnrfont, fontsize=14)
ax[2].set_xlabel('DWI scan', **tnrfont, fontsize=14)
ax[2].set_title('Median Diffusion in Tumour', **tnrfont, fontsize=14)


# parameter: D overseg bval thres ivim, feature: stdev
# All patients

avg_vals4 = [0.2945, 0.2329, 0.2341]
std_err_vals4 = [0.0255, 0.0349, 0.0449]

CV4 = 14.92
std_err_CV4 = 3.86
upper_bound4 = [avg_vals4[0] + (1.96*std_err_CV4/100) * avg_vals4[0] for tp in range(len(timepoints))]
lower_bound4 = [avg_vals4[0] - (1.96*std_err_CV4/100) * avg_vals4[0] for tp in range(len(timepoints))]


# plt.plot(timepoints, avg_vals, linewidth=3)
tnrfont = {'fontname':'Times New Roman'}
ax[3].errorbar(timepoints, avg_vals4, yerr=std_err_vals4, ecolor='black', color='tab:blue', capsize=2, linewidth=3)
# sns.relplot(x=timepoints, y=avg_vals, ci='sd');
ax[3].plot(timepoints, upper_bound4, '--', color='orange', linewidth=2)
ax[3].plot(timepoints, lower_bound4, '--', color='orange', linewidth=2)
ax[3].fill_between(timepoints, upper_bound4[0],  lower_bound4[0], alpha=0.2, color='orange')
ax[3].set_ylabel('Kurtosis', **tnrfont, fontsize=14)
ax[3].set_xlabel('DWI scan', **tnrfont, fontsize=14)
ax[3].set_title('Kurtosis Standard Deviation in Tumour', **tnrfont, fontsize=14)



