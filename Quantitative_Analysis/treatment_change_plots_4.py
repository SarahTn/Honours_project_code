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
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)

# left  = 0.125  # the left side of the subplots of the figure
# right = 0.9    # the right side of the subplots of the figure
# bottom = 0.1   # the bottom of the subplots of the figure
# top = 0.9      # the top of the subplots of the figure
# wspace = 0.2   # the amount of width reserved for blank space between subplots
# hspace = 0.2   # the amount of height reserved for white space between subplots


timepoints1 = ['2','3 *','4 *'] 

# parameter: ADC, feature: Range
# All patients
avg_vals1 = [2022.4, 1867.1, 2117.1]
std_err_vals1 = [119.9, 165.8, 132.9]

CV1 = 6.35
std_err_CV1 = 1.26
upper_bound1 = [avg_vals1[0] + (1.96*std_err_CV1/100) * avg_vals1[0] for tp in range(len(timepoints1))]
lower_bound1 = [avg_vals1[0] - (1.96*std_err_CV1/100) * avg_vals1[0] for tp in range(len(timepoints1))]
# upper_bound = avg_vals[0] + 
# (1.96*std_err_CV/100) * avg_vals[0]
# lower_bound = avg_vals[0] - (1.96*std_err_CV/100) * avg_vals[0]

# plt.plot(timepoints, avg_vals, linewidth=3)
tnrfont = {'fontname':'Times New Roman'}
ax[0].errorbar(timepoints1, avg_vals1, yerr=std_err_vals1, ecolor='black', color='tab:blue', capsize=4, linewidth=4)
# sns.relplot(x=timepoints, y=avg_vals, ci='sd');
ax[0].plot(timepoints1, upper_bound1, '--', color='orange', linewidth=4)
ax[0].plot(timepoints1, lower_bound1, '--', color='orange', linewidth=4)
ax[0].fill_between(timepoints1, upper_bound1[0],  lower_bound1[0], alpha=0.2, color='orange')
ax[0].set_ylabel('ADC ($10^{-6} mm^2/s$)', **tnrfont, fontsize=28)
ax[0].set_xlabel('DWI scan', **tnrfont, fontsize=28)
ax[0].set_title('ADC Range in Tumour', **tnrfont, fontsize=28)
ax[0].tick_params(axis='x', labelsize=24)
ax[0].tick_params(axis='y', labelsize=24)

# parameter: Kurtosis, feature: Range
# All patients
timepoints2 = ['2','3 *','4 *'] 
avg_vals2 = [2.26, 2.10, 1.84]
std_err_vals2 = [0.15, 0.20, 0.30]

CV2 = 15.2
std_err_CV2 = 4.24


upper_bound2 = [avg_vals2[0] + (1.96*std_err_CV2/100) * avg_vals2[0] for tp in range(len(timepoints2))]
lower_bound2 = [avg_vals2[0] - (1.96*std_err_CV2/100) * avg_vals2[0] for tp in range(len(timepoints2))]


ax[1].plot(timepoints2, avg_vals2, linewidth=3)
ax[1].errorbar(timepoints2, avg_vals2, yerr=std_err_vals2, ecolor='black', color='tab:blue', capsize=4, linewidth=4)
# sns.relplot(x=timepoints, y=avg_vals, ci='sd');
ax[1].plot(timepoints2, upper_bound2, '--', color='orange', linewidth=4)
ax[1].plot(timepoints2, lower_bound2, '--', color='orange', linewidth=4)
ax[1].fill_between(timepoints2, upper_bound2[0],  lower_bound2[0], alpha=0.2, color='orange')
ax[1].set_ylabel('Kurtosis',**tnrfont,  fontsize=28)
ax[1].set_xlabel('DWI scan', **tnrfont, fontsize=28)
ax[1].set_title('Kurtosis Range in Tumour', **tnrfont, fontsize=28)
ax[1].tick_params(axis='x', labelsize=24)
ax[1].tick_params(axis='y', labelsize=24)

# parameter: D overseg bval thres ivim, feature: stdev
# All patients
timepoints3 = ['2','3 *','4 *'] 
avg_vals3 = [1143.87, 1221.38, 1194.17]
std_err_vals3 = [31.42, 36.94, 40.63]

# CV3 = 14.2
# std_err_CV3 = 4.9
CV3 = 3.29
std_err_CV3 = 1.23


upper_bound3 = [avg_vals3[0] + (1.96*std_err_CV3/100) * avg_vals3[0] for tp in range(len(timepoints3))]
lower_bound3 = [avg_vals3[0] - (1.96*std_err_CV3/100) * avg_vals3[0] for tp in range(len(timepoints3))]


ax[2].plot(timepoints3, avg_vals3, linewidth=3)
tnrfont = {'fontname':'Times New Roman'}
ax[2].errorbar(timepoints3, avg_vals3, yerr=std_err_vals3, ecolor='black', color='tab:blue', capsize=4, linewidth=4)
# sns.relplot(x=timepoints, y=avg_vals, ci='sd');
ax[2].plot(timepoints3, upper_bound3, '--', color='orange', linewidth=4)
ax[2].plot(timepoints3, lower_bound3, '--', color='orange', linewidth=4)
ax[2].fill_between(timepoints3, upper_bound3[0],  lower_bound3[0], alpha=0.2, color='orange')
ax[2].set_ylabel('$D_{IVIM}$ ($10^{-6} mm^2/s$)', **tnrfont, fontsize=28)
ax[2].set_xlabel('DWI scan', **tnrfont, fontsize=28)
ax[2].set_title('Median Diffusion in Tumour', **tnrfont, fontsize=28)
ax[2].tick_params(axis='x', labelsize=24)
ax[2].tick_params(axis='y', labelsize=24)


# parameter: f, feature: stdev
# All patients
timepoints4 = ['2','3 *','4'] 
avg_vals4 = [6.894, 6.513, 7.33]
std_err_vals4 = [0.199, 0.329, 0.4959]

CV4 = 5.23
std_err_CV4 = 1.45

upper_bound4 = [avg_vals4[0] + (1.96*std_err_CV4/100) * avg_vals4[0] for tp in range(len(timepoints4))]
lower_bound4 = [avg_vals4[0] - (1.96*std_err_CV4/100) * avg_vals4[0] for tp in range(len(timepoints4))]


# plt.plot(timepoints, avg_vals, linewidth=3)
tnrfont = {'fontname':'Times New Roman'}
ax[3].errorbar(timepoints4, avg_vals4, yerr=std_err_vals4, ecolor='black', color='tab:blue', capsize=4, linewidth=4)
# sns.relplot(x=timepoints, y=avg_vals, ci='sd');
ax[3].plot(timepoints4, upper_bound4, '--', color='orange', linewidth=4)
ax[3].plot(timepoints4, lower_bound4, '--', color='orange', linewidth=4)
ax[3].fill_between(timepoints4, upper_bound4[0],  lower_bound4[0], alpha=0.2, color='orange')
ax[3].set_ylabel('$f_{IVIM}$', **tnrfont, fontsize=28)
ax[3].set_xlabel('DWI scan', **tnrfont, fontsize=28)
ax[3].set_title('$f_{IVIM}$ Standard Deviation in Tumour', **tnrfont, fontsize=28)
ax[3].tick_params(axis='x', labelsize=24)
ax[3].tick_params(axis='y', labelsize=24)


