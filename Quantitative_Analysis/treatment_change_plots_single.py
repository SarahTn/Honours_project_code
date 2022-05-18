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

# parameter: ADC, feature: Range
# All patients

timepoints = ['2','3','4'] 
# timepoints  = [2,3,4]
avg_vals = [2022.4, 1867.1, 2117.1]
std_err_vals = [119.9, 165.8, 132.9]

CV = 6.35
std_err_CV = 1.26
upper_bound = [avg_vals[0] + (1.96*std_err_CV/100) * avg_vals[0] for tp in range(len(timepoints))]
lower_bound = [avg_vals[0] - (1.96*std_err_CV/100) * avg_vals[0] for tp in range(len(timepoints))]
# upper_bound = avg_vals[0] + 
(1.96*std_err_CV/100) * avg_vals[0]
# lower_bound = avg_vals[0] - (1.96*std_err_CV/100) * avg_vals[0]

# plt.plot(timepoints, avg_vals, linewidth=3)
tnrfont = {'fontname':'Times New Roman'}
plt.errorbar(timepoints, avg_vals, yerr=std_err_vals, ecolor='black', capsize=2, linewidth=3)
# sns.relplot(x=timepoints, y=avg_vals, ci='sd');
plt.plot(timepoints, upper_bound, '--', color='orange', linewidth=2)
plt.plot(timepoints, lower_bound, '--', color='orange', linewidth=2)
plt.fill_between(timepoints, upper_bound[0],  lower_bound[0], alpha=0.2, color='orange')
plt.ylabel('ADC ($10^{-6} mm^2/s$)', **tnrfont, fontsize=14)
plt.xlabel('DWI scan', **tnrfont, fontsize=14)
plt.title('Apparent Diffusion Coefficient Range in Tumour', **tnrfont, fontsize=14)

#%%

# parameter: Kurtosis, feature: Range
# All patients
timepoints = ['2','3','4'] 
# timepoints  = [2,3,4]
avg_vals = [2.26, 2.10, 1.84]
std_err_vals = [0.15, 0.20, 0.30]

CV = 15.2
std_err_CV = 4.24
upper_bound = [avg_vals[0] + (1.96*std_err_CV/100) * avg_vals[0] for tp in range(len(timepoints))]
lower_bound = [avg_vals[0] - (1.96*std_err_CV/100) * avg_vals[0] for tp in range(len(timepoints))]
# upper_bound = avg_vals[0] +  (1.96*std_err_CV/100) * avg_vals[0]
# lower_bound = avg_vals[0] - (1.96*std_err_CV/100) * avg_vals[0]

# plt.plot(timepoints, avg_vals, linewidth=3)
tnrfont = {'fontname':'Times New Roman'}
plt.errorbar(timepoints, avg_vals, yerr=std_err_vals, ecolor='black', capsize=2, linewidth=3)
# sns.relplot(x=timepoints, y=avg_vals, ci='sd');
plt.plot(timepoints, upper_bound, '--', color='orange', linewidth=2)
plt.plot(timepoints, lower_bound, '--', color='orange', linewidth=2)
plt.fill_between(timepoints, upper_bound[0],  lower_bound[0], alpha=0.2, color='orange')
plt.ylabel('Kurtosis',**tnrfont,  fontsize=14)
plt.xlabel('DWI scan', **tnrfont, fontsize=14)
plt.title('Kurtosis Range in Tumour', **tnrfont, fontsize=14)

#%%
# parameter: D overseg bval thres ivim, feature: stdev
# All patients
timepoints = ['2','3','4'] 
# timepoints  = [2,3,4]
avg_vals = [1143.87, 1221.38, 1194.17]
std_err_vals = [31.42, 36.94, 40.63]

CV = 14.2
std_err_CV = 4.9
upper_bound = [avg_vals[0] + (1.96*std_err_CV/100) * avg_vals[0] for tp in range(len(timepoints))]
lower_bound = [avg_vals[0] - (1.96*std_err_CV/100) * avg_vals[0] for tp in range(len(timepoints))]
# upper_bound = [avg_vals[0] + (std_err_CV/100) * avg_vals[0] for tp in range(len(timepoints))]
# lower_bound = [avg_vals[0] - (std_err_CV/100) * avg_vals[0] for tp in range(len(timepoints))]
# upper_bound = avg_vals[0] +  (1.96*std_err_CV/100) * avg_vals[0]
# lower_bound = avg_vals[0] - (1.96*std_err_CV/100) * avg_vals[0]

# plt.plot(timepoints, avg_vals, linewidth=3)
tnrfont = {'fontname':'Times New Roman'}
plt.errorbar(timepoints, avg_vals, yerr=std_err_vals, ecolor='black', capsize=2, linewidth=3)
# sns.relplot(x=timepoints, y=avg_vals, ci='sd');
plt.plot(timepoints, upper_bound, '--', color='orange', linewidth=2)
plt.plot(timepoints, lower_bound, '--', color='orange', linewidth=2)
plt.fill_between(timepoints, upper_bound[0],  lower_bound[0], alpha=0.2, color='orange')
plt.ylabel('Diffusion IVIM ($10^{-6} mm^2/s$)', **tnrfont, fontsize=14)
plt.xlabel('DWI scan', **tnrfont, fontsize=14)
plt.title('Median Diffusion in Tumour', **tnrfont, fontsize=14)

#%%
# parameter: D overseg bval thres ivim, feature: stdev
# All patients
timepoints = ['2','3','4'] 
# timepoints  = [2,3,4]
avg_vals = [0.2945, 0.2329, 0.2341]
std_err_vals = [0.0255, 0.0349, 0.0449]

# CV = 14.92
# std_err_CV = 3.86
CV = 3.29
std_err_CV = 1.23
upper_bound = [avg_vals[0] + (1.96*std_err_CV/100) * avg_vals[0] for tp in range(len(timepoints))]
lower_bound = [avg_vals[0] - (1.96*std_err_CV/100) * avg_vals[0] for tp in range(len(timepoints))]
# upper_bound = [avg_vals[0] + (std_err_CV/100) * avg_vals[0] for tp in range(len(timepoints))]
# lower_bound = [avg_vals[0] - (std_err_CV/100) * avg_vals[0] for tp in range(len(timepoints))]
# upper_bound = avg_vals[0] +  (1.96*std_err_CV/100) * avg_vals[0]
# lower_bound = avg_vals[0] - (1.96*std_err_CV/100) * avg_vals[0]

# plt.plot(timepoints, avg_vals, linewidth=3)
tnrfont = {'fontname':'Times New Roman'}
plt.errorbar(timepoints, avg_vals, yerr=std_err_vals, ecolor='black', capsize=2, linewidth=3)
# sns.relplot(x=timepoints, y=avg_vals, ci='sd');
plt.plot(timepoints, upper_bound, '--', color='orange', linewidth=2)
plt.plot(timepoints, lower_bound, '--', color='orange', linewidth=2)
plt.fill_between(timepoints, upper_bound[0],  lower_bound[0], alpha=0.2, color='orange')
plt.ylabel('Kurtosis', **tnrfont, fontsize=14)
plt.xlabel('DWI scan', **tnrfont, fontsize=14)
plt.title('Kurtosis Standard Deviation in Tumour', **tnrfont, fontsize=14)

# hfont = {'fontname':'Helvetica'}


