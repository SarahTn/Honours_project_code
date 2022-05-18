# -*- coding: utf-8 -*-
"""
Created on Sat May 14 11:46:22 2022

@author: ST
"""

# libraries
import numpy as np
import matplotlib.pyplot as plt
 
# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
# [ADC, Divim, D*, Dkurt, f, K, b-val thres.]
bars_tumour = [3.695882033,	3.294063657,	51.72400772,	4.180132831,	25.96161088,	4.314640161,	16.97056275]
 
# Choose the height of the cyan bars
# [ADC, Divim, D*, Dkurt, f, K, b-val thres.]
bars_benign = [2.192104327, 2.658185288, 55.83014527, 0, 18.88169553, 0, 18.85618083]
 
# Choose the height of the error bars (bars1)
yer_tumour = [1.592936051, 1.229999799, 19.18345818, 0.63772287, 13.30301633, 1.57489975,
7.138834846]
 
# Choose the height of the error bars (bars2)
yer_benign = [0.556349044, 0.675225566, 21.01656953, 0,  8.165323299, 0,  7.698003589]
 
# The x position of bars
r1 = np.arange(len(bars_tumour))
r2 = [x + barWidth for x in r1]
 
# Select font
plt.rc('font',family='Times New Roman')

# Create blue bars
plt.bar(r1, bars_tumour, width = barWidth, color = 'mediumvioletred', edgecolor = 'black', yerr=yer_tumour, capsize=7, label='Tumour')
 
# Create cyan bars
plt.bar(r2, bars_benign, width = barWidth, color = 'cornflowerblue', edgecolor = 'black', yerr=yer_benign, capsize=7, label='Benign')
 
# general layout
plt.xticks([r + barWidth/2 for r in range(len(bars_tumour))], ['ADC', '$D_{IVIM}$', '$D^*$', '$D_{Kurt}$', 'f', 'K', 'b-val. thres.'], fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('CV (%)', fontsize=16)
plt.xlabel('Diffusion Model Parameter', fontsize=16)
plt.legend(prop={"size":14})
 
# Show graphic
plt.show()