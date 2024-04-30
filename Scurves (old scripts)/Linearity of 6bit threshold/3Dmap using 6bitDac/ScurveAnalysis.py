from typing import Tuple, Any
from sklearn.decomposition import FastICA, PCA
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
import pandas as pd
from pylab import figure, cm
import csv
import glob, os
import numpy as np
import math
from scipy import stats

import scipy.signal
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from numpy import matlib
from numpy import linalg as LA
import matplotlib.pyplot as plt
from distinctipy import distinctipy
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
from numpy import ndarray
import statsmodels.api as sm
from statsmodels.graphics import tsaplots

import scipy.signal
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from numpy import matlib
import matplotlib
matplotlib.use('TkAgg')
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
plt.rcParams.update({'figure.max_open_warning': 0})
from sklearn.linear_model import LinearRegression
from distinctipy import distinctipy
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
from numpy import ndarray
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from matplotlib import cm
from matplotlib.ticker import LinearLocator


os.system('SETLOCAL EnableDelayedExpansion')

fileList = []
chList = []
FDList = []
QTList = []
AsicList = []
InfoList = []
configFileList = []
oldMethodFile = "";
cwd = os.getcwd()
os.chdir(cwd)

clkPeriod = 0
winLength = 0
numberOfRep = 0



for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith(".csv"):
            if file == "old_method.csv":
                oldMethodFile = root + "\\" + file
            else:
                fileList.append(os.path.join(root, file))
                fileInfo = file.split('_')
                chList.append(int(fileInfo[1][fileInfo[2].find('CH')-1:]))
                FDList.append(fileInfo[3][0])
                QTList.append(fileInfo[2])

                AsicList.append(int(root[root.find("ASIC_")+5]))
        elif file.endswith(".config"):
            configFileList.append(root + "\\" + file)
plt.close('all')

m = -1


raw_d_counts_new = []
raw_d_counts_new_std = []
raw_f_counts_new = []
raw_f_counts_new_std = []
threshold_10bitDAC_units_d_count = []
threshold_10bitDAC_units_f_count = []
threshold_6bitDAC_units_d_count = []
threshold_6bitDAC_units_f_count = []
freq_list = []
prob_from_d_counts= []
prob_from_d_counts_std= []
for file in fileList:
    m +=1

    with open(file, 'r') as currentFile:
        data = pd.read_csv(currentFile,sep=',')
        if FDList[m]=='D':
            raw_d_counts_new.append(data.values[:, 2:].mean(1))
            raw_d_counts_new_std.append(data.values[:, 2:].std(1))
            threshold_10bitDAC_units_d_count.append(data.values[:, 0])
            threshold_6bitDAC_units_d_count.append(data.values[:, 1])
            lookForFreq = currentFile.name.split("_")
            freq_list.append(lookForFreq[1])
            prob_est = np.gradient(np.maximum.accumulate(np.convolve(raw_d_counts_new[-1],np.ones(10)/10,'same')))
            prob_from_d_counts.append(prob_est/prob_est.sum())
        else:
            raw_f_counts_new.append(data.values[:, 2:].mean(1))
            threshold_10bitDAC_units_f_count.append(data.values[:, 0])
            threshold_6bitDAC_units_f_count.append(data.values[:, 1])

Nexp = len(raw_d_counts_new)
clkPeriod = []
winLength = []
numberOfRep = []
# extract info from the configuration file
for configFile in configFileList:
    scurveConfig = open(configFile, "r")
    for line in scurveConfig:
      params = line.split('=')
      if params[0] == "ScurveClockPeriod":
          clkPeriod.append(float(params[1]))
      elif params[0] == "ScurveWindowLength":
          winLength.append(int(params[1]))
      elif params[0] == "ScurveRepetitionNumber":
          numberOfRep.append(int(params[1]))

threshold_10bit_vector = np.unique(np.asarray(threshold_10bitDAC_units_d_count))
threshold_6bit_vector = np.unique(np.asarray(threshold_6bitDAC_units_d_count))
threshold_10bit_vector_f = np.unique(np.asarray(threshold_10bitDAC_units_f_count))
threshold_6bit_vector_f = np.unique(np.asarray(threshold_6bitDAC_units_f_count))


column_data_duration = np.hstack((np.asarray(threshold_10bitDAC_units_d_count).reshape(-1, 1), np.asarray(threshold_6bitDAC_units_d_count).reshape(-1, 1), np.asarray(raw_d_counts_new).reshape(-1, 1))).astype(int)
column_data_frequency = np.hstack((np.asarray(threshold_10bitDAC_units_f_count).reshape(-1, 1), np.asarray(threshold_6bitDAC_units_f_count).reshape(-1, 1), np.asarray(raw_f_counts_new).reshape(-1, 1))).astype(int)

thresholdCountMap = np.empty((len(threshold_6bit_vector),len(threshold_10bit_vector)))
thresholdCountMap[:] = 0
thresholdFreqMap = np.empty((len(threshold_6bit_vector_f),len(threshold_10bit_vector_f)))
thresholdFreqMap[:] = 0

for i in range(len(column_data_duration)):
    ind_10b = np.where(column_data_duration[i,0] == threshold_10bit_vector)
    ind_6b  = np.where(column_data_duration[i,1] == threshold_6bit_vector)
    thresholdCountMap[ind_6b,ind_10b] = column_data_duration[i,2]

for i in range(len(column_data_frequency)):
    ind_10b = np.where(column_data_frequency[i, 0] == threshold_10bit_vector_f)
    ind_6b = np.where(column_data_frequency[i, 1] == threshold_6bit_vector_f)
    thresholdFreqMap[ind_6b,ind_10b] = column_data_frequency[i,2]


T10,T6= np.meshgrid(threshold_10bit_vector, threshold_6bit_vector)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(T10, T6, thresholdCountMap/thresholdCountMap.max()*100, cmap=cm.jet,
                       linewidth=0, antialiased=False)
ax.set_zlim(0, 100)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('Vth 10 bit DAC')
ax.set_ylabel('Vth 6 bit DAC')
ax.set_zlabel('Trigger efficiency %')


T10f,T6f= np.meshgrid(threshold_10bit_vector_f, threshold_6bit_vector_f)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(T10, T6, thresholdFreqMap, cmap=cm.jet,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('Vth 10 bit DAC')
ax.set_ylabel('Vth 6 bit DAC')
ax.set_zlabel('Trigger edge frequency ')



plt.show()

