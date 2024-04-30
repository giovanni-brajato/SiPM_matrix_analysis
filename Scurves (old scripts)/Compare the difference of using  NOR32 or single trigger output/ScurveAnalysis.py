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
matplotlib.use('QtAgg')
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


os.system('SETLOCAL EnableDelayedExpansion')

fileList = []
chList = []
FDList = []
QTList = []
AsicList = []
InfoList = []
configFile = ""
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
            configFile = root + "\\" + file
plt.close('all')

# extract info from the configuration file
scurveConfig = open(configFile, "r")
for line in scurveConfig:
  params = line.split('=')
  if params[0] == "ScurveClockPeriod":
      clkPeriod = float(params[1])
  elif params[0] == "ScurveWindowLength":
      winLength = int(params[1])
  elif params[0] == "ScurveRepetitionNumber":
      numberOfRep = int(params[1])
m = -1


raw_d_counts_new = []
raw_f_counts_new = []
threshold_10bitDAC_units_d_count = []
threshold_10bitDAC_units_f_count = []
threshold_6bitDAC_units_d_count = []
threshold_6bitDAC_units_f_count = []
exp_list = []
prob_from_d_counts= []

for file in fileList:
    m +=1

    with open(file, 'r') as currentFile:
        data = pd.read_csv(currentFile,sep=',')
        if FDList[m]=='D':
            raw_d_counts_new.append(data.values[:, 2:].mean(1))
            threshold_10bitDAC_units_d_count.append(data.values[:, 0])
            threshold_6bitDAC_units_d_count.append(data.values[:, 1])
            lookForFreq = currentFile.name.split("_")

            prob_est = np.gradient(np.maximum.accumulate(np.convolve(raw_d_counts_new[-1],np.ones(10)/10,'same')))
            prob_from_d_counts.append(prob_est/prob_est.sum())
        else:
            raw_f_counts_new.append(data.values[:, 2:].mean(1))
            threshold_10bitDAC_units_f_count.append(data.values[:, 0])
            threshold_6bitDAC_units_f_count.append(data.values[:, 1])

Nexp = len(raw_d_counts_new)


freq_list_number = np.asarray((1e3,10e3,50e3,100e3,500e3,1e6,5e6,10e6,100e6,500e6,1e9))
exp_list= ['Trigb','NOR32']




plt.figure()
for m in range(Nexp):
    plt.plot(threshold_10bitDAC_units_f_count[m], raw_f_counts_new[m]/raw_f_counts_new[m].max(),  label=exp_list[m])
plt.legend()
plt.xlabel('Vth time DAC units')
plt.ylabel('Normalized raw count')
plt.title("Frequency counter on 1s window")

plt.figure()
for m in range(Nexp):
    plt.plot(threshold_10bitDAC_units_f_count[m], raw_f_counts_new[m] / (winLength * clkPeriod),  label=exp_list[m])
plt.legend()
plt.xlabel('Vth time DAC units')
plt.ylabel("Count frequency [Hz]")
plt.title("Frequency counter on 1s window")


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for m in range(Nexp):
    ax1.plot(threshold_10bitDAC_units_d_count[m], raw_d_counts_new[m]/raw_d_counts_new[m].max(),  label=exp_list[m])
ax1.legend()
ax1.set_ylabel('Normalized raw duration')
ax1.set_title("Duration counter on 1s window ")
for m in range(Nexp):
    ax2.plot(threshold_10bitDAC_units_d_count[m], prob_from_d_counts[m]/prob_from_d_counts[m].sum(),  label=exp_list[m])
ax2.legend()
ax2.set_xlabel('Vth time DAC units')
ax2.set_ylabel("Noise probabiliy density ")
ax2.set_title("Estimated noise probability density from duration counter on 1s window")


"""
plt.figure()
cmap = matplotlib.colormaps("gist_rainbow")
norm = matplotlib.colors.SymLogNorm(10, vmin=freq_list_number[0], vmax=freq_list_number[-1])
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
for m in range(Nexp):
    plt.plot(threshold_10bitDAC_units_d_count[m], raw_d_counts_new[m]/raw_d_counts_new[m].max(), color=cmap(norm(freq_list_number[m])), label=freq_list[m])
cbar = plt.colorbar(sm, label="Clock frequency",ticks=freq_list_number, format=matplotlib.ticker.ScalarFormatter(),
                    shrink=1.0, fraction=0.1, pad=0)
cbar.set_ticklabels(freq_list)
plt.legend()
plt.xlabel('Vth time DAC units')
plt.ylabel('Normalized raw duration')
plt.title("Duration counter on 1s window with different clocks")

plt.figure()
cmap = matplotlib.colormaps("gist_rainbow")
norm = matplotlib.colors.SymLogNorm(10, vmin=freq_list_number[0], vmax=freq_list_number[-1])
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
for m in range(Nexp):
    plt.plot(threshold_10bitDAC_units_f_count[m], raw_f_counts_new[m]/raw_f_counts_new[m].max(), color=cmap(norm(freq_list_number[m])), label=freq_list[m])
cbar = plt.colorbar(sm, label="Clock frequency",ticks=freq_list_number, format=matplotlib.ticker.ScalarFormatter(),
                    shrink=1.0, fraction=0.1, pad=0)
cbar.set_ticklabels(freq_list)
plt.legend()
plt.xlabel('Vth time DAC units')
plt.ylabel('Normalized raw count')
plt.title("Frequency counter on 1s window with different clocks")



plt.figure()
cmap = matplotlib.colormaps("gist_rainbow")
norm = matplotlib.colors.SymLogNorm(10, vmin=freq_list_number[0], vmax=freq_list_number[-1])
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
for m in range(Nexp):
    plt.plot(threshold_10bitDAC_units_f_count[m], raw_f_counts_new[m]/ (winLength*clkPeriod), color=cmap(norm(freq_list_number[m])), label=freq_list[m])
cbar = plt.colorbar(sm, label="Clock frequency",ticks=freq_list_number, format=matplotlib.ticker.ScalarFormatter(),
                    shrink=1.0, fraction=0.1, pad=0)
cbar.set_ticklabels(freq_list)
plt.legend()
plt.xlabel('Vth time DAC units')
plt.ylabel("Count frequency [Hz]")
plt.title("Frequency counter on 1s window with different clocks")




plt.figure()
cmap = matplotlib.colormaps("gist_rainbow")
norm = matplotlib.colors.SymLogNorm(10, vmin=freq_list_number[0], vmax=freq_list_number[-1])
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
for m in range(Nexp):
    plt.plot(threshold_10bitDAC_units_d_count[m], prob_from_d_counts[m]/prob_from_d_counts[m].sum(), color=cmap(norm(freq_list_number[m])), label=freq_list[m])
cbar = plt.colorbar(sm, label="Clock frequency",ticks=freq_list_number, format=matplotlib.ticker.ScalarFormatter(),
                    shrink=1.0, fraction=0.1, pad=0)
cbar.set_ticklabels(freq_list)
plt.legend()
plt.xlabel('Vth time DAC units')
plt.ylabel("Noise probabiliy density ")
plt.title("Estimated noise probability density from duration counter on 1s window with different clocks")
"""

plt.show()

