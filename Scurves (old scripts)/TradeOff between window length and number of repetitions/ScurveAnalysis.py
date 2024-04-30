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

winLength_s = np.asarray(winLength)
numberOfRep_s = np.asarray(numberOfRep)
winLength_s_str = ["1 rep 10000 win","10 rep 1000 win","100 rep 100 win"]

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for m in range(Nexp):
    ax1.errorbar(threshold_10bitDAC_units_d_count[m], raw_d_counts_new[m]/raw_d_counts_new[m].max(),xerr=None,yerr=raw_d_counts_new_std[m]/raw_d_counts_new[m].max(), label=winLength_s_str[m])
ax1.legend()
ax1.set_ylabel('Normalized raw duration')
ax1.set_title("Duration counter 1GHz clock on different windows")

for m in range(Nexp):
    ax2.plot(threshold_10bitDAC_units_d_count[m], prob_from_d_counts[m]/prob_from_d_counts[m].sum(),  label=winLength_s_str[m])
ax2.legend()
ax2.set_xlabel('Vth time DAC units')
ax2.set_ylabel("Noise probabiliy density ")
ax2.set_title("Estimated noise probability density from duration counter 1GHz clock on different windows")


plt.figure()
for m in range(Nexp):
    plt.plot(threshold_10bitDAC_units_d_count[m], raw_d_counts_new[m], label=winLength_s_str[m])
    plt.fill_between(threshold_10bitDAC_units_d_count[m], raw_d_counts_new[m] - raw_d_counts_new_std[m], raw_d_counts_new[m] - raw_d_counts_new_std[m],alpha=0.3)
plt.legend()
plt.xlabel('Vth time DAC units')
plt.ylabel('Raw duration')
plt.title("Duration counter 1GHz clock on different windows")


plt.figure()
for m in range(Nexp):
    plt.plot(threshold_10bitDAC_units_d_count[m], raw_d_counts_new[m]/raw_d_counts_new[m].max()*100,  label=winLength_s_str[m])
    plt.fill_between(threshold_10bitDAC_units_d_count[m], (raw_d_counts_new[m] - raw_d_counts_new_std[m])/raw_d_counts_new[m].max()*100,(raw_d_counts_new[m] + raw_d_counts_new_std[m])/raw_d_counts_new[m].max()*100, alpha=0.3)
plt.legend()
plt.xlabel('Vth time DAC units')
plt.ylabel('Trigger efficiency %')
plt.title("Duration counter 100MHz for different window sizes and repetitions")


plt.figure()
for m in range(Nexp):
    plt.plot(threshold_10bitDAC_units_f_count[m], raw_f_counts_new[m]/raw_f_counts_new[m].max(),  label=winLength_s_str[m])

plt.legend()
plt.xlabel('Vth time DAC units')
plt.ylabel('Normalized raw count')
plt.title("Frequency counter 1GHz clock on different windows")



plt.figure()
for m in range(Nexp):
    plt.plot(threshold_10bitDAC_units_f_count[m], raw_f_counts_new[m]/winLength_s[m],  label=winLength_s_str[m])
plt.legend()
plt.xlabel('Vth time DAC units')
plt.ylabel("Count frequency [Hz]")
plt.title("Frequency counter 1GHz clock on different windows")





plt.show()

