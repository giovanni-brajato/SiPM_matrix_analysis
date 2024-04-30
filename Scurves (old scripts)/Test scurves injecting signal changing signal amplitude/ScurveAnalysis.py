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
from numpy.linalg import matrix_rank
from gekko import GEKKO
from distinctipy import distinctipy


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
import statsmodels.api as sm_ch
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
import statsmodels.api as sm_ch
from statsmodels.graphics import tsaplots

def linear_extrap(TrEff,TrEffp,VDacp):
    lowerTrEffpointMask = TrEff <= TrEffp.min()
    higherTrEffpointMask = TrEff >= TrEffp.max()
    middleTrEffpointMask = ~lowerTrEffpointMask & ~higherTrEffpointMask

    Vdac = np.zeros(len(TrEff))
    if sum(lowerTrEffpointMask) == 1:
        lowerRampDacValue= VDacp[len(TrEffp) - np.argmin(np.flip((TrEff[lowerTrEffpointMask] - np.maximum.accumulate(TrEffp)) ** 2)) - 1]
        Vdac[lowerTrEffpointMask] = lowerRampDacValue
    else:
        lowerRampDacValue = np.polyfit(TrEffp[:3], VDacp[:3], 1)
        Vdac[lowerTrEffpointMask] = lowerRampDacValue[0] * TrEff[lowerTrEffpointMask] + lowerRampDacValue[1]

    if sum(higherTrEffpointMask) == 1:
        higherRampDacValue = VDacp[np.argmin(((TrEff[higherTrEffpointMask] - np.maximum.accumulate(TrEffp)) ** 2))]
        Vdac[higherTrEffpointMask] = higherRampDacValue
    else:
        higherRampDacValue = np.polyfit(TrEffp[-3:], VDacp[-3:], 1)
        Vdac[higherTrEffpointMask] = higherRampDacValue[0]*TrEff[higherTrEffpointMask] + higherRampDacValue[1]
    Vdac[middleTrEffpointMask] = np.interp(TrEff[middleTrEffpointMask], TrEffp, VDacp)
    return Vdac



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


for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith(".csv"):
            fileList.append(os.path.join(root, file))
            fileInfo = file.split('_')
            chList.append(int(fileInfo[1][fileInfo[2].find('CH')-1:]))
            FDList.append(fileInfo[3][0])
            QTList.append(fileInfo[2])
            AsicList.append(int(root[root.find("ASIC_")+5]))
        elif file.endswith(".config"):
            configFileList.append(root + "\\" + file)
plt.close('all')



d_counts_mean_list = []
d_counts_std_list = []
v10dac_list = []
v6dac_list = []
prob_mean_list= []
prob_std_list= []
winSizeList =[]
clkPeriodList =[]
useNOR32List = []
Asic_List= []
ChList= []
numberOfRepList = []
rootFolderNameList = []
parameterList = []
v10minval = 1023
v10maxval = 0
v6minval = 63
v6maxval =0
m = -1



winsize = 0
for file in fileList:
    m +=1

    with open(file, 'r') as currentFile:
        data = pd.read_csv(currentFile,sep=',')
        if FDList[m]=='D':
            currFolders = currentFile.name.split("\\")
            parameterList.append("".join(currFolders[6].split('_')[2:]))
            rootFolderNameList.append(currFolders[currFolders.index("ASIC_" + str(AsicList[m])) - 2])
            currConfigFile = "\\".join(map(str, currFolders[ :currFolders.index("ASIC_"+str(AsicList[m]))])) + "\\" + "scurves.config"
            if currConfigFile in configFileList:
                scurveConfig = open(currConfigFile, "r")
                for line in scurveConfig:
                    params = line.split('=')
                    if params[0] == "ScurveClockPeriod":
                        clkPeriodList.append(float(params[1]))
                    elif params[0] == "ScurveWindowLength":
                        winSizeList.append(int(params[1]))
                        winsize = int(params[1])
                    elif params[0] == "ScurveRepetitionNumber":
                        numberOfRepList.append(int(params[1]))
                    elif params[0] == "ScurveUseNor32":
                        useNOR32List.append(bool(params[1]))
                scurveConfig.close()
            else:
                clkPeriodList.append("Err")
                winSizeList.append("Err")
                winSizeList.append("Err")
                useNOR32List.append("Err")
            raw_dcount_mean = data.values[:, 2:].mean(1)
            raw_dcount_std = data.values[:, 2:].std(1)
            raw_v10dac = data.values[:, 0]
            avail_values_mask = np.concatenate(
                (np.zeros(raw_v10dac[0]), np.ones(len(raw_v10dac)), np.zeros(1023 - raw_v10dac[-1]))).astype(bool)
            unavail_values_mask = ~avail_values_mask
            final_dcount_mean = np.zeros(1024)
            final_dcount_std = np.zeros(1024)
            final_dcount_mean[avail_values_mask] = raw_dcount_mean
            final_dcount_mean[unavail_values_mask & (np.arange(1024) < raw_v10dac[0])] = 0
            final_dcount_mean[unavail_values_mask & (np.arange(1024) > raw_v10dac[-1])] = winsize
            final_dcount_std[avail_values_mask] = raw_dcount_std
            final_dcount_std[unavail_values_mask] = 0
            final_v10dac = np.arange(1024)
            d_counts_mean_list.append(final_dcount_mean)
            d_counts_std_list.append(final_dcount_std)
            v10dac_list.append(final_v10dac)
            v10minval = 0
            v10maxval = 1023
            v6dac_list.append(data.values[:, 1])
            v6minval = np.min(np.append(data.values[:, 1], v6minval))
            v6maxval = np.max(np.append(data.values[:, 1], v6maxval))
            prob_est = np.gradient(np.maximum.accumulate(np.convolve(d_counts_mean_list[-1], np.ones(10) / 10, 'same')))
            prob_mean_list.append(prob_est / prob_est.sum())
            Asic_List.append(AsicList[m])
            ChList.append(chList[m])




Ncurves = len(d_counts_mean_list)
NAsics= len(np.unique(np.asarray(Asic_List)))
eta_vdac10 = np.ones((len(v10dac_list[0]), 32, NAsics))
vdac10_eta = np.zeros((101, 32, NAsics))
vdac6 = np.zeros((32,NAsics))
eta_prompt = np.arange(101)


N = 32 # number of channels (or parameters to be optimized)
eta = 90
c10 = 0.92e-3
c6 = 1.5e-3
DC = 0.89

# globalChannelColors
GlobalChannelcolors = distinctipy.get_colors(Ncurves)
cmap = matplotlib.cm.get_cmap('rainbow')

plt.figure()
for m in range(Ncurves):
    if (ChList[m]==3):
        #plt.plot(np.asarray(v10dac_list[m]), d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100, color=cmap(m/Ncurves), label="CH"+str(ChList[m]) +","+parameterList[m], linewidth=2)
        plt.plot(np.asarray(v10dac_list[m]), d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100, color=GlobalChannelcolors[m], label="CH"+str(ChList[m]) +","+parameterList[m], linewidth=2)
plt.yscale("log")
plt.xlabel('Vth 10 bit DAC')
plt.ylabel('Trigger efficiency %')
plt.legend()
plt.title("Signal injection")



# calculate actual amplitude

att_dB= np.asarray([10,10,10,20,20,20,30,30,30,0,10])
amplitudeColors = distinctipy.get_colors(len(att_dB))
vin =  np.asarray([20,40,80,80,40,20,20,40,80,20,50])*1e-3
R = 68.1
vout = vin*(R/(R+50))*(10**(-att_dB/20))
plt.figure()
for i in range(len(att_dB)):
    plt.plot(d_counts_mean_list[2] / d_counts_mean_list[2].max(),d_counts_mean_list[3+i] / d_counts_mean_list[3+i].max(),label=parameterList[3+i],color=GlobalChannelcolors[3+i])
plt.xlabel("False Alarm Probability")
plt.ylabel("Detection Probabiltiy")
plt.legend()

plt.figure()

plt.plot(np.asarray(v10dac_list[0]), prob_mean_list[0], color=GlobalChannelcolors[0], label="Noise",
         linewidth=2)
plt.plot(np.asarray(v10dac_list[-1]), prob_mean_list[-1], color=GlobalChannelcolors[-1],
        label="Noise +pulse "+"{:.2f}".format(vout[-1] / 1e-3)+ " mVpp", linewidth=2)
plt.yscale("symlog", linthresh=1e-7)
plt.xlabel('Vth 10 bit DAC')
plt.ylabel('Amplitude Probability Density')
plt.legend()
plt.title("Example of Signal injection, CH3 asic1")

plt.show()