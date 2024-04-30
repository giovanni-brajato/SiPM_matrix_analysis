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




for file in fileList:
    m +=1

    with open(file, 'r') as currentFile:
        data = pd.read_csv(currentFile,sep=',')
        if FDList[m]=='D':
            d_counts_mean_list.append(data.values[:, 2:].mean(1))
            d_counts_std_list.append(data.values[:, 2:].std(1))
            v10dac_list.append(data.values[:, 0])
            v10minval = np.min(np.append(data.values[:, 0],v10minval))
            v10maxval = np.max(np.append(data.values[:, 0], v10maxval))
            v6dac_list.append(data.values[:, 1])
            v6minval = np.min(np.append(data.values[:, 1],v6minval))
            v6maxval = np.max(np.append(data.values[:, 1], v6maxval))
            prob_est = np.gradient(np.maximum.accumulate(np.convolve(d_counts_mean_list[-1], np.ones(10) / 10, 'same')))
            prob_mean_list.append(prob_est / prob_est.sum())
            Asic_List.append(AsicList[m])
            ChList.append(chList[m])
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





plt.show()