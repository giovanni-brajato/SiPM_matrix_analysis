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

performGlobalCalibration = False
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
from scipy import stats
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

def linear_extrap(TrEff,TrEffp,VDacp,useLog):
    TrEff = np.asarray(TrEff).reshape(-1)
    lowerTrEffpointMask = np.asarray(TrEff <= TrEffp.min()).reshape(-1)
    higherTrEffpointMask = np.asarray(TrEff >= TrEffp.max()).reshape(-1)
    middleTrEffpointMask = ~lowerTrEffpointMask & ~higherTrEffpointMask

    Vdac = np.zeros(np.size(np.asarray(TrEff)))
    if sum(lowerTrEffpointMask) == 1:
        lowerRampDacValue= VDacp[len(TrEffp) - np.argmin(np.flip((TrEff[lowerTrEffpointMask] - np.maximum.accumulate(TrEffp)) ** 2)) - 1]
        Vdac[lowerTrEffpointMask] = lowerRampDacValue
    else:
        try:
            lowerRampDacValue = np.polyfit(TrEffp[:3], VDacp[:3], 1)
            Vdac[lowerTrEffpointMask] = lowerRampDacValue[0] * TrEff[lowerTrEffpointMask] + lowerRampDacValue[1]
        except:
            Vdac[lowerTrEffpointMask] = -np.Inf

    if sum(higherTrEffpointMask) == 1:
        higherRampDacValue = VDacp[np.argmin(((TrEff[higherTrEffpointMask] - np.maximum.accumulate(TrEffp)) ** 2))]
        Vdac[higherTrEffpointMask] = higherRampDacValue
    else:
        try:
            higherRampDacValue = np.polyfit(TrEffp[-3:], VDacp[-3:], 1)
            Vdac[higherTrEffpointMask] = higherRampDacValue[0]*TrEff[higherTrEffpointMask] + higherRampDacValue[1]
        except:
            Vdac[higherTrEffpointMask] =+np.Inf
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

root_folder_name = ""

for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith(".csv"):
            root_folder_name = " ".join(root.split('\\')[-2].split('_')[1:])
            fileList.append(os.path.join(root, file))
            fileInfo = file.split('_')
            chList.append(int(fileInfo[1][fileInfo[2].find('CH')-1:]))
            FDList.append(fileInfo[3][0])
            QTList.append(fileInfo[2])
            AsicList.append(int(root[root.find("ASIC_")+5]))
plt.close('all')




inverse_scurve_list = []
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
v10minval = 1023
v10maxval = 0
v6minval = 63
v6maxval =0
m = -1
winsize = 0

trigger_efficiency_list = np.zeros((4,32,1024))
efficiency = np.linspace(0.0, 100.0, num=100, endpoint=True)
calibration_efficiency = 50.0
position_at_calibration_point = []
valid_scurve = np.zeros((4, 32)).astype(bool)
v10_eta = np.zeros((4, 32))
v6_eta = np.zeros((4, 32))
vdac10 = np.arange(1024).astype(int)
for file in fileList:
    m +=1

    with open(file, 'r') as currentFile:
        data = pd.read_csv(currentFile,sep=',')
        if FDList[m]=='D':
            raw_trigger_efficiency = data.values[:, 3:].mean(1)/ max(data.values[:, 3:].mean(1)) * 100
            raw_trigger_efficiency = np.maximum.accumulate(raw_trigger_efficiency)

            raw_v10dac = data.values[:, 0].astype(int)
            avail_values_mask = np.concatenate((np.zeros(raw_v10dac[0]),np.ones(len(raw_v10dac)),np.zeros(1023-raw_v10dac[-1]))).astype(bool)
            unavail_values_mask = ~avail_values_mask
            final_trigger_efficiency = np.zeros(1024)
            final_trigger_efficiency[avail_values_mask] = raw_trigger_efficiency
            final_trigger_efficiency[unavail_values_mask & (np.arange(1024) < raw_v10dac[0])] = min(raw_trigger_efficiency)
            final_trigger_efficiency[unavail_values_mask & (np.arange(1024) > raw_v10dac[-1])] = max(raw_trigger_efficiency)
            trigger_efficiency_list[AsicList[m] % 4,chList[m]] = final_trigger_efficiency
            v10minval = 0
            v10maxval = 1023
            v6dac_list.append(data.values[:, 1].mean())
            v6minval = np.min(np.append(data.values[:, 1],v6minval))
            v6maxval = np.max(np.append(data.values[:, 1], v6maxval))
            prob_est = np.gradient(np.maximum.accumulate(np.convolve(raw_trigger_efficiency, np.ones(10) / 10, 'same')))
            prob_mean_list.append(prob_est / prob_est.sum())
            Asic_List.append(AsicList[m])
            ChList.append(chList[m])
            #invert scurves
            #inverse_scurve_list.append(linear_extrap(efficiency,final_trigger_efficiency,np.arange(1024),False))
            v10_calibration_point = linear_extrap(calibration_efficiency,final_trigger_efficiency,vdac10,False)
            position_at_calibration_point.append(v10_calibration_point)
            if final_trigger_efficiency[0] < final_trigger_efficiency[-1]:
                valid_scurve[AsicList[m] % 4,chList[m]] =True
            else:
                valid_scurve[AsicList[m] % 4,chList[m]] =False
            v10_eta[AsicList[m] % 4,chList[m]] = v10_calibration_point
            v6_eta[AsicList[m] % 4,chList[m]] = data.values[:, 1].mean()


Ncurves = len(position_at_calibration_point)
NAsics= len(np.unique(np.asarray(Asic_List)))


c10 = 0.92e-3
c6 = 1.5e-3


#v6_eta_cal = v10_eta*c10/c6 - v6_eta
v6_eta_cal = v10_eta*c10/c6
v6_eta_cal =  np.round(v6_eta_cal + v6_eta)
v6_eta_cal = np.round(v6_eta_cal - v6_eta_cal[valid_scurve].min())
v6_eta_cal = v6_eta_cal[valid_scurve].max() - v6_eta_cal
v6_eta_cal[v6_eta_cal<0]=0
v6_eta_cal[v6_eta_cal>63]=63
v6_eta_cal[np.isnan(v6_eta_cal)] = 0
# now if we calculate the new points

v10_eta_post = v10_eta + c6/c10*(v6_eta_cal+v6_eta)
delta_v10 =v10_eta_post -v10_eta
delta_v10[np.isnan(delta_v10)] = 0

minimal_vth_time=np.zeros((4, 32)).astype(int)
print("V6bit Calibration data for " + root_folder_name)
for a in range(4):
    if a == 0: asicN = 4
    else: asicN=a
    for c in range(32):
        print("ASIC_" + str(asicN) + " sixbDAC_ch" + str(c) + "  " + str(int(v6_eta_cal[a,c])))
        nz_indeces = np.nonzero(trigger_efficiency_list[a,c,:])[0]
        minimal_vth_time[a,c] = nz_indeces[0]-1
print("Maximum vth time before noise: " + str((np.min(minimal_vth_time[valid_scurve]))))

# generate N visually distinct colours
Channelcolors = distinctipy.get_colors(32, rng=1)


fig, axs = plt.subplots(NAsics,sharey=True,sharex=True)
for m in range(Ncurves):
    axs[Asic_List[m]-1].plot(vdac10, trigger_efficiency_list[Asic_List[m] % 4,ChList[m],:], color=Channelcolors[ChList[m]], label="CH" + str(ChList[m]), linewidth=2)
    axs[Asic_List[m] - 1].plot(v10_eta[Asic_List[m] % 4,ChList[m]],calibration_efficiency,color=Channelcolors[ChList[m]],marker='o',linestyle=None)
    axs[Asic_List[m] - 1].set_ylabel('Trigger efficiency %')
    axs[Asic_List[m] - 1].set_title("ASIC "+ str(Asic_List[m]))
axs[-1].set_xlabel('Vth time 10 bit DAC units')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(),loc = 'lower center', ncol=8, labelspacing=0.)
fig.suptitle(root_folder_name)

fig, axs = plt.subplots(NAsics,sharey=True,sharex=True)
for m in range(Ncurves):
    axs[Asic_List[m]-1].plot(vdac10+delta_v10[Asic_List[m]%4,ChList[m]], trigger_efficiency_list[Asic_List[m] % 4,ChList[m],:], color=Channelcolors[ChList[m]], label="CH" + str(ChList[m]), linewidth=2)
    #axs[Asic_List[m]-1].plot(vdac10, trigger_efficiency_list[AsicList[m] % 4,chList[m],:], color=Channelcolors[ChList[m]], label="CH" + str(ChList[m]), linewidth=2)
    axs[Asic_List[m] - 1].plot(v10_eta_post[Asic_List[m]%4,ChList[m]],calibration_efficiency,color=Channelcolors[ChList[m]],marker='o',linestyle=None)
    axs[Asic_List[m] - 1].set_ylabel('Trigger efficiency %')
    axs[Asic_List[m] - 1].set_title("ASIC "+ str(Asic_List[m]))
axs[-1].set_xlabel('Vth time 10 bit DAC units')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(),loc = 'lower center', ncol=8, labelspacing=0.)
fig.suptitle(root_folder_name + ":expected improvement")





plt.show()