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
from mpl_toolkits.mplot3d import Axes3D

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
import matplotlib as mpl
#mpl.use('TkAgg')
mpl.use('QtAgg')
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

def capaToTime(n,type):
    switch_dict_cin = {
        0 : "25 ns",
        1 : "50 ns",
        2 : "75 ns",
        3 : "100 ns"
    }
    switch_dict_cout = {
        0: "25 ns",
        1: "75 ns",
        2: "50 ns",
        3: "100 ns"
    }
    if type==1:
        return switch_dict_cin.get(n, '')
    else:
        return switch_dict_cout.get(n, '')

os.system('SETLOCAL EnableDelayedExpansion')

fileList = []
cwd = os.getcwd()
os.chdir(cwd)


for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith(".csv"):
            if file == 'BDF_ELECTRONIC_NOISE_Pedestal.csv':
                pedestalData = pd.read_csv(file, sep=',')
                Cpedestal0 = pedestalData.values[0][1:]
                Cpedestal1 = pedestalData.values[1][1:]
            else:
                fileList.append(os.path.join(root, file))
plt.close('all')



CinData = []
CfData = []
VppInData = []
VppInData_mean= []
HoldData = []
ADCuData_mean = []
ADCuData_std = []
AsicChannelData = []

Nexp =len(fileList)
for file in fileList:

    with open(file, 'r') as currentFile:
        data = pd.read_csv(currentFile,sep=',')
        VppInData.append(data.values[:,0])
        VppInData_mean.append(data.values[:,0].mean())
        CinData.append(data.values[:, 1].astype(int))
        CfData.append(data.values[:, 2].astype(int))
        HoldData.append(data.values[:, 3].astype(int))
        ADCuData_mean.append(data.values[:,4:].mean(1))
        ADCuData_std.append(data.values[:,4:].std(1))
        AsicChannelData.append(currentFile.name.split('\\')[7] + '_' + currentFile.name.split('\\')[8].split('.')[0])

VoltageCinCfHold_tensor = np.zeros((Nexp,4,4,256))
VoltageCinCfHold_tensor_std = np.zeros((Nexp,4,4,256))
for v in range(Nexp):
    for ci in range(4):
        for cf in range(4):
            for h in range(256):
                VoltageCinCfHold_tensor[v,ci,cf,:] = ADCuData_mean[v][np.logical_and(CinData[v]==ci,CfData[v]==cf)]
                VoltageCinCfHold_tensor_std[v, ci, cf, :] = ADCuData_std[v][
                    np.logical_and(CinData[v] == ci, CfData[v] == cf)]

# normalize charge by subtracting the pedestal



chargePedestal = np.asarray([Cpedestal0[4],Cpedestal0[4]])
VoltageCinCfHold_tensor = np.reshape(chargePedestal,(Nexp,1,1,1)) - VoltageCinCfHold_tensor
holdVector_ns = np.arange(256)*0.34

VoltageCinCfADC_peakPos =np.zeros((Nexp,4,4))
VoltageCinCf_peakValue = np.zeros((Nexp,4,4))



channelColors =  distinctipy.get_colors(Nexp, rng=2)
#voltageColors = mpl.cm.viridis(np.linspace(0,1,Nexp))
fig, ax = plt.subplots(1)


for cin in range(4):
    for cf in range(4):
        fig, ax = plt.subplots(1)
        for v in range(Nexp):
            ax.plot(holdVector_ns,VoltageCinCfHold_tensor[v,cin,cf,:], label=AsicChannelData[v] + " Vpp "+ f"{VppInData_mean[v]*1e3:.{2}f}"+ " mV",color=channelColors[v])
            ax.fill_between(holdVector_ns, VoltageCinCfHold_tensor[v,cin,cf,:] - VoltageCinCfHold_tensor_std[v,cin,cf,:], VoltageCinCfHold_tensor[v,cin,cf,:] + VoltageCinCfHold_tensor_std[v,cin,cf,:], color=channelColors[v], alpha=0.2)
            PeakPosition = np.argmax(VoltageCinCfHold_tensor[v, cin, cf, :])
            ADCPeakValue = VoltageCinCfHold_tensor[v,cin,cf,PeakPosition]
            holdPeakPosition = holdVector_ns[PeakPosition]
            VoltageCinCfADC_peakPos[v,cin,cf] = holdPeakPosition
            VoltageCinCf_peakValue[v,cin,cf] = ADCPeakValue
        ax.plot(holdPeakPosition, ADCPeakValue,marker='x',color=channelColors[v])
        ax2 = ax.twinx()
        ax2.set_ylabel(r' $\tau_{rise}$=' + capaToTime(cin,1))
        ax2.set_yticklabels([])
        ax.set_xlabel("Hold delay [ns]" )
        ax.set_ylabel("Avg charge [ADCu]")
        ax.set_title(r'$\tau_{fall}$=' + capaToTime(cf,2),fontsize = 10)
        ax.legend()




plt.show()