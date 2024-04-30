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
vindac_list = []
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
vindacminval = 0
vindacmaxval = 255

m = -1




for file in fileList:
    m +=1

    with open(file, 'r') as currentFile:
        data = pd.read_csv(currentFile,sep=',')
        if FDList[m]=='D':
            d_counts_mean_list.append(data.values[:, 3:].mean(1))
            d_counts_std_list.append(data.values[:, 3:].std(1))
            v10dac_list.append(data.values[:, 0])
            v10minval = np.min(np.append(data.values[:, 0],v10minval))
            v10maxval = np.max(np.append(data.values[:, 0], v10maxval))
            v6dac_list.append(data.values[:, 1])
            v6minval = np.min(np.append(data.values[:, 1],v6minval))
            v6maxval = np.max(np.append(data.values[:, 1], v6maxval))
            vindac_list.append(data.values[:,2])
            vindacminval = np.min(np.append(data.values[:, 2],vindacminval))
            vindacmaxval = np.max(np.append(data.values[:, 2],vindacmaxval))
            prob_est = np.gradient(np.maximum.accumulate(np.convolve(d_counts_mean_list[-1], np.ones(10) / 10, 'same')))
            prob_mean_list.append(prob_est / prob_est.sum())
            Asic_List.append(AsicList[m])
            ChList.append(chList[m])
            currFolders = currentFile.name.split("\\")
            parameterList.append("".join(currFolders[7].split('_')[2:]))

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


N_v10th, v10th_indeces= np.unique(np.asarray(v10dac_list),return_inverse=True)
N_v6th, v6th_indeces = np.unique(np.asarray(v6dac_list),return_inverse=True)
N_vindac, vindac_indeces = np.unique(np.asarray(vindac_list),return_inverse=True)

scurve_set= np.zeros((Ncurves,len(N_v10th),len(N_v6th),len(N_vindac)))

for m in range(Ncurves):
    for i in range(len(d_counts_mean_list[0])):
        scurve_set[m,v10th_indeces[i],v6th_indeces[i],vindac_indeces[i]] = d_counts_mean_list[m][i]



# globalChannelColors
GlobalChannelcolors = distinctipy.get_colors(len(N_v6th))
cmap = matplotlib.cm.get_cmap('rainbow')


# gradient over colors
cmap_v6 = matplotlib.colormaps["viridis"]
norm_v6 = matplotlib.colors.Normalize(vmin=0, vmax=63)
sm_v6 = matplotlib.cm.ScalarMappable(norm=norm_v6, cmap=cmap_v6)
sm_v6.set_array([])


fig, axs = plt.subplots(1,1)
for m in range(Ncurves):
    for v in range(len(N_v6th)):
        axs.plot(N_v10th, scurve_set[m,:,v,:].squeeze()/(scurve_set[m,:,v,:].squeeze().max())*100, color=cmap_v6(norm_v6(N_v6th[v])), linewidth=2)
axs.set_xlabel('Vth 10 bit DAC')
axs.set_ylabel('Trigger efficiency %')
axs.set_title("Noise")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm_v6, cax=cbar_ax, ticks=N_v6th[0:64:3], label="Vth 6bit DAC",format=matplotlib.ticker.ScalarFormatter(), shrink=1.0,
                    fraction=0.1, pad=0)
cbar.set_ticklabels([str(v) for v in N_v6th[0:64:3]])

fig, axs = plt.subplots(1,1)
for m in range(Ncurves):
    for v in range(len(N_v6th)):
        axs.plot(N_v10th, scurve_set[m,:,v,:].squeeze()/(scurve_set[m,:,v,:].squeeze().max())*100, color=cmap_v6(norm_v6(N_v6th[v])), linewidth=2)
axs.set_xlabel('Vth 10 bit DAC')
axs.set_yscale('log')
axs.set_ylabel('Trigger efficiency %')
axs.set_title("Noise")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm_v6, cax=cbar_ax, ticks=N_v6th[0:64:3], label="Vth 6bit DAC",format=matplotlib.ticker.ScalarFormatter(), shrink=1.0,
                    fraction=0.1, pad=0)
cbar.set_ticklabels([str(v) for v in N_v6th[0:64:3]])


# we should check linearity for each trigger efficiency
eta_prompt = np.logspace(-5,2,1000)
V6_linearity_curves = np.zeros((1000,len(N_v6th)))
for v in range(len(N_v6th)):
    V6_linearity_curves[:,v] = linear_extrap(eta_prompt, scurve_set[0,:,v,:].squeeze()/(scurve_set[0,:,v,:].squeeze().max())*100, N_v10th)


cmap_eta = matplotlib.colormaps["viridis"]
norm_eta = matplotlib.colors.LogNorm(vmin=eta_prompt[0], vmax=eta_prompt[-1])
sm_eta = matplotlib.cm.ScalarMappable(norm=norm_eta, cmap=cmap_eta)
sm_eta.set_array([])


fig, axs = plt.subplots(1,1)
coeff_linfit = np.zeros((len(eta_prompt),2))
err_linfit = np.zeros(len(eta_prompt))
for et in range(len(eta_prompt)):
    axs.plot(N_v6th,V6_linearity_curves[et,:], color=cmap_eta(norm_eta(eta_prompt[et])))
    # calculate lineary error
    coeff_linfit[et,:] = np.polyfit(N_v6th, V6_linearity_curves[et,:], 1)
    err_linfit[et] = ((coeff_linfit[et,0]*N_v6th + coeff_linfit[et,1] - V6_linearity_curves[et,:])**2).mean() #mse

axs.set_xlabel('Vth 10 bit DAC')
axs.set_ylabel('Vth 6 bit DAC')
axs.set_title("6bit DAC linearity - Noise")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm_eta, cax=cbar_ax, label="Trigger Efficiency", shrink=1.0,
                    fraction=0.1, pad=0)

plt.figure()
plt.plot(eta_prompt,coeff_linfit[:,0],label='Linear slope')
plt.xlabel('Trigger efficiency %')
plt.ylabel('v10/v6')
plt.xscale('log')
plt.title('Linearity of V6Dac vs V10Dac')
plt.legend()

plt.figure()
plt.plot(eta_prompt,coeff_linfit[:,1],label='Linear Intercept')
plt.xlabel('Trigger efficiency %')
plt.ylabel('v10 DacU')
plt.xscale('log')
plt.title('Linearity of V6Dac vs V10Dac')
plt.legend()

plt.figure()
plt.plot(eta_prompt,err_linfit,label='Linear Error')
plt.xlabel('Trigger efficiency %')
plt.ylabel('Linear MSE DacU^2')
plt.xscale('log')
plt.title('Linearity of V6Dac vs V10Dac')
plt.legend()

plt.figure()
plt.plot(eta_prompt,coeff_linfit[:,0],label='Linear slope')
plt.xlabel('Trigger efficiency %')
plt.ylabel('v10/v6')
plt.title('Linearity of V6Dac vs V10Dac')
plt.legend()

plt.figure()
plt.plot(eta_prompt,coeff_linfit[:,1],label='Linear Intercept')
plt.xlabel('Trigger efficiency %')
plt.ylabel('v10 DacU')
plt.title('Linearity of V6Dac vs V10Dac')
plt.legend()

plt.figure()
plt.plot(eta_prompt,err_linfit,label='Linear Error')
plt.xlabel('Trigger efficiency %')
plt.ylabel('Linear MSE DacU^2')
plt.title('Linearity of V6Dac vs V10Dac')
plt.legend()

plt.show()