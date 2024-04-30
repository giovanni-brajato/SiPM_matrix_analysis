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

def linear_extrap(TrEff,TrEffp,VDacp,useLog):
    lowerTrEffpointMask = TrEff <= TrEffp.min()
    higherTrEffpointMask = TrEff >= TrEffp.max()
    middleTrEffpointMask = ~lowerTrEffpointMask & ~higherTrEffpointMask

    Vdac = np.zeros(len(TrEff))
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
            if root.find("ASIC_") != -1:
                AsicList.append(int(root[root.find("ASIC_")+5]))
            else:
                AsicList.append("ASIC 1")
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
            parameterList.append("".join(currFolders[7].split('_')[0:]))


Ncurves = len(d_counts_mean_list)
NAsics= len(np.unique(np.asarray(Asic_List)))
eta_vdac10 = np.ones((len(v10dac_list[0]), 32, NAsics))
vdac10_eta = np.zeros((101, 32, NAsics))
vdac6 = np.zeros((32,NAsics))
useLog = True
if useLog:
    eta_prompt = np.logspace(-3,2,1000)
else:
    eta_prompt = np.linspace(1e-3, 100, 1e-3)


N = 32 # number of channels (or parameters to be optimized)
eta = 90
c10 = 0.92e-3
c6 = 1.5e-3
DC = 0.89

# calculate inverse curve hence check dispersion
v_eta = []
for m in range(Ncurves):
    tempDist=  np.maximum.accumulate(d_counts_mean_list[m] / d_counts_mean_list[m].max())*100
    #tempDist = np.concatenate((np.zeros(v10dac_list[m][0]),tempDist/tempDist.max(),np.ones(1023- v10dac_list[m][-1])))*100
    tempDac = np.array(v10dac_list[m])
    v_eta.append(linear_extrap(eta_prompt,tempDist,tempDac,useLog))



# globalChannelColors
GlobalChannelcolors = distinctipy.get_colors(Ncurves)
cmap = matplotlib.cm.get_cmap('rainbow')

plt.figure()
for m in range(Ncurves):
    if parameterList[m]=='Injection':
        plt.plot(np.asarray(v10dac_list[m]), d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100,
                 color='r', label="CH" + str(ChList[m]) + "," + parameterList[m], linewidth=2)

    else:
        plt.plot(np.asarray(v10dac_list[m]), d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100,
                 color='b', label="CH" + str(ChList[m]) + "," + parameterList[m], linewidth=2)

plt.yscale("log")
plt.xlabel('Vth 10 bit DAC')
plt.ylabel('Trigger efficiency %')
plt.legend()
plt.title("Scurve comparison ASIC 1")

fig, axs = plt.subplots(1, Ncurves//2, sharex=True, sharey=True)
for m in range(Ncurves):
    if parameterList[m] == 'Injection':
        axs[m%(Ncurves//2)].plot(np.asarray(v10dac_list[m]), d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100,
                 color='r', label="CH" + str(ChList[m]) + "," + parameterList[m], linewidth=2)

    else:
        axs[m%(Ncurves//2)].plot(np.asarray(v10dac_list[m]), d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100,
                 color='b', label="CH" + str(ChList[m]) + "," + parameterList[m], linewidth=2)
    #axs[m%(Ncurves//2)].set_yscale("log")
    axs[m%(Ncurves//2)].set_xlabel('Vth 10 bit DAC')
    axs[m%(Ncurves//2)].set_ylabel('Trigger efficiency %')
    axs[m%(Ncurves//2)].legend()
fig.suptitle("Scurve comparison ASIC 1")

plt.figure()
for m in range(Ncurves):
    plt.plot(np.asarray(v10dac_list[m]), d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100, color=GlobalChannelcolors[m], label="CH"+str(ChList[m]) +","+parameterList[m], linewidth=2)
        #plt.plot(np.asarray(v10dac_list[m]), d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100, color=cmap(m/Ncurves), label="CH"+str(ChList[m]) +","+parameterList[m], linewidth=2)
plt.xlabel('Vth 10 bit DAC')
plt.ylabel('Trigger efficiency %')
plt.legend()
plt.title("Signal injection")

plt.figure()
for m in range(Ncurves):
    plt.plot(eta_prompt, v_eta[m] , color=GlobalChannelcolors[m], label="CH"+str(ChList[m]) +","+parameterList[m], linewidth=2)
        #plt.plot(np.asarray(v10dac_list[m]), d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100, color=cmap(m/Ncurves), label="CH"+str(ChList[m]) +","+parameterList[m], linewidth=2)
plt.xscale("log")
plt.ylabel('Vth 10 bit DAC')
plt.xlabel('Trigger efficiency %')
plt.legend()
plt.title("Signal injection")

plt.figure()
plt.plot(eta_prompt,np.asarray(v_eta).std(0))
plt.xscale("log")
plt.ylabel('Dispersion DACu2')
plt.xlabel('Trigger efficiency %')
plt.legend()
plt.title("Dispersion with signal injection")

# compute matrix distance
dist_mat_signal = np.zeros((Ncurves//2,Ncurves//2,len(eta_prompt)))
dist_mat_noise = np.zeros((Ncurves//2,Ncurves//2,len(eta_prompt)))
dist_mat_signal_prop = np.zeros((Ncurves//2,Ncurves//2,len(eta_prompt)))
dist_mat_noise_prop = np.zeros((Ncurves//2,Ncurves//2,len(eta_prompt)))


for i in range (Ncurves):
    for j in range(Ncurves):
        if parameterList[i] == parameterList[j]:
            distance = np.sqrt((np.asarray(v_eta[i]) - np.asarray(v_eta[j]))**2)
            if parameterList[i] == "Injection":
                dist_mat_signal[i%(Ncurves//2),j%(Ncurves//2),:] = distance
            else:
                dist_mat_noise[i%(Ncurves//2),j%(Ncurves//2),:] = distance

for et in range(len(eta_prompt)):
    dist_mat_signal_prop[:,:,et] = dist_mat_signal[:,:,et]/(dist_mat_signal[:,:,et].sum())
    dist_mat_noise_prop[:, :, et] = dist_mat_noise[:, :, et] / (dist_mat_noise[:, :, et].sum())

eta_link = np.zeros(len(eta_prompt)).astype(int)
dist_min = np.zeros(len(eta_prompt))
eta_link_prop = np.zeros(len(eta_prompt)).astype(int)
dist_min_prop = np.zeros(len(eta_prompt))

for et in range(len(eta_prompt)):
    distance_sig = dist_mat_signal[:,:,et]
    # calculate the frobenius norm between this matrix and every possible matrix on the other curve
    matrix_norm = np.sum((distance_sig.reshape((Ncurves//2,Ncurves//2,1)) - dist_mat_noise[:,:,1:-1])**2,axis=(0,1))
    matrix_norm_prop = np.sum((dist_mat_signal_prop[:,:,et].reshape((Ncurves//2,Ncurves//2,1)) - dist_mat_noise_prop[:,:,1:-1])**2,axis=(0,1))

    eta_link[et] = np.argmin(matrix_norm)+1
    dist_min[et] = matrix_norm[eta_link[et]-1]
    eta_link_prop[et] = np.argmin(matrix_norm_prop) + 1
    dist_min_prop[et] = matrix_norm_prop[eta_link_prop[et] - 1]


# find lowest disance
calibration_point = np.argmin(dist_min)
calibration_point_prop = np.argmin(dist_min_prop)
etaCal = eta_prompt[eta_link[calibration_point]]
etaCal_prop = eta_prompt[eta_link_prop[calibration_point_prop]]
fif,axs = plt.subplots(2,1,sharex=True)
axs[0].plot(eta_prompt,eta_prompt[eta_link],label="Scurve dispersion match")
axs[0].plot(eta_prompt[calibration_point],etaCal,label=r"Best calibration point,$\eta=$"+str(etaCal)+"%",marker='o',linestyle=None)
axs[0].set_xlabel(r'$\eta$, signal injected')
axs[0].set_ylabel(r'$\eta$, noise')
axs[0].set_xscale('log')
axs[0].legend()
axs[1].plot(eta_prompt,dist_min,label="Minimal Distance")
axs[1].plot(eta_prompt[calibration_point],dist_min[calibration_point],label="Best calibration point",marker='o',linestyle=None)
axs[1].set_xlabel(r'$\eta$, signal injected')
axs[1].set_ylabel('Minimal Frobenius distance')
axs[1].set_xscale('log')
axs[1].legend()

fif,axs = plt.subplots(2,1,sharex=True)
axs[0].plot(eta_prompt,eta_prompt[eta_link_prop],label="Scurve dispersion match")
axs[0].plot(eta_prompt[calibration_point_prop],etaCal_prop,label=r"Best calibration point,$\eta=$"+str(etaCal)+"%",marker='o',linestyle=None)
axs[0].set_xlabel(r'$\eta$, signal injected')
axs[0].set_ylabel(r'$\eta$, noise')
axs[0].set_xscale('log')
axs[0].legend()
axs[1].plot(eta_prompt,dist_min_prop,label="Minimal proportional Distance")
axs[1].plot(eta_prompt[calibration_point_prop],dist_min_prop[calibration_point_prop],label="Best calibration point",marker='o',linestyle=None)
axs[1].set_xlabel(r'$\eta$, signal injected')
axs[1].set_ylabel('Minimal proportional Frobenius distance')
axs[1].set_xscale('log')
axs[1].legend()

plt.figure()
for m in range(Ncurves):
    if parameterList[m]=='Injection':
        plt.plot(np.asarray(v_eta[m]), eta_prompt,
                 color='r', label="CH" + str(ChList[m]) + "," + parameterList[m], linewidth=1)
        plt.plot(np.asarray(v_eta[m])[eta_link[calibration_point]],etaCal,color='r',marker='o',linestyle=None,  label="Match distance solution - Injected" )
    else:
        plt.plot(np.asarray(v_eta[m]), eta_prompt,
                 color='b', label="CH" + str(ChList[m]) + "," + parameterList[m], linewidth=1)
        plt.plot(np.asarray(v_eta[m])[eta_link[calibration_point]], etaCal, color='b', marker='X', linestyle=None,
                 label="Match distance solution - Noise")
plt.yscale("log")
plt.xlabel('Vth 10 bit DAC')
plt.ylabel('Trigger efficiency %')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title("Scurve comparison ASIC 1")

plt.figure()
for m in range(Ncurves):
    if parameterList[m]=='Injection':
        plt.plot(np.asarray(v_eta[m]), eta_prompt,
                 color='r', label="CH" + str(ChList[m]) + "," + parameterList[m], linewidth=1)
        plt.plot(np.asarray(v_eta[m])[eta_link_prop[calibration_point_prop]],etaCal_prop,color='r',marker='o',linestyle=None,  label="Match proportional distance solution - Injected" )
    else:
        plt.plot(np.asarray(v_eta[m]), eta_prompt,
                 color='b', label="CH" + str(ChList[m]) + "," + parameterList[m], linewidth=1)
        plt.plot(np.asarray(v_eta[m])[eta_link_prop[calibration_point_prop]], etaCal_prop, color='b', marker='X', linestyle=None,
                 label="Match proportional distance solution - Noise")
plt.yscale("log")
plt.xlabel('Vth 10 bit DAC')
plt.ylabel('Trigger efficiency %')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title("Scurve comparison ASIC 1")




plt.show()