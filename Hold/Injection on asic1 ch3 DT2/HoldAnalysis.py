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
mpl.use('TkAgg')
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
            fileList.append(os.path.join(root, file))
plt.close('all')



CinData = []
CfData = []
VppInData = []
VppInData_mean= []
HoldData = []
ADCuData_mean = []
ADCuData_std = []

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
chargePedestal = 982.6643718592965
VoltageCinCfHold_tensor = chargePedestal - VoltageCinCfHold_tensor
holdVector_ns = np.arange(256)*0.34

VoltageCinCfADC_peakPos =np.zeros((Nexp,4,4))
VoltageCinCf_peakValue = np.zeros((Nexp,4,4))
peak_pos = np.zeros((Nexp,4,4))
peak_pos_ind = np.zeros((Nexp,4,4),dtype=int)

#voltageColors =  distinctipy.get_colors(Nexp, rng=2)
voltageColors = mpl.cm.viridis(np.linspace(0,1,Nexp))
fig, axs = plt.subplots(4, 4, figsize=(10, 10),sharex=True,sharey=True)
# Adjust the spacing between subplots to zero
plt.subplots_adjust(wspace=0, hspace=0)

for cin in range(4):
    for cf in range(4):

        for v in range(Nexp):
            axs[cin,cf].plot(holdVector_ns,VoltageCinCfHold_tensor[v,cin,cf,:], label="Vpp "+ f"{VppInData_mean[v]*1e3:.{2}f}"+ " mV",color=voltageColors[v])
            axs[cin,cf].fill_between(holdVector_ns, VoltageCinCfHold_tensor[v,cin,cf,:] - VoltageCinCfHold_tensor_std[v,cin,cf,:], VoltageCinCfHold_tensor[v,cin,cf,:] + VoltageCinCfHold_tensor_std[v,cin,cf,:], color=voltageColors[v], alpha=0.2)
            PeakPosition = np.argmax(VoltageCinCfHold_tensor[v, cin, cf, :])
            ADCPeakValue = VoltageCinCfHold_tensor[v,cin,cf,PeakPosition]
            holdPeakPosition = holdVector_ns[PeakPosition]
            peak_pos[v,cin,cf] = holdPeakPosition;
            peak_pos_ind[v,cin,cf] = PeakPosition
            VoltageCinCfADC_peakPos[v,cin,cf] = holdPeakPosition
            VoltageCinCf_peakValue[v,cin,cf] = ADCPeakValue
            axs[cin,cf].plot(holdPeakPosition, ADCPeakValue,marker='x',color=voltageColors[v])
        if cf== 3:
            ax2 = axs[cin,cf].twinx()
            ax2.set_ylabel(r' $\tau_{rise}$=' + capaToTime(cin,1))
            ax2.set_yticklabels([])
        if cin==3:
            axs[cin,cf].set_xlabel("Hold delay [ns]" )
        if cf == 0:
            axs[cin,cf].set_ylabel("Avg charge [ADCu]")
        if cin==0:
            axs[cin,cf].set_title(r'$\tau_{fall}$=' + capaToTime(cf,2),fontsize = 10)

        print("t_rise = " + capaToTime(cin,1) + ", t_fall = " + capaToTime(cf,2) + " : Optimal Hold: " + str(peak_pos[:,cin,cf].mean()))

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=Nexp)

fig, axs = plt.subplots(4, 4, figsize=(10, 10),sharex=True,sharey=True)
# Adjust the spacing between subplots to zero
plt.subplots_adjust(wspace=0, hspace=0)

for cin in range(4):
    for cf in range(4):
        for v in range(Nexp):
            axs[cin,cf].plot(VppInData_mean[v]*1e3, VoltageCinCf_peakValue[v,cin,cf], marker='x',color='b')
        if cf== 3:
            ax2 = axs[cin,cf].twinx()
            ax2.set_ylabel(r' $\tau_{rise}$=' + capaToTime(cin,1))
            ax2.set_yticklabels([])
        if cin==3:
            axs[cin,cf].set_xlabel("Input Pulse [mVpp]" )
        if cf == 0:
            axs[cin,cf].set_ylabel("Peak avg charge [ADCu]")
        if cin==0:
            axs[cin,cf].set_title(r'$\tau_{fall}$=' + capaToTime(cf,2),fontsize = 10)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=Nexp)







fig, axs = plt.subplots(4, 4, figsize=(10, 10),sharex=True,sharey=True)
# Adjust the spacing between subplots to zero
plt.subplots_adjust(wspace=0, hspace=0)

for cin in range(4):
    for cf in range(4):
        for v in range(Nexp):
            timingErrorADC = np.gradient(VoltageCinCfHold_tensor[v,cin,cf,:],holdVector_ns)
            axs[cin,cf].plot(holdVector_ns,timingErrorADC, label="Vpp "+ f"{VppInData_mean[v]*1e3:.{2}f}"+ " mV",color=voltageColors[v])
            PeakPosition = np.argmax(VoltageCinCfHold_tensor[v, cin, cf, :])
            timingErrorADCPeakValue = timingErrorADC[PeakPosition]
            holdPeakPosition = holdVector_ns[PeakPosition]
            axs[cin,cf].plot(holdPeakPosition, timingErrorADCPeakValue,marker='x',color='r')
        if cf== 3:
            ax2 = axs[cin,cf].twinx()
            ax2.set_ylabel(r' $\tau_{rise}$=' + capaToTime(cin,1))
            ax2.set_yticklabels([])
        if cin==3:
            axs[cin,cf].set_xlabel("Hold delay [ns]" )
        if cf == 0:
            axs[cin,cf].set_ylabel(r'$\delta$Charge/$\delta$t [ADCu/ns]')
        if cin==0:
            axs[cin,cf].set_title(r'$\tau_{fall}$=' + capaToTime(cf,2),fontsize = 10)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=Nexp)



holdColors = mpl.cm.plasma(np.linspace(0,1,256))
fig, axs = plt.subplots(4, 4, figsize=(10, 10),sharex=True,sharey=True)
# Adjust the spacing between subplots to zero
plt.subplots_adjust(wspace=0, hspace=0)
for cin in range(4):
    for cf in range(4):
        for h in range (256):
            axs[cin, cf].plot(np.asarray(VppInData_mean)*1e3, VoltageCinCfHold_tensor[:, cin, cf, h],color=holdColors[h],alpha=0.1)
        if cf== 3:
            ax2 = axs[cin,cf].twinx()
            ax2.set_ylabel(r' $\tau_{rise}$=' + capaToTime(cin,1))
            ax2.set_yticklabels([])
        if cin==3:
            axs[cin,cf].set_xlabel("Input Pulse [mVpp]" )
        if cf == 0:
            axs[cin,cf].set_ylabel("Avg charge [ADCu]")
        if cin==0:
            axs[cin,cf].set_title(r'$\tau_{fall}$=' + capaToTime(cf,2),fontsize = 10)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cmap = plt.get_cmap('plasma', 256)
norm = mpl.colors.Normalize(vmin=0,vmax=256)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm,cax=cbar_ax,label='Hold Delay')

fig, axs = plt.subplots(4, 4, figsize=(10, 10),sharex=True,sharey=True)
# Adjust the spacing between subplots to zero
plt.subplots_adjust(wspace=0, hspace=0)
for cin in range(4):
    for cf in range(4):
        ExtrapInputDynamicRange = np.zeros((2,256))
        MeasuredInputDynamicRange = np.zeros((2,256))
        for h in range (256):
            # interpolate to find the dynamic range for every hold
            y = np.asarray(VppInData_mean)*1e3
            x = VoltageCinCfHold_tensor[:, cin, cf, h]
            x_prompt = np.arange(1024) # adc range

            y_interp = linear_extrap(x_prompt, x, y, False) #input pulse range
            ExtrapInputDynamicRange[1,h] = y_interp[-1]
            ExtrapInputDynamicRange[0, h] = y_interp[0]
            MeasuredInputDynamicRange[1,h] = y[-1]
            MeasuredInputDynamicRange[0, h] = y[0]
        # find also the dynamic range at the peak
        print("t_rise = " + capaToTime(cin, 1) + ", t_fall = " + capaToTime(cf, 2) + " : Dyn Range = " + str(
            ExtrapInputDynamicRange[1,peak_pos_ind[:,cin,cf]].mean()) + " mVpp, Resolution: " + str(1024/ExtrapInputDynamicRange[1,peak_pos_ind[:,cin,cf]].mean()) + " ADCu/mV")

        axs[cin, cf].plot(np.arange(256) * 0.34, ExtrapInputDynamicRange[1,:],label='Max extrapolated')
        axs[cin, cf].plot(np.arange(256) * 0.34, ExtrapInputDynamicRange[0, :],label='Min extrapolated')
        axs[cin, cf].plot(np.arange(256) * 0.34, MeasuredInputDynamicRange[1, :], label='Max measured')
        axs[cin, cf].plot(np.arange(256) * 0.34, MeasuredInputDynamicRange[0, :], label='Min measured')

        if cf== 3:
            ax2 = axs[cin,cf].twinx()
            ax2.set_ylabel(r' $\tau_{rise}$=' + capaToTime(cin,1))
            ax2.set_yticklabels([])
        if cin==3:
            axs[cin,cf].set_xlabel("Hold Delay [ns]" )
        if cf == 0:
            axs[cin,cf].set_ylabel("DR [mVpp]")
        if cin==0:
            axs[cin,cf].set_title(r'$\tau_{fall}$=' + capaToTime(cf,2),fontsize = 10)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4)





# same figure of above but imshow
vpp = np.asarray(VppInData_mean)*1e3
hold = np.arange(256)*0.34
H, V = np.meshgrid(hold,vpp)
fig, axs = plt.subplots(4, 4, figsize=(10, 10),sharex=True,sharey=True)
for cin in range(4):
    for cf in range(4):
        axs[cin,cf].imshow(VoltageCinCfHold_tensor[:, cin, cf, :],cmap='plasma',vmin=0,vmax=1023,aspect='auto',extent=[0,255*0.34,vpp[-1],vpp[0]])
        if cf== 3:
            ax2 =axs[cin,cf].twinx()
            ax2.set_ylabel(r' $\tau_{rise}$=' + capaToTime(cin,1))
            ax2.set_yticklabels([])
        if cf==0:
            axs[cin,cf].set_ylabel("Input Pulse [mVpp]" )
        if cin == 3:
            axs[cin, cf].set_xlabel("Hold delay [ns]")
        if cin==0:
            axs[cin, cf].set_title(r'$\tau_{fall}$=' + capaToTime(cf,2),fontsize = 10)
fig.subplots_adjust(wspace=0, hspace=0,right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cmap = plt.get_cmap('plasma', 1024)
norm = mpl.colors.Normalize(vmin=0,vmax=1023)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm,cax=cbar_ax,label='AVG charge [ADCu]')

# estimate the gain
fig, axs = plt.subplots(4, 4, figsize=(10, 10),sharex=True,sharey=True)
# Adjust the spacing between subplots to zero
plt.subplots_adjust(wspace=0, hspace=0)
for cin in range(4):
    for cf in range(4):
        for h in range (256):
            gain = np.gradient( VoltageCinCfHold_tensor[:, cin, cf, h],np.asarray(VppInData_mean)*1e3)
            axs[cin, cf].plot(np.asarray(VppInData_mean)*1e3, gain,color=holdColors[h],alpha=0.1)
        if cf== 3:
            ax2 = axs[cin,cf].twinx()
            ax2.set_ylabel(r' $\tau_{rise}$=' + capaToTime(cin,1))
            ax2.set_yticklabels([])
        if cin==3:
            axs[cin,cf].set_xlabel("Input Pulse [mVpp]" )
        if cf == 0:
            axs[cin,cf].set_ylabel(r'Gain $\delta$ADCu/$\delta$Vpp [ADCu/mV]')
        if cin==0:
            axs[cin,cf].set_title(r'$\tau_{fall}$=' + capaToTime(cf,2),fontsize = 10)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cmap = plt.get_cmap('plasma', 256)
norm = mpl.colors.Normalize(vmin=0,vmax=256)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm,cax=cbar_ax,label='Hold Delay')





capaColors = distinctipy.get_colors(16, rng=1)
plt.figure()
for cin in range(4):
    for cf in range(4):
        plt.plot(np.asarray(VppInData_mean)*1e3,VoltageCinCfADC_peakPos[:,cin,cf],label=r'$\tau_{rise}$ =' + capaToTime(cin,1) + r' , $\tau_{fall}$ =' + capaToTime(cf,2),color = capaColors[cf + 4*cin])
plt.ylabel("Pulse peak position - best hold calibraiton  [ns]")
plt.xlabel("Input pulse voltage [mVpp]")
plt.legend()
plt.title("Best hold calibration as a function of input voltage and shaper settings")

"""

for v in range(Nexp):
    plt.figure()
    for cin in range(4):
        for cf in range(4):
            plt.plot(holdVector_ns, VoltageCinCfHold_tensor.max() - VoltageCinCfHold_tensor[v, cin, cf, :],
                 label=r'$\tau_{rise}$ =' + capaToTime(cin) + r' , $\tau_{fall}$ =' + capaToTime(cf),color = capaColors[cf + 4*cin])
    plt.xlabel("Hold delay [ns]")
    plt.ylabel("Avg charge [ADCu]")
    plt.title("Hold scan with Vpp " + f"{VppInData_mean[v]*1e3:.{2}f}" + " mV"  )
    plt.legend()
"""


# position of hold peak w.r.t the injected voltage



plt.show()