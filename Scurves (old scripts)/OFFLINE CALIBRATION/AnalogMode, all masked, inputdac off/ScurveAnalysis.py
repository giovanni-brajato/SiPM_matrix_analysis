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
for file in fileList:
    m +=1

    with open(file, 'r') as currentFile:
        data = pd.read_csv(currentFile,sep=',')
        if FDList[m]=='D':

            currFolders = currentFile.name.split("\\")
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
            raw_dcount_mean =data.values[:, 3:].mean(1)
            raw_dcount_std =data.values[:, 3:].std(1)
            raw_v10dac =data.values[:, 0]
            avail_values_mask = np.concatenate((np.zeros(raw_v10dac[0]),np.ones(len(raw_v10dac)),np.zeros(1023-raw_v10dac[-1]))).astype(bool)
            unavail_values_mask = ~avail_values_mask
            final_dcount_mean = np.zeros(1024)
            final_dcount_std =np.zeros(1024)
            final_dcount_mean[avail_values_mask] = raw_dcount_mean
            final_dcount_mean[unavail_values_mask & (np.arange(1024) < raw_v10dac[0])] = 0
            final_dcount_mean[unavail_values_mask & (np.arange(1024) > raw_v10dac[-1])] = winsize
            final_dcount_std[avail_values_mask] = raw_dcount_std
            final_dcount_std[unavail_values_mask ] = 0
            final_v10dac = np.arange(1024)
            d_counts_mean_list.append(final_dcount_mean)
            d_counts_std_list.append(final_dcount_std)
            v10dac_list.append(final_v10dac)
            v10minval = 0
            v10maxval = 1023
            v6dac_list.append(data.values[:, 1])
            v6minval = np.min(np.append(data.values[:, 1],v6minval))
            v6maxval = np.max(np.append(data.values[:, 1], v6maxval))
            prob_est = np.gradient(np.maximum.accumulate(np.convolve(d_counts_mean_list[-1], np.ones(10) / 10, 'same')))
            prob_mean_list.append(prob_est / prob_est.sum())
            Asic_List.append(AsicList[m])
            ChList.append(chList[m])


Ncurves = len(d_counts_mean_list)
NAsics= len(np.unique(np.asarray(Asic_List)))
eta_vdac10 = np.ones((len(v10dac_list[0]), 32, NAsics))


vdac6 = np.zeros((32,NAsics))
#eta_prompt = np.arange(101)
eta_prompt = np.logspace(-5,2,1000)
vdac10_eta = np.zeros((len(eta_prompt), 32, NAsics))
# display noise heatmap
v10_range= np.arange(v10minval,v10maxval+1,1)
v6_range= np.arange(v6minval,v6maxval+1,1)
noise_density_heatmap = np.zeros((1024, 32,4))
trigg_eff_heatmap = np.zeros((len(v10_range), 32,4))
SiPM_slow_trigger_eff_heatmap = np.zeros((len(v10_range), 64))
SiPM_fast_trigger_eff_heatmap = np.zeros((len(v10_range), 64))
asicChannel2Pixel = np.stack((np.arange(0,64,2),(np.arange(1,64,2))),axis=1)+1
SiPM_slow_pollution_level = np.zeros(64)
SiPM_fast_pollution_level = np.zeros(64)


N = 32 # number of channels (or parameters to be optimized)
eta_obj = 1e-2
eta_obj_ind = np.argmin((eta_obj - eta_prompt)**2)
c10 = 0.92e-3
c6 = 1.5e-3
DC = 0.89

eff_cruves_per_asic = [[[] for a in range(4)] for c in range(32)]
v10dac_per_asic = [[[] for a in range(4)] for c in range(32)]
alpha = 1
asicChPollutionLevel = np.zeros((4,32))
asicChNoiseSpanLevel = np.zeros((4,32))
#pollution level factor to penalize more the region of the scurve where efficiency does not drop to zero
for m in range(Ncurves):
    noise_density_heatmap[v10dac_list[m], ChList[m],Asic_List[m]-1] = prob_mean_list[m]
    eff_curve =np.maximum.accumulate(d_counts_mean_list[m]/ d_counts_mean_list[m].max() * 100)
    eff_cruves_per_asic[ ChList[m]][Asic_List[m]-1] = eff_curve
    v10dac_per_asic[ ChList[m]][Asic_List[m]-1] = v10dac_list[m]
    trigg_eff_heatmap[v10dac_list[m], ChList[m],Asic_List[m]-1]=eff_curve
    eff_curve_noise = (eff_curve/100)
    noise_level = np.trapz(eff_curve_noise[eff_curve_noise<1]**(1/alpha))
    noise_span_level = ((eff_curve_noise<1 )* (eff_curve_noise>0)).sum()*c10
    #noise_level = np.sum((eff_curve != 0)*(eff_curve != 100))*c10
    #noise_level = np.trapz(eff_curve_noise)
    if Asic_List[m] == 1 or Asic_List[m] == 4:
        SiPM_slow_trigger_eff_heatmap[v10dac_list[m], asicChannel2Pixel[ChList[m],Asic_List[m]%4]-1]=eff_curve
        SiPM_slow_pollution_level[asicChannel2Pixel[ChList[m],Asic_List[m]%4]-1] = noise_level

    elif Asic_List[m] == 2 or Asic_List[m] == 3:
        SiPM_fast_trigger_eff_heatmap[v10dac_list[m], asicChannel2Pixel[ChList[m],Asic_List[m]%2]-1]=eff_curve
        SiPM_fast_pollution_level[asicChannel2Pixel[ChList[m], Asic_List[m] % 2] - 1] = noise_level
    asicChPollutionLevel[Asic_List[m]-1,ChList[m]] = noise_level
    asicChNoiseSpanLevel[Asic_List[m]-1,ChList[m]] = noise_span_level

    eta_vdac10[:, ChList[m], Asic_List[m]-1] = eff_curve
    vdac10_eta[:, ChList[m], Asic_List[m]-1] = linear_extrap(eta_prompt, eff_curve, v10dac_list[m],False)


#saving pollution file
np.save("scurvePollutionAsicCh.npy", asicChPollutionLevel)
np.save("scurveNoiseSpanAsicCh.npy", asicChNoiseSpanLevel)
# calculating maximum offset
max_offset = np.zeros((len(eta_prompt),NAsics))
max_theoretical_correction = 63*c6
for a in range(NAsics):
    for et in range(len(eta_prompt)):
        max_offset[et,a] = vdac10_eta[et,:,a].max() - vdac10_eta[et,:,a].min()


AsicColors = distinctipy.get_colors(5, rng=2)

plt.figure()
for a in range(NAsics):
    plt.plot(eta_prompt,max_offset[:,a],label='max scurve offset ASIC'+str(a+1),color=AsicColors[a])
plt.axhline(max_theoretical_correction/c10,label='Max theoretical v6 correction',color='r')
plt.xscale('log')
plt.xlabel('Trigger efficiency %')
plt.ylabel('Maximum offset 10bit DACu')
plt.legend()


# solving the system of equations to find the optimal 6 bit threshold

v6_sol = np.zeros((N,NAsics))
for a in range(NAsics):
    v10_eta = vdac10_eta[eta_obj_ind, :, a]
    for ch in range(32):
        v10_eta[ch] = linear_extrap(eta_obj, eff_cruves_per_asic[ch][a], v10dac_per_asic[ch][a],False)
    v6_eta = vdac6[:,a]
    
    # define the dispersion
    disp2 = lambda x: 1/N*np.sum((v10_eta*c10 + (x)*c6 - 1/N*np.sum(v10_eta*c10 + (x)*c6))**2)
    bnds = ([(0, 63) for i in range(N)]) # only positive

    #optimizing over reals
    res_opt1 = minimize(disp2, np.ones(N) * 31, method='Powell', bounds=bnds)
    
    # this optimization is very similar to the one obtained with the integer method
    print(str(eta_obj) + "% 6bit DAC calibration values - Powell scipy method rounded off to the nearest integer")
    for c in range(N):
        print("ASIC_" + str(a) + " sixbDAC_ch" + str(chList[c]) + "  " + str(int(np.round(res_opt1.x[c]))))
    
    # optimizing over integers
    """
    m = GEKKO() # create GEKKO model
    # create integer variables
    x = m.Array(m.Var,N,integer=True,lb=0,ub=63,value=1)
    m.Minimize(m.sum(1/N*((v10_eta*c10 + (x)*c6 - 1/N*m.sum(v10_eta*c10 + (x)*c6))**2)))
    m.options.SOLVER = 1 # APOPT solver
    m.solve()
    
    print(str(eta_obj) + "% 6bit DAC calibration values - Gekko integer optimization solver for python")
    for c in range(N):
        print("ASIC_" + str(a) + " sixbDAC_ch" + str(c) + "  " + str(int(x[c].value[0])))
   """
    #exact solution approach

    # trivial solution
    x_triv = v10_eta*c10/c6 - v6_eta
    x_triv = np.round(x_triv - x_triv.min())
    x_triv = x_triv.max() - x_triv
    x_triv[x_triv<0]=0
    x_triv[x_triv>63]=63

    print(str(eta_obj) + "% 6bit DAC calibration values - trivial solution")
    for c in range(N):
        print("ASIC_" + str(a+1) + " sixbDAC_ch" + str(c) + "  " + str(int(x_triv[c])))
    v6_sol[:,a] = x_triv


# joint calibration - bruteForce (need a faster method)
#calculate max shifting possible
v10glob_eta = vdac10_eta[eta_obj_ind, :, :]
max_positiveShift_v6 = 63 - v6_sol.max(0).astype(int)
max_negativeShift_v6 =v6_sol.min(0).astype(int)
Nshift_list = [[] for a in range(NAsics)]


for a in range(NAsics):
    Nshift_list[a] = np.arange(-max_negativeShift_v6[a],max_positiveShift_v6[a]+1,1)
disp2_matrix = np.zeros((max_negativeShift_v6 + max_positiveShift_v6 + 1))
for dv6_1 in range(len(Nshift_list[0])):
    for dv6_2 in range(len(Nshift_list[1])):
        for dv6_3 in range(len(Nshift_list[2])):
            for dv6_4 in range(len(Nshift_list[3])):
                vsol_temp = v6_sol + np.asarray([Nshift_list[0][dv6_1],Nshift_list[2][dv6_2],Nshift_list[2][dv6_3],Nshift_list[3][dv6_4]])
                vth_temp = vth = DC + v10glob_eta*c10 + vsol_temp*c6
                disp2_matrix[dv6_1,dv6_2,dv6_3,dv6_4] = ((vth_temp.reshape((N * NAsics, 1), order='F') - vth_temp.reshape((N * NAsics, 1), order='F').mean()) ** 2).mean()

# now find the minimum value of the dispersion
global_sol_ind = np.unravel_index(np.argmin(disp2_matrix, axis=None), disp2_matrix.shape)
global_sol_offset = np.asarray([Nshift_list[0][global_sol_ind[0]],Nshift_list[1][global_sol_ind[1]],Nshift_list[2][global_sol_ind[2]],Nshift_list[3][global_sol_ind[3]]])

global_sol = v6_sol + global_sol_offset

print(str(eta_obj) + "% 6bit DAC calibration values - trivial solution + global asic alignement with brute force")
for m in range(Ncurves):
    print("ASIC_" + str(Asic_List[m]) + " sixbDAC_ch" + str(ChList[m]) + "  " + str(int(global_sol[ChList[m],Asic_List[m]-1])))

# plot data before calibration


# generate N visually distinct colours
Channelcolors = distinctipy.get_colors(32, rng=1)

# Each channel has a distinct color
usabilityMatrix = np.zeros((4,32)).astype(bool)
vdacMarginMatrix = np.zeros((4,32))
minTriggerEffMatrix = np.zeros((4,32))
fig, axs = plt.subplots(NAsics,sharey=True,sharex=True)
for m in range(Ncurves):
    axs[Asic_List[m]-1].plot(v10dac_list[m], d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100, color=Channelcolors[ChList[m]],label="CH"+str(ChList[m]), linewidth=2)
    axs[Asic_List[m] - 1].set_ylabel('Trigger efficiency %')
    axs[Asic_List[m] - 1].set_title("ASIC "+ str(Asic_List[m]))
    firstTrigger = np.argmax(np.asarray(d_counts_mean_list[m])>0)
    if firstTrigger==0:
        usabilityMatrix[Asic_List[m]-1, ChList[m]] = False
    else:
        usabilityMatrix[Asic_List[m] - 1, ChList[m]] = True
    # find first trigger eff > 0
    firstNonZeroIndex = np.max([np.argwhere(d_counts_mean_list[m]>0)[0,0]-1,0])
    vdacMarginMatrix[Asic_List[m]-1, ChList[m]] = v10dac_list[m][firstNonZeroIndex]
    minTriggerEffMatrix[Asic_List[m]-1, ChList[m]] = np.asarray(d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100)[firstNonZeroIndex]

axs[-1].set_xlabel('Vth time 10 bit DAC units')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(),loc = 'lower center', ncol=8, labelspacing=0.)
fig.suptitle("Raw Data - " + rootFolderNameList[0])

plt.figure()
plt.imshow(usabilityMatrix,vmin=0,vmax=1, aspect='auto', origin='lower', extent=[0, 32 , 0, 4], interpolation='none')
plt.title("Channels with a vdac range without noisy triggers")
plt.xlabel('Channel')
plt.xticks(np.arange(32) + 0.5, labels=np.arange(32), rotation='vertical')
plt.ylabel('Asic')
plt.yticks(np.arange(4) + 0.5, labels=np.arange(1,5))

plt.figure()
plt.imshow(vdacMarginMatrix, aspect='auto', origin='lower', extent=[0, 32 , 0, 4], interpolation='none')
plt.title("Vdac range without triggers")
plt.xlabel('Channel')
plt.xticks(np.arange(32) + 0.5, labels=np.arange(32), rotation='vertical')
plt.ylabel('Asic')
plt.yticks(np.arange(4) + 0.5, labels=np.arange(1,5))
cbar = plt.colorbar()
cbar.set_label('vth 10bit [VDacU]', rotation=270)

plt.figure()
plt.imshow(minTriggerEffMatrix, aspect='auto', origin='lower', extent=[0, 32 , 0, 4], interpolation='none')
plt.title("Minimum achievable trigger efficiency")
plt.xlabel('Channel')
plt.xticks(np.arange(32) + 0.5, labels=np.arange(32), rotation='vertical')
plt.ylabel('Asic')
plt.yticks(np.arange(4) + 0.5, labels=np.arange(1,5))
cbar = plt.colorbar()
cbar.set_label('Min trigger efficiency %', rotation=270)

# quantifiy the noise in a different way
noiseFactor = minTriggerEffMatrix/minTriggerEffMatrix.max() - vdacMarginMatrix/vdacMarginMatrix.max()
plt.figure()
plt.imshow(noiseFactor, aspect='auto', origin='lower', extent=[0, 32 , 0, 4], interpolation='none')
plt.title("Noise factor")
plt.xlabel('Channel')
plt.xticks(np.arange(32) + 0.5, labels=np.arange(32), rotation='vertical')
plt.ylabel('Asic')
plt.yticks(np.arange(4) + 0.5, labels=np.arange(1,5))
cbar = plt.colorbar()
cbar.set_label('Noise Factor [a.u]', rotation=270)

np.save("noiseFactorAsicCh.npy", noiseFactor)

# gradient over colors
cmap_ch = matplotlib.colormaps["viridis"]
norm_ch = matplotlib.colors.Normalize(vmin=0, vmax=31)
sm_ch = matplotlib.cm.ScalarMappable(norm=norm_ch, cmap=cmap_ch)
sm_ch.set_array([])
fig, axs = plt.subplots(NAsics,sharey=True,sharex=True)
for m in range(Ncurves):
    axs[Asic_List[m]-1].plot(v10dac_list[m], d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100, color=cmap_ch(norm_ch(ChList[m])), label="CH" + str(ChList[m]), linewidth=2)
    axs[Asic_List[m] - 1].set_ylabel('Trigger efficiency %')
    axs[Asic_List[m] - 1].set_title("ASIC "+ str(Asic_List[m]))

# gradient over colors
cmap_ch = matplotlib.colormaps["viridis"]
norm_ch = matplotlib.colors.Normalize(vmin=0, vmax=31)
sm_ch = matplotlib.cm.ScalarMappable(norm=norm_ch, cmap=cmap_ch)
sm_ch.set_array([])
fig, axs = plt.subplots(NAsics,sharey=True,sharex=True)
for m in range(Ncurves):
    axs[Asic_List[m]-1].plot(v10dac_list[m], d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100, color=cmap_ch(norm_ch(ChList[m])), label="CH" + str(ChList[m]), linewidth=2)
    axs[Asic_List[m] - 1].set_ylabel('Trigger efficiency %')
    axs[Asic_List[m] - 1].set_title("ASIC "+ str(Asic_List[m]))
    axs[Asic_List[m] - 1].set_yscale('log')



fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm_ch, cax=cbar_ax, ticks=ChList, format=matplotlib.ticker.ScalarFormatter(), shrink=1.0, fraction=0.1, pad=0)
cbar.set_ticklabels(["CH" + str(c) for c in ChList])
axs[-1].set_xlabel('Vth time 10 bit DAC units')
fig.suptitle("Raw Data - " + rootFolderNameList[0])


fig, axs = plt.subplots(ncols=NAsics,sharey=True,sharex=True,gridspec_kw = {'wspace':0, 'hspace':0})
AsicShowOrder = [4,1,2,3]
for a in range(NAsics):
    im = axs[AsicShowOrder[a]-1].imshow(noise_density_heatmap[:,:,a],vmin=0,vmax=noise_density_heatmap.max(), aspect='auto', origin='lower', extent=[0, 32 , v10minval, v10maxval], interpolation='none')
    axs[AsicShowOrder[a]-1].set_title("ASIC "+ str(a+1))
    axs[AsicShowOrder[a]-1].set_xlabel('Channel')
    axs[AsicShowOrder[a]-1].set_xticks(np.arange(32) + 0.5, labels=np.arange(32),rotation='vertical')
axs[0].set_ylabel('Vth time 10 bit DAC units')
fig.colorbar(im, ax=axs.ravel().tolist(),label="Noise probability density", format=matplotlib.ticker.ScalarFormatter(),orientation='horizontal')
fig.suptitle("Estimated Noise density - " + rootFolderNameList[0])


fig, axs = plt.subplots(ncols=NAsics,sharey=True,sharex=True,gridspec_kw = {'wspace':0, 'hspace':0})
AsicShowOrder = [4,1,2,3]
for a in range(NAsics):
    im = axs[AsicShowOrder[a]-1].imshow(trigg_eff_heatmap[:,:,a],vmin=0,vmax=100, aspect='auto', origin='lower', extent=[0, 32 , v10minval, v10maxval], interpolation='none')
    axs[AsicShowOrder[a]-1].set_title("ASIC "+ str(a+1))
    axs[AsicShowOrder[a]-1].set_xlabel('Channel')
    axs[AsicShowOrder[a]-1].set_xticks(np.arange(32) + 0.5, labels=np.arange(32),rotation='vertical')
axs[0].set_ylabel('Vth time 10 bit DAC units')
fig.colorbar(im, ax=axs.ravel().tolist(),label="Trigger Efficiency %", format=matplotlib.ticker.ScalarFormatter(),orientation='horizontal')
fig.suptitle("Trigger efficiency - " + rootFolderNameList[0])

fig, axs = plt.subplots(ncols=NAsics,sharey=True,sharex=True,gridspec_kw = {'wspace':0, 'hspace':0})
AsicShowOrder = [4,1,2,3]
for a in range(NAsics):
    im = axs[AsicShowOrder[a]-1].imshow(trigg_eff_heatmap[:,:,a], aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1,
                                              vmin=0, vmax=100, base=10), origin='lower', extent=[0, 32 , v10minval, v10maxval], interpolation='none')
    axs[AsicShowOrder[a]-1].set_title("ASIC "+ str(a+1))
    axs[AsicShowOrder[a]-1].set_xlabel('Channel')
    axs[AsicShowOrder[a]-1].set_xticks(np.arange(32) + 0.5, labels=np.arange(32),rotation='vertical')
axs[0].set_ylabel('Vth time 10 bit DAC units')
#fig.colorbar(im, ax=axs.ravel().tolist(),label="Trigger Efficiency %", format=matplotlib.ticker.ScalarFormatter(),orientation='horizontal')
fig.suptitle("Trigger efficiency - " + rootFolderNameList[0])
plt.tight_layout()

fig, axs = plt.subplots(ncols=2,sharey=True,sharex=False,gridspec_kw = {'wspace':0, 'hspace':0})

im = axs[0].imshow(SiPM_fast_trigger_eff_heatmap, aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1,
                                          vmin=0, vmax=100, base=10), origin='lower', extent=[0, 64 , v10minval, v10maxval], interpolation='none')
axs[0].set_title("Fast pixels ")
axs[0].set_xticks(np.arange(64) + 0.5, labels=["F<"+str(p)+">" for p in range(1,65)],rotation='vertical')
axs[0].set_ylabel('Vth time 10 bit DAC units')
axs[1].imshow(SiPM_slow_trigger_eff_heatmap, aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1,
                                          vmin=0, vmax=100, base=10), origin='lower', extent=[0, 64 , v10minval, v10maxval], interpolation='none')
axs[1].set_title("Slow pixels ")
axs[1].set_xticks(np.arange(64) + 0.5, labels=["S<"+str(p)+">" for p in range(1,65)],rotation='vertical')
fig.colorbar(im, ax=axs.ravel().tolist(),label="Trigger Efficiency %", format=matplotlib.ticker.ScalarFormatter(),orientation='horizontal')
fig.suptitle("Trigger efficiency - " + rootFolderNameList[0])


fig, axs = plt.subplots(ncols=2,sharey=True,sharex=False,gridspec_kw = {'wspace':0, 'hspace':0})
axs[0].bar(np.arange(1,65),SiPM_fast_pollution_level)
axs[0].set_title("Fast pixels ")
axs[0].set_ylabel('Pollution level')
axs[0].set_xticks(np.arange(1,65), labels=["F<"+str(p)+">" for p in range(1,65)],rotation='vertical')
axs[1].bar(np.arange(1,65),SiPM_slow_pollution_level)
axs[1].set_title("Slow pixels ")
axs[1].set_xticks(np.arange(1,65), labels=["S<"+str(p)+">" for p in range(1,65)],rotation='vertical')
fig.suptitle("Trigger pollution level - " + rootFolderNameList[0])
#plt.tight_layout()

custom_colors = ['#0019ff','#0051ff','#00aeff','#00ffbb','#00ff00','#9dff00','#f7ff00','#ff7b00','#ff0000','#ff00e6']
cmap_noise = matplotlib.colors.LinearSegmentedColormap.from_list("custom_colormap", custom_colors , N=len(custom_colors))
#cmap_noise = matplotlib.colormaps["plasma"]
norm_noise = matplotlib.colors.Normalize(vmin=np.min(np.concatenate((SiPM_fast_pollution_level,SiPM_slow_pollution_level))), vmax=np.max(np.concatenate((SiPM_fast_pollution_level,SiPM_slow_pollution_level))))
sm_noise = matplotlib.cm.ScalarMappable(norm=norm_noise, cmap=cmap_noise)
sm_noise.set_array([])
fig, axs = plt.subplots(ncols=2,sharey=True,gridspec_kw = {'wspace':0, 'hspace':0})
for i in range(64):
    axs[0].bar(i+1,SiPM_fast_pollution_level[i],color=cmap_noise(norm_noise(SiPM_fast_pollution_level[i])))
axs[0].set_title("Fast pixels ")
axs[0].set_ylabel('Pollution level Vpp')
axs[0].set_xticks(np.arange(1,65), labels=["F<"+str(p)+">" for p in range(1,65)],rotation='vertical',fontsize=8)
for i in range(64):
    axs[1].bar(i+1,SiPM_slow_pollution_level[i],color=cmap_noise(norm_noise(SiPM_slow_pollution_level[i])))
axs[1].set_title("Slow pixels ")
axs[1].set_xticks(np.arange(1,65), labels=["S<"+str(p)+">" for p in range(1,65)],rotation='vertical',fontsize=8)
fig.suptitle("Trigger pollution level - " + rootFolderNameList[0])
cbar = fig.colorbar(sm_noise,ax=axs[1], label="Pollution level", format=matplotlib.ticker.ScalarFormatter(), shrink=1.0, fraction=0.1, pad=0)
#cbar = fig.colorbar(sm_noise,ax=axs[0], label="Noise level", format=matplotlib.ticker.ScalarFormatter(), shrink=1.0, fraction=0.1, pad=0)
# noise density heatmap - show on the specific asic order


# plot data after calibration
fig, axs = plt.subplots(NAsics,sharey=True,sharex=True)
for m in range(Ncurves):
    vth_post = DC + np.asarray(v10dac_list[m]) * c10 + v6_sol[ChList[m], Asic_List[m] - 1] * c6
    axs[Asic_List[m]-1].plot(vth_post / 1e-3, d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100, color=Channelcolors[ChList[m]], label="CH" + str(ChList[m]), linewidth=2)
    axs[Asic_List[m] - 1].set_ylabel('Trigger efficiency %')
    axs[Asic_List[m] - 1].set_yscale('log')
    axs[Asic_List[m] - 1].set_title("ASIC "+ str(Asic_List[m]))
axs[-1].set_xlabel('Vth time mV')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(),loc = 'lower center', ncol=8, labelspacing=0.)
fig.suptitle("Expected individual calibration results")


# plot data after calibration
fig, axs = plt.subplots(NAsics,sharey=True,sharex=True)
for m in range(Ncurves):
    vth_post = DC + np.asarray(v10dac_list[m]) * c10 + global_sol[ChList[m], Asic_List[m] - 1] * c6
    axs[Asic_List[m]-1].plot(vth_post / 1e-3, d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100, color=Channelcolors[ChList[m]], label="CH" + str(ChList[m]), linewidth=2)
    axs[Asic_List[m] - 1].set_ylabel('Trigger efficiency %')
    axs[Asic_List[m] - 1].set_title("ASIC "+ str(Asic_List[m]))
axs[-1].set_xlabel('Vth time mV')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(),loc = 'lower center', ncol=8, labelspacing=0.)
fig.suptitle("Expected global calibration results")

# globalChannelColors
GlobalChannelcolors = distinctipy.get_colors(128)
plt.figure()
for m in range(Ncurves):
    vth_post = DC + np.asarray(v10dac_list[m]) * c10 + global_sol[ChList[m], Asic_List[m] - 1] * c6
    if Asic_List[m]== 1 or 4:
        plt.plot(vth_post / 1e-3, d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100, color=GlobalChannelcolors[m], label="S<"+str((m%64)+1) +">", linewidth=2)
    else:
        plt.plot(vth_post / 1e-3, d_counts_mean_list[m] / d_counts_mean_list[m].max() * 100, color=GlobalChannelcolors[m], label="F<" + str((m % 64) + 1) + ">", linewidth=2)
plt.xlabel('Vth time mV')
plt.ylabel('Trigger efficiency %')
plt.title("ASIC " + str(Asic_List[m]))
plt.legend(loc = 'upper left', ncol=8, labelspacing=0.)
fig.suptitle("Expected global calibration results")




plt.show()