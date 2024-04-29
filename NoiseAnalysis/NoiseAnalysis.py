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
import cProfile
import time
import csv
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
from bs4 import BeautifulSoup
import re

# import noise level from surve
ScurveNoiseLevel = np.load("scurvePollutionAsicCh.npy")
ScurveSpanNoiseLevel = np.load("scurveNoiseSpanAsicCh.npy")
NoiseFactor = np.load("noiseFactorAsicCh.npy")

Pixel2Channel_map = np.zeros((2,64),dtype=int)
Pixel2Asic_map = np.zeros((2,64),dtype=int)
AsicChannel2PixelMap = np.zeros((4,32),dtype=int)
AsicChannel2TypeMap = np.zeros((4,32),dtype=int)

for type in range(2):
    for pixel in range(64):
        if type == 0: #slow
            if (pixel+1)%2==0 :#even
                Pixel2Asic_map[type, pixel] = 1
                Pixel2Channel_map[type, pixel] = pixel//2
            else: #odd
                Pixel2Asic_map[type, pixel] = 4
                Pixel2Channel_map[type, pixel] = pixel//2
        else: #fast
            if (pixel+1)%2==0 :#even
                Pixel2Asic_map[type, pixel] = 3
                Pixel2Channel_map[type, pixel] = pixel//2
            else: #odd
                Pixel2Asic_map[type, pixel] = 2
                Pixel2Channel_map[type, pixel] = pixel//2
        AsicChannel2PixelMap[Pixel2Asic_map[type, pixel]-1,Pixel2Channel_map[type, pixel]] = pixel+1
        AsicChannel2TypeMap[Pixel2Asic_map[type, pixel]-1,Pixel2Channel_map[type, pixel]] = type


#extract pin position from html talbe, using beautifulSoup4
with open("Pin_position_data.htm") as fp:
    soup = BeautifulSoup(fp, 'html.parser')
table = soup.find("table")
# The first tr contains the field names.
headings = [td.get_text() for td in table.find("tr").find_all("td")]
pins_datasets = [[] for i in range(len(headings))]
for row in table.find_all("tr")[1:]:
    # Find all data for each column
    columns = row.find_all('td')
    for i in range(len(headings)):
        pins_datasets[i].append(columns[i].contents[0])


# we are interested only in the y position of the pin, so
lenDatasets = len(pins_datasets[0])
refDesCol = headings.index("REFDES")
pinNameCol = headings.index("NET_NAME")
pinYcol = headings.index('PIN_Y')
asicChPinPosition = np.zeros((4,32))

for d in range(lenDatasets):
    asic = pins_datasets[refDesCol][d]
    if re.search("M[1234]",asic) != None:
        netName = pins_datasets[pinNameCol][d]
        if re.search("[S|F]<[0-9]+>",netName) != None:
            asicN = int(re.findall("[1234]",asic)[0])
            chN = Pixel2Channel_map[0,int(re.findall("[0-9]+",netName)[0])-1]
            asicChPinPosition[asicN-1,chN] = float(pins_datasets[pinYcol][d])
            print(asic + " " + netName + " : ASIC"+str(asicN)+' CH'+str(chN))

# extract the etch length by layer
with open("etch_lenght_by_layer.htm") as fp:
    soup = BeautifulSoup(fp, 'html.parser')
table = soup.find("table")
# The first tr contains the field names.
headings = [td.get_text() for td in table.find("tr").find_all("td")]
etch_length_dataset = [[] for i in range(len(headings))]
for row in table.find_all("tr")[1:]:
    # Find all data for each column
    columns = row.find_all('td')
    for i in range(len(headings)):
        etch_length_dataset[i].append(columns[i].contents[0])
Layers =np.asarray(["TOP","L_2","L_3","L_4","L_5","L_6","L_7","L_8","L_9","BOTTOM"])
asicChLayerLength = np.zeros((4,32,len(Layers)))
for d in range(len(etch_length_dataset[0])):
    netName = etch_length_dataset[0][d]
    if re.search("[S|F]<[0-9]+>", netName) != None:
        pixel = int(re.findall("[0-9]+", netName)[0])
        if 'F' in netName:
            asic_ind = Pixel2Asic_map[1, pixel - 1]
            channel_ind = Pixel2Channel_map[1, pixel - 1]
        elif 'S' in netName:
            asic_ind = Pixel2Asic_map[0, pixel - 1]
            channel_ind = Pixel2Channel_map[0, pixel - 1]
        layerName = etch_length_dataset[1][d]
        asicChLayerLength[asic_ind-1,channel_ind,Layers==layerName] = float(etch_length_dataset[2][d])
# load impedance data
Nvias = np.zeros((4,32))
maxImpedance = np.zeros((4,32))
minImpedance = np.zeros((4,32))
typImpedance = np.zeros((4,32))
maxImpedanceLengthPerc = np.zeros((4,32))
minImpedanceLengthPerc = np.zeros((4,32))
typImpedanceLengthPerc = np.zeros((4,32))
totalLength = np.zeros((4,32))
totalDelay = np.zeros((4,32))
totalResistance = np.zeros((4,32))
totalInductance = np.zeros((4,32))
totalCapacitance = np.zeros((4,32))

averageWeightedImpedance = np.zeros((4,32))
stdWeighetedImpedance = np.zeros((4,32))
with open("impedance_analysis.csv", 'r') as currentFile:
    ImpedanceData = pd.read_csv(currentFile,sep=',')
    net = ImpedanceData.values[:,0]
    for n in range(len(net)):
        pixel = int(re.findall("[0-9]+", net[n])[0])
        if 'F' in net[n]:
            asic_ind = Pixel2Asic_map[1, pixel - 1]
            channel_ind = Pixel2Channel_map[1, pixel - 1]
        elif 'S' in net[n]:
            asic_ind = Pixel2Asic_map[0, pixel - 1]
            channel_ind = Pixel2Channel_map[0, pixel - 1]

        Nvias[asic_ind-1,channel_ind] =  ImpedanceData.values[n,1]
        maxImpedance[asic_ind-1,channel_ind] =  ImpedanceData.values[n,3]
        minImpedance[asic_ind-1,channel_ind] =  ImpedanceData.values[n,4]
        typImpedance[asic_ind-1,channel_ind] =  ImpedanceData.values[n,5]
        maxImpedanceLengthPerc[asic_ind-1,channel_ind] =  ImpedanceData.values[n,6]
        minImpedanceLengthPerc[asic_ind-1,channel_ind] =  ImpedanceData.values[n,7]
        typImpedanceLengthPerc[asic_ind-1,channel_ind] =  ImpedanceData.values[n,8]
        totalLength[asic_ind-1,channel_ind] =  ImpedanceData.values[n,9]
        totalDelay[asic_ind-1,channel_ind] =  ImpedanceData.values[n,10]
        totalResistance[asic_ind-1,channel_ind] =  ImpedanceData.values[n,11]
        totalInductance[asic_ind-1,channel_ind] =  ImpedanceData.values[n,12]
        totalCapacitance[asic_ind-1,channel_ind] =  ImpedanceData.values[n,13]
        # build the impedance curve
        #zmin = np.asarray([minImpedance[asic_ind-1,channel_ind],minImpedance[asic_ind-1,channel_ind]])
        #zmin_x = np.asarray([0,minImpedanceLengthPerc[asic_ind-1,channel_ind]/100*totalLength[asic_ind-1,channel_ind]])

        #zmax = np.asarray([maxImpedance[asic_ind-1,channel_ind],maxImpedance[asic_ind-1,channel_ind]])
        #zmax_x = np.asarray([(1-maxImpedanceLengthPerc[asic_ind-1,channel_ind]/100)*totalLength[asic_ind-1,channel_ind], totalLength[asic_ind-1,channel_ind]])

        #z_x_m = (zmin_x[1] + zmax_x[0])/2
        #ztyp =  np.asarray([typImpedance[asic_ind-1,channel_ind],typImpedance[asic_ind-1,channel_ind]])
        #ztyp_x = z_x_m + np.asarray([-typImpedanceLengthPerc[asic_ind-1,channel_ind]/200,typImpedanceLengthPerc[asic_ind-1,channel_ind]/200])*totalLength[asic_ind-1,channel_ind]

        Zmin = minImpedance[asic_ind-1,channel_ind]
        Zmax = maxImpedance[asic_ind-1,channel_ind]
        Ztyp = typImpedance[asic_ind-1,channel_ind]
        wZmin = minImpedanceLengthPerc[asic_ind-1,channel_ind]/100*totalLength[asic_ind-1,channel_ind]
        wZmax = maxImpedanceLengthPerc[asic_ind-1,channel_ind]/100*totalLength[asic_ind-1,channel_ind]
        wZtyp = typImpedanceLengthPerc[asic_ind-1,channel_ind]/100*totalLength[asic_ind-1,channel_ind]
        Zwavg = (Zmin*wZmin + Zmax*wZmax + Ztyp*wZtyp)/(wZmin + wZmax + wZtyp)
        ZwSTD =np.sqrt((((Zmin-Zwavg)**2)*wZmin + ((Zmax-Zwavg)**2)*wZmax + ((Ztyp-Zwavg)**2)*wZtyp)/(wZmin + wZmax + wZtyp))

        averageWeightedImpedance[asic_ind-1,channel_ind] = Zwavg
        stdWeighetedImpedance[asic_ind-1,channel_ind] = ZwSTD

maxAggressorAsic = np.zeros((4,32))
maxAggressorChannel = np.zeros((4,32))

maxCouplingCoeff = np.zeros((4,32))
maxCouplingCoeffLength = np.zeros((4,32))
lengthWithCouplingCoeffMoreThan5 = np.zeros((4,32))
lengthWithCouplingCoeffbetween2and5 = np.zeros((4,32))

with open("Coupling_analysis_results.csv",'r') as currentFile:
    couplingData = pd.read_csv(currentFile,sep=',')
    net = couplingData.values[:, 0]
    for n in range(len(net)):
        pixel = int(re.findall("[0-9]+", net[n])[0])
        if 'F' in net[n]:
            asic_ind = Pixel2Asic_map[1, pixel - 1]
            channel_ind = Pixel2Channel_map[1, pixel - 1]
        elif 'S' in net[n]:
            asic_ind = Pixel2Asic_map[0, pixel - 1]
            channel_ind = Pixel2Channel_map[0, pixel - 1]
        aggNet = couplingData.values[n, 1]
        aggpixel = int(re.findall("[0-9]+", aggNet)[0])
        if 'F' in aggNet:
            asic_agg_ind = Pixel2Asic_map[1, aggpixel - 1]
            channel_agg_ind = Pixel2Channel_map[1, aggpixel - 1]
        elif 'S' in aggNet:
            asic_agg_ind = Pixel2Asic_map[0, aggpixel - 1]
            channel_agg_ind = Pixel2Channel_map[0, aggpixel - 1]
        maxAggressorAsic[asic_ind-1,channel_ind] = asic_agg_ind
        maxAggressorChannel[asic_ind-1,channel_ind] = channel_agg_ind
        maxCouplingCoeff[asic_ind-1,channel_ind] = couplingData.values[n, 2]
        maxCouplingCoeffLength[asic_ind - 1, channel_ind] = couplingData.values[n, 3]
        lengthWithCouplingCoeffMoreThan5[asic_ind - 1, channel_ind] = couplingData.values[n, 4]
        lengthWithCouplingCoeffbetween2and5[asic_ind - 1, channel_ind] = couplingData.values[n, 5]



asicChTotalLenght = asicChLayerLength.sum(2,keepdims=True)
asicChetchDistribution = asicChLayerLength/ asicChTotalLenght


layerColor = distinctipy.get_colors(len(Layers),rng=1)

def netToIndex(s):
    pixel = int(re.findall("[0-9]+", s)[0])
    if 'F' in s:
        asicind = Pixel2Asic_map[1, pixel - 1]
        channelind = Pixel2Channel_map[1, pixel - 1]
    elif 'S' in s:
        asicind = Pixel2Asic_map[0, pixel - 1]
        channelind = Pixel2Channel_map[0, pixel - 1]
    index = (asicind-1)*32 + (channelind)
    return index

def indexToNet(a):
    nets = [[] for n in range(len(a))]
    for a_ind in range(len(a)):
        n = a[a_ind]
        asicInd = n//32 +1
        channelInd = n%32
        pixel = AsicChannel2PixelMap[asicInd-1,channelInd]
        type = AsicChannel2TypeMap[asicInd-1,channelInd]
        if type == 0:
            nets[a_ind]= 'S<' + str(pixel) + '>'
        else:
            nets[a_ind]= 'F<' + str(pixel) + '>'
    return nets

CHindex = ["CH"+str(i) for i in range(32)]
asicOrder = [3,0,1,2]
#analyze detailed coupling
CouplingMatrix = np.zeros((128,128))
CouplingMatrixLengthOnly = np.zeros((128,128))
CouplingMatrixPercOnly = np.zeros((128,128))
TotalLength = np.zeros(128)
average_time = 0
with open("DAQTEMP_detailed_coupling.csv",'r') as currentFile:
    #detailedCouplingData = pd.read_csv(currentFile, sep=',', engine="pyarrow")
    #detailedCouplingData = csv.DictReader(currentFile)
    detailedCouplingData = csv.reader(currentFile)
    n = 0
    #for n in range(len(detailedCouplingData)):
    for currRow in detailedCouplingData:
        if n!=0:
            #currRow = detailedCouplingData.values[n,:] # 55ms
            if len(currRow[0])>0: #0.11ms
                lastVictimNet = currRow[0]
                victimIndex = netToIndex(lastVictimNet)

            if len(currRow)>4: #0.24ms
                lastCouplingLength = float(currRow[4])


            if currRow[2] == 'Total':# 0.47 ms
                TotalLength[victimIndex] = lastCouplingLength
            else:
                if not('-' in currRow[2]): # we have aggressors
                    currAggressorNet = currRow[2][-5:]
                    if (currAggressorNet[0] != 'F') and (currAggressorNet[0] != 'S'):
                        currAggressorNet = currAggressorNet[1:]
                    aggressorIndex = netToIndex(currAggressorNet)
                    currCouplingCoeff = (currRow[3])
                    if currCouplingCoeff[0] =='<':
                        currCouplingCoeff = currCouplingCoeff[1:]
                    currCouplingCoeff = float(currCouplingCoeff)
                    if '<' in currRow[3]:
                        currCouplingCoeff = currCouplingCoeff*0.9
                    CouplingMatrix[victimIndex,aggressorIndex] += currCouplingCoeff/100*lastCouplingLength
                    CouplingMatrixLengthOnly[victimIndex, aggressorIndex] += lastCouplingLength
                    CouplingMatrixPercOnly[victimIndex, aggressorIndex] += currCouplingCoeff / 100
        n+=1

HT_coupling = np.zeros(128)
with open("HT_SIPM_coupling.csv",'r') as currentFile:
    HTCouplingData = csv.reader(currentFile)
    firstRow = True
    for currRow in HTCouplingData:
        if not firstRow:
            # currRow = detailedCouplingData.values[n,:] # 55ms
            if len(currRow[0]) > 0:  # 0.11ms
                lastVictimNet = currRow[0]

            if len(currRow) > 4:  # 0.24ms
                lastCouplingLength = float(currRow[4])

            if currRow[2] == 'Total':  # 0.47 ms
                TotalLengthHT = lastCouplingLength
            else:
                if not ('-' in currRow[2]):  # we have aggressors
                    currAggressorNet = currRow[2][-5:]
                    if (currAggressorNet[0] != 'F') and (currAggressorNet[0] != 'S'):
                        currAggressorNet = currAggressorNet[1:]
                    if not 'GND' in currAggressorNet:
                        aggressorIndex = netToIndex(currAggressorNet)
                        currCouplingCoeff = (currRow[3])
                        if currCouplingCoeff[0] == '<':
                            currCouplingCoeff = currCouplingCoeff[1:]
                        currCouplingCoeff = float(currCouplingCoeff)
                        if '<' in currRow[3]:
                            currCouplingCoeff = currCouplingCoeff * 0.9
                        HT_coupling[aggressorIndex] += currCouplingCoeff / 100 * lastCouplingLength
        else:
            firstRow = False

HT_coupling_sorted_index = np.flip(np.argsort(HT_coupling))
HT_interesting_coupling_mask = HT_coupling[HT_coupling_sorted_index] > 0
NcoupledChannelsHT = sum(HT_interesting_coupling_mask)
plt.figure()
plt.bar(range(NcoupledChannelsHT),HT_coupling[HT_coupling_sorted_index[HT_interesting_coupling_mask]])
plt.xticks(range(NcoupledChannelsHT),labels=indexToNet(HT_coupling_sorted_index[HT_interesting_coupling_mask]))
plt.ylabel("Weighted coupling length [mm] with HT_SIPM")


# analyzed detailed impedance
TraceSegmentedImpedance = [[] for i in range(128)]
TraceSegmentedLength = [[] for i in range(128)]
TraceSegmentedDelay = [[] for i in range(128)]
TraceSegmentedLayer = [[] for i in range(128)]
TraceSegmentedResistance = [[] for i in range(128)]
TraceSegmentedInductance = [[] for i in range(128)]
TraceSegmentedCapacitance = [[] for i in range(128)]

min_impedance = float('+inf')
max_impedance = float('-inf')
min_length = float('+inf')
max_length = float('-inf')
min_delay = float('+inf')
max_delay =float('-inf')
min_resistance = float('+inf')
max_resistance =float('-inf')
min_inductance = float('+inf')
max_inductance = float('-inf')
min_capacitance = float('+inf')
max_capacitance = float('-inf')

with open("DAQTEMP_detailed_impedance.csv",'r') as currentFile:
    #detailedCouplingData = pd.read_csv(currentFile, sep=',', engine="pyarrow")
    #detailedCouplingData = csv.DictReader(currentFile)
    detailedImpedanceData = csv.reader(currentFile)
    firstRow = True
    #for n in range(len(detailedCouplingData)):
    for currRow in detailedImpedanceData:
        if not firstRow:
            if len(currRow)==1:
                currNet = currRow[0]
            else:
                if currRow[1]!='Total':
                    segmentImpedance = float(currRow[2])
                    min_impedance = min(segmentImpedance,min_impedance)
                    max_impedance = max(segmentImpedance, max_impedance)
                    segmentLength = float(currRow[3])
                    segmentDelay = float(currRow[4])
                    segmentLayer = currRow[6][currRow[6].find('$')+1:]
                    segmentResistance = float(currRow[7])
                    segmentInductance = float(currRow[8])
                    segmentCapacitance = float(currRow[9])
                    TraceSegmentedImpedance[netToIndex(currNet)].append(segmentImpedance)
                    TraceSegmentedLength[netToIndex(currNet)].append(segmentLength)
                    TraceSegmentedDelay[netToIndex(currNet)].append(segmentDelay)
                    TraceSegmentedLayer[netToIndex(currNet)].append(segmentLayer)
                    TraceSegmentedResistance[netToIndex(currNet)].append(segmentResistance)
                    TraceSegmentedInductance[netToIndex(currNet)].append(segmentInductance)
                    TraceSegmentedCapacitance[netToIndex(currNet)].append(segmentCapacitance)

        else:
            firstRow = False
def centersFromEdges(array):
    return (array[1:] + array[:-1])/2


chColors = distinctipy.get_colors(32,rng=1)
fig, axs = plt.subplots(nrows=4)
for a in range(4):
    for c in range(32):
        axs[asicOrder[a]].plot(centersFromEdges(np.cumsum(np.asarray([0]+TraceSegmentedLength[a*32 + c]))),TraceSegmentedImpedance[a*32 + c],color=chColors[c],label='CH'+str(c))
    axs[asicOrder[a]].set_xlabel('trace length [mm]')
    axs[asicOrder[a]].set_ylabel('trace impedance [Ohm]')
    axs[asicOrder[a]].set_title('ASIC'+str(a+1))

fig, axs = plt.subplots(ncols=4,nrows=4,sharex=True,sharey=True)
for a_vic in range(4):
    for a_agg in range(4):
        im = axs[asicOrder[a_vic],asicOrder[a_agg]].imshow(CouplingMatrix[(a_vic*32+0):(a_vic*32+32),(a_agg*32+0):(a_agg*32+32)],origin='upper', extent=[0,32,32,0], aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=0.001,vmin=CouplingMatrix.min(), vmax=CouplingMatrix.max()))
        if asicOrder[a_agg] == 0:
            axs[asicOrder[a_vic],asicOrder[a_agg]].set_ylabel("ASIC"+str(a_vic+1)+ " ch")
        if asicOrder[a_vic] == 3:
            axs[asicOrder[a_vic], asicOrder[a_agg]].set_xlabel("ASIC" + str(a_agg + 1) + " ch")
        axs[asicOrder[a_vic],asicOrder[a_agg]].set_xticks(np.arange(32)+0.5,labels=np.arange(32),fontsize=7,rotation=90)
        axs[asicOrder[a_vic], asicOrder[a_agg]].set_yticks(np.arange(32)+0.5,labels=np.arange(32),fontsize=7)
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0, hspace=0)
cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax, label='Weighted coupling lenght [mm]')
fig.delaxes(axs[0,1])
fig.delaxes(axs[0,2])
fig.delaxes(axs[0,3])
fig.delaxes(axs[1,2])
fig.delaxes(axs[1,3])
fig.delaxes(axs[2,3])

fig, axs = plt.subplots(ncols=4,nrows=1,sharex=False,sharey=True)
for a in range(4):
    totalCouplingLevel = CouplingMatrix[:,(a*32+0):(a*32+32)].sum(0)
    if AsicChannel2TypeMap[a, 0] == 0: #slow
        pixelIndex = ["S<"+str(i)+">" for i in AsicChannel2PixelMap[a, :]]
    elif AsicChannel2TypeMap[a, 0] == 1: #fast
        pixelIndex = ["F<" + str(i) + ">" for i in AsicChannel2PixelMap[a, :]]
    axs[asicOrder[a]].bar([i + " " + j for i, j in zip(CHindex, pixelIndex)], totalCouplingLevel, 0.8)
    axs[asicOrder[a]].set_title('ASIC' + str(a + 1))
    axs[asicOrder[a]].set_xticks(range(32), [i + " " + j for i, j in zip(CHindex, pixelIndex)], rotation='vertical',fontsize=8)
axs[0].set_ylabel('Total Coupling Level')
plt.subplots_adjust(wspace=0, hspace=0)



fig, axs = plt.subplots(ncols=4,nrows=1,sharex=False,sharey=True)
for a in range(4):
    noiseLevel = ScurveNoiseLevel[a,:]
    barWeights = {'TOP' : asicChetchDistribution[a,:,0]*noiseLevel,
                       'L_2': asicChetchDistribution[a,:,1]*noiseLevel,
                       'L_3': asicChetchDistribution[a, :, 2] * noiseLevel,
                       'L_4': asicChetchDistribution[a, :, 3] * noiseLevel,
                       'L_5': asicChetchDistribution[a, :, 4] * noiseLevel,
                       'L_6': asicChetchDistribution[a, :, 5] * noiseLevel,
                       'L_7': asicChetchDistribution[a, :, 6] * noiseLevel,
                       'L_8': asicChetchDistribution[a, :, 7] * noiseLevel,
                       'L_9': asicChetchDistribution[a, :, 8] * noiseLevel,
                       'BOTTOM': asicChetchDistribution[a, :, 9] * noiseLevel}
    bottom = np.zeros(32)
    if AsicChannel2TypeMap[a, 0] == 0: #slow
        pixelIndex = ["S<"+str(i)+">" for i in AsicChannel2PixelMap[a, :]]
    elif AsicChannel2TypeMap[a, 0] == 1: #fast
        pixelIndex = ["F<" + str(i) + ">" for i in AsicChannel2PixelMap[a, :]]
    for boolean, barWeight in barWeights.items():
        axs[asicOrder[a]].bar([i+" "+j for i,j in zip(CHindex,pixelIndex)],barWeight,0.8,label=boolean, bottom=bottom)
        bottom += barWeight
    axs[asicOrder[a]].set_title('ASIC'+str(a+1))
    axs[asicOrder[a]].set_xticks(range(32),[i+" "+j for i,j in zip(CHindex,pixelIndex)],rotation='vertical',fontsize=8)
axs[0].set_ylabel('Noise Level')
fig.legend(Layers,ncol=len(Layers),loc='lower center')
plt.subplots_adjust(wspace=0, hspace=0)



CHindex = ["CH"+str(i) for i in range(32)]
asicOrder = [3,0,1,2]
fig, axs = plt.subplots(ncols=4,nrows=1,sharex=False,sharey=True)
for a in range(4):
    noiseLevel = ScurveSpanNoiseLevel[a,:]
    barWeights = {'TOP' : asicChetchDistribution[a,:,0]*noiseLevel,
                       'L_2': asicChetchDistribution[a,:,1]*noiseLevel,
                       'L_3': asicChetchDistribution[a, :, 2] * noiseLevel,
                       'L_4': asicChetchDistribution[a, :, 3] * noiseLevel,
                       'L_5': asicChetchDistribution[a, :, 4] * noiseLevel,
                       'L_6': asicChetchDistribution[a, :, 5] * noiseLevel,
                       'L_7': asicChetchDistribution[a, :, 6] * noiseLevel,
                       'L_8': asicChetchDistribution[a, :, 7] * noiseLevel,
                       'L_9': asicChetchDistribution[a, :, 8] * noiseLevel,
                       'BOTTOM': asicChetchDistribution[a, :, 9] * noiseLevel}
    bottom = np.zeros(32)
    if AsicChannel2TypeMap[a, 0] == 0: #slow
        pixelIndex = ["S<"+str(i)+">" for i in AsicChannel2PixelMap[a, :]]
    elif AsicChannel2TypeMap[a, 0] == 1: #fast
        pixelIndex = ["F<" + str(i) + ">" for i in AsicChannel2PixelMap[a, :]]
    for boolean, barWeight in barWeights.items():
        axs[asicOrder[a]].bar([i+" "+j for i,j in zip(CHindex,pixelIndex)],barWeight,0.8,label=boolean, bottom=bottom)
        bottom += barWeight
    axs[asicOrder[a]].set_title('ASIC'+str(a+1))
    axs[asicOrder[a]].set_xticks(range(32),[i+" "+j for i,j in zip(CHindex,pixelIndex)],rotation='vertical',fontsize=8)
axs[0].set_ylabel('Noise Level Span [V]')
fig.legend(Layers,ncol=len(Layers),loc='lower center')
plt.subplots_adjust(wspace=0, hspace=0)

asicOrder = [3,0,1,2]
fig, axs = plt.subplots(ncols=4,nrows=1,sharex=False,sharey=True)
for a in range(4):
    noiseLevel = NoiseFactor[a,:]+1
    barWeights = {'TOP' : asicChetchDistribution[a,:,0]*noiseLevel,
                       'L_2': asicChetchDistribution[a,:,1]*noiseLevel,
                       'L_3': asicChetchDistribution[a, :, 2] * noiseLevel,
                       'L_4': asicChetchDistribution[a, :, 3] * noiseLevel,
                       'L_5': asicChetchDistribution[a, :, 4] * noiseLevel,
                       'L_6': asicChetchDistribution[a, :, 5] * noiseLevel,
                       'L_7': asicChetchDistribution[a, :, 6] * noiseLevel,
                       'L_8': asicChetchDistribution[a, :, 7] * noiseLevel,
                       'L_9': asicChetchDistribution[a, :, 8] * noiseLevel,
                       'BOTTOM': asicChetchDistribution[a, :, 9] * noiseLevel}
    bottom = np.zeros(32)
    if AsicChannel2TypeMap[a, 0] == 0: #slow
        pixelIndex = ["S<"+str(i)+">" for i in AsicChannel2PixelMap[a, :]]
    elif AsicChannel2TypeMap[a, 0] == 1: #fast
        pixelIndex = ["F<" + str(i) + ">" for i in AsicChannel2PixelMap[a, :]]
    for boolean, barWeight in barWeights.items():
        axs[asicOrder[a]].bar([i+" "+j for i,j in zip(CHindex,pixelIndex)],barWeight,0.8,label=boolean, bottom=bottom)
        bottom += barWeight
    axs[asicOrder[a]].set_title('ASIC'+str(a+1))
    axs[asicOrder[a]].set_xticks(range(32),[i+" "+j for i,j in zip(CHindex,pixelIndex)],rotation='vertical',fontsize=8)
axs[0].set_ylabel('Noise Factor [a.u.]')
fig.legend(Layers,ncol=len(Layers),loc='lower center')
plt.subplots_adjust(wspace=0, hspace=0)


fig, axs = plt.subplots(ncols=4,nrows=1,sharex=False,sharey=True)
for a in range(4):
    noiseLevel = NoiseFactor[a,:]+1
    if AsicChannel2TypeMap[a, 0] == 0: #slow
        pixelIndex = ["S<"+str(i)+">" for i in AsicChannel2PixelMap[a, :]]
    elif AsicChannel2TypeMap[a, 0] == 1: #fast
        pixelIndex = ["F<" + str(i) + ">" for i in AsicChannel2PixelMap[a, :]]

    bars = axs[asicOrder[a]].bar([i+" "+j for i,j in zip(CHindex,pixelIndex)],noiseLevel,0.8)
    lim = axs[asicOrder[a]].get_xlim() + axs[asicOrder[a]].get_ylim()
    ch=0
    for bar in bars:
        # build the colorface
        x_values = np.cumsum(np.asarray([0]+TraceSegmentedLength[a*32 + ch]))
        y_values =  np.asarray([TraceSegmentedImpedance[a*32 + ch][0]] + TraceSegmentedImpedance[a*32 + ch])
        f_interp = interp1d(x_values,y_values,kind='next')
        y_interp = f_interp(np.linspace(x_values.min(),x_values.max(),1000))
        # apply the colorface
        bar.set_zorder(1)
        bar.set_facecolor("none")
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        axs[asicOrder[a]].imshow(np.atleast_2d(y_interp).T, extent=[x, x + w, y, y + h],vmin= min_impedance,vmax = max_impedance,aspect="auto", zorder=0)
        ch+=1
    axs[asicOrder[a]].axis(lim)
    axs[asicOrder[a]].set_ylim([0,2])
    axs[asicOrder[a]].set_title('ASIC'+str(a+1))
    axs[asicOrder[a]].set_xticks(range(32),[i+" "+j for i,j in zip(CHindex,pixelIndex)],rotation='vertical',fontsize=8)

axs[0].set_ylabel('Noise Factor [a.u.]')
plt.subplots_adjust(wspace=0, hspace=0)
divider = make_axes_locatable(axs[-1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min_impedance, vmax=max_impedance), cmap='viridis'), cax=cax,label="Impedance [ohm]")








# show noise as function of z-center of position routage
depth_values_per_layer_mm = np.asarray([i*(0.03048+0.2032) for i in range(10)])
z_center_routage = np.sum(asicChetchDistribution*depth_values_per_layer_mm,axis=2)


fig = plt.figure()
ax = plt.axes(projection='3d')
for a in range(4):
    ax.scatter3D(asicChPinPosition[a,:],z_center_routage[a,:],NoiseFactor[a,:]+1,label="ASIC"+str(a+1))
ax.set_zlabel("Noise Factor")
ax.set_ylabel("Depth of routage w.r.t TOP layer [mm]")
ax.set_xlabel("Channel Y position [mm]")
ax.legend()








plt.show()

