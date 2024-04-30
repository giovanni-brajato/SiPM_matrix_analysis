

from __future__ import print_function
from matplotlib.widgets import RectangleSelector, EllipseSelector, TextBox, Button
from typing import Tuple, Any
from sklearn.decomposition import FastICA, PCA
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from functools import partial
from matplotlib.gridspec import GridSpec
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from pylab import figure, cm
import csv
import glob, os
import numpy as np
import math
import imageio
from scipy import stats
from matplotlib.artist import Artist

# Library functions for prime
import sympy
from matplotlib.table import CustomCell
from matplotlib.widgets import TextBox
from matplotlib.colors import LogNorm

def Sgaussians(adc,*params):
    amp = params[0:(len(params) // 3)]
    mu = params[((len(params)//3)):(2 * (len(params)//3))]
    fwhm = params[(2*(len(params)//3)):(3 * (len(params)//3))]
    return (np.exp(-4*np.log(2)*((adc.reshape(-1, 1) - mu)/fwhm)**2)*amp).sum(1)

def find_intersection(x, y, y_target):
    # Find indices where the curve crosses the horizontal line
    indices = np.where(np.isclose(y, y_target))[0]

    # Extract x-values at the intersection indices
    intersection_points = x[indices]

    return intersection_points
vdac_valid = True
target_energy = 250

class EditableTable():
    def __init__(self, table):
        self.table = table
        self.ax = self.table.axes
        celld = table.get_celld()
        for key in celld.keys():
            if key[0] > 0 and key[1] > -1:
                cell = celld[key]
                cell.set_picker(True)
        self.canvas = self.table.get_figure().canvas
        self.cid = self.canvas.mpl_connect('pick_event', self.on_pick)
        self.tba = self.ax.figure.add_axes([0,0,.01,.01])
        self.tba.set_visible(False)
        self.tb = TextBox(self.tba, '', initial="")
        self.cid2 = self.tb.on_submit(self.on_submit)
        self.currentcell = celld[(1,0)]

    def on_pick(self, event):
        if isinstance(event.artist, CustomCell):
            # clear axes and delete textbox
            self.tba.clear()
            del self.tb
            # make textbox axes visible
            self.tba.set_visible(True)
            self.currentcell = event.artist
            # set position of textbox axes to the position of the current cell
            trans = self.ax.figure.transFigure.inverted()
            trans2 = self.ax.transAxes
            bbox = self.currentcell.get_bbox().transformed(trans2 + trans)
            self.tba.set_position(bbox.bounds)
            # create new Textbox with text of the current cell
            cell_text = self.currentcell.get_text().get_text()
            self.tb = TextBox(self.tba, '', initial=cell_text)
            self.cid2 = self.tb.on_submit(self.on_submit)

            self.canvas.draw()

    def on_submit(self, text):
        # write the text box' text back to the current cell
        self.currentcell.get_text().set_text(text)
        self.tba.set_visible(False)
        self.canvas.draw_idle()


import scipy.signal
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from numpy import matlib
import matplotlib
#matplotlib.use('QtAgg')
matplotlib.use('TkAgg')
from numpy import linalg as LA
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from sklearn.linear_model import LinearRegression
from distinctipy import distinctipy
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
from numpy import ndarray
import statsmodels.api as sm
from statsmodels.graphics import tsaplots

def linear_extrap(x,x_train,y_train):
    x = np.asarray(x).reshape(-1)
    lowerTrEffpointMask = np.asarray(x <= x_train.min()).reshape(-1)
    higherTrEffpointMask = np.asarray(x >= x_train.max()).reshape(-1)
    middleTrEffpointMask = ~lowerTrEffpointMask & ~higherTrEffpointMask

    Vdac = np.zeros(np.size(np.asarray(x)))
    if sum(lowerTrEffpointMask) == 1:
        lowerRampDacValue= y_train[len(x_train) - np.argmin(np.flip((x[lowerTrEffpointMask] - np.maximum.accumulate(x_train)) ** 2)) - 1]
        Vdac[lowerTrEffpointMask] = lowerRampDacValue
    else:
        try:
            lowerRampDacValue = np.polyfit(x_train[:3], y_train[:3], 1)
            Vdac[lowerTrEffpointMask] = lowerRampDacValue[0] * x[lowerTrEffpointMask] + lowerRampDacValue[1]
        except:
            Vdac[lowerTrEffpointMask] = -np.Inf

    if sum(higherTrEffpointMask) == 1:
        higherRampDacValue = y_train[np.argmin(((x[higherTrEffpointMask] - np.maximum.accumulate(x_train)) ** 2))]
        Vdac[higherTrEffpointMask] = higherRampDacValue
    else:
        try:
            higherRampDacValue = np.polyfit(x_train[-3:], y_train[-3:], 1)
            Vdac[higherTrEffpointMask] = higherRampDacValue[0] * x[higherTrEffpointMask] + higherRampDacValue[1]
        except:
            Vdac[higherTrEffpointMask] =+np.Inf
    Vdac[middleTrEffpointMask] = np.interp(x[middleTrEffpointMask], x_train, y_train)
    return Vdac


def findMiddle(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        return (input_list[int(middle)], input_list[int(middle-1)])



os.system('SETLOCAL EnableDelayedExpansion')

fileList = []
stringList = []
setNumberList = []
cwd = os.getcwd()
os.chdir(cwd)

Cpedestal0 =np.zeros(32)
Cpedestal1 =np.zeros(32)
STDpedestal0 = np.zeros(32)
STDpedestal1 = np.zeros(32)
pixelToChannelMap = np.reshape(np.floor(np.arange(0,32,0.5)).astype(int),(8,8))
pixelToAsicMap = np.matlib.repmat([0,1],8,4)
channelAsicToPixelMap = np.stack((np.arange(0,64,2),(np.arange(1,64,2))),axis=1)+1

def bestSubplotSize(N):
    if sympy.isprime(N):
        M = N
        done = False
        while(~done):
            M +=1
            if ~sympy.isprime(M):
                done = True
                break
    else:
        M=N
    divs = np.asarray(sympy.divisors(M))
    divs_f = np.flip(divs)
    num_el = np.ceil(len(divs)/2).astype(int)
    pairs = np.asarray([divs[:num_el],divs_f[:num_el]])
    ratio = pairs[0,:]/pairs[1,:]
    best_ratio = np.argmin(1-ratio)
    return (pairs[0,best_ratio],pairs[1,best_ratio])




def reshapeToSiPM(OldStructure):
    newStructure = np.zeros((len(OldStructure),8,8))
    for i in range(8):
        for j in range(8):
            newStructure[:,i,j] = OldStructure[:,pixelToAsicMap[i,j],pixelToChannelMap[i,j]]
    return newStructure

def param2val(param,val):
    if param == "Cin":
        match val:
            case 0: return "25ns"
            case 1: return "50ns"
            case 2: return "75ns"
            case 3: return "100ns"
    elif param == "Cf":
        match val:
            case 0:
                return "25ns"
            case 1:
                return "75ns"
            case 2:
                return "50ns"
            case 3:
                return "100ns"
    elif param == "Vpol [V]":
        return str(val)
    else:
        return str(val)
# define spatial mapping SiPM
SiPM_X_pos = np.matlib.repmat(1.58 + np.arange(8)*(3.16+0.2),8,1)
SiPM_Y_pos = np.rot90(SiPM_X_pos)

SiPM_XY_pos = np.stack((SiPM_X_pos,SiPM_Y_pos),axis=2)
SiPM_offset = np.array([26.68/2,26.68/2])
SiPM_XY_pos_rel = SiPM_XY_pos - SiPM_offset



distanceMeasures = False
currTestHasDistance = []
currTestDistance = []
currTestSource = []
Nexp = 0

for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith(".csv"):
            if file== 'ParamScanSet.csv':
                paramData = pd.read_csv(root+'\\'+file, sep=',')
                paramHeader =pd.read_csv(root+'\\'+file, sep=',', index_col=0, nrows=0).columns.tolist()
            else:
                fileList.append(os.path.join(root, file))
                source_start_index = file.find('_')
                source_end_index = file.find('.csv')
                Nexp += 1

X_interactions = [None for e in range(Nexp)]
Y_interactions = [None for e in range(Nexp)]
plt.close('all')
m = -1
board_indeces = np.zeros(Nexp,dtype=int)
board_indeces = board_indeces*3
for n in range(Nexp):
    labelString = ""
    for p in range(len(paramHeader)):
        labelString += paramHeader[p] + "=" + param2val(paramHeader[p],paramData.values[n,p+1]) + " "
    stringList.append(labelString)
raw_histogram_charge_per_channel = np.zeros((2,32,1024,Nexp))
corrected_histogram_charge_per_channel = np.zeros((2, 32, 1024, Nexp))
corrected_histogram_bin_centers_per_channel = np.zeros((2, 32, 1024, Nexp))
corrected_histogram_charge_per_channel_highlights =  np.zeros((2, 32, 1024, Nexp))


corrected_histogram_charge_half_sum  = np.zeros((2, 1024, Nexp))
corrected_histogram_bin_centers_half_sum = np.zeros((2, 1024, Nexp))
corrected_histogram_charge_full_sum  = np.zeros((1024, Nexp))
corrected_histogram_bin_centers_full_sum = np.zeros((1024, Nexp))

corrected_histogram_charge_half_sum  = [[None for j in range(2)] for e in range(Nexp)]
corrected_histogram_bin_centers_half_sum = [[None for j in range(2)] for e in range(Nexp)]
corrected_histogram_charge_full_sum  = [[] for e in range(Nexp)]
corrected_histogram_bin_centers_full_sum = [[] for e in range(Nexp)]
corrected_histogram_charge_full_avg  = [[] for e in range(Nexp)]
corrected_histogram_bin_centers_full_avg = [[] for e in range(Nexp)]
HeatmapAvgCSipm = np.zeros((8, 8, Nexp))

X_interactions = [None for e in range(Nexp)]
Y_interactions = [None for e in range(Nexp)]

corrected_histogram_charge_per_channel = np.zeros((2, 32, 1024, Nexp))
corrected_histogram_bin_centers_per_channel = np.zeros((2, 32, 1024, Nexp))
pixel_histogram_raw_peak_position = np.zeros((2, 32, Nexp))
pixel_histogram_improved_peak_position = np.zeros((2, 32, Nexp))
pixel_avg_deposed_energy = np.zeros((2, 32, Nexp))
avg_deposed_charge = np.zeros(Nexp)
autobins= 'auto'
Cpedestal = np.asarray([np.ones((4,32))*1023,
                       np.ones((4,32))*1023, # DT1 Missing
                       np.ones((4,32))*1023, # DT2 Missing
                       [[900.0975, 900.732, 901.741, 904.3745, 896.5595, 898.941,
                         899.17, 901.854, 898.603, 899.5885, 899.3215, 898.9815,
                         898.647, 898.8335, 898.1915, 903.004, 902.4635, 900.1575,
                         897.8815, 901.647, 898.4885, 900.124, 897.2955, 898.6735,
                         899.9965, 900.0225, 900.8535, 898.646, 898.948, 898.9665,
                         898.311, 900.604],
                        [935.6565, 933.4835, 934.5105, 936.812, 935.792, 935.058,
                         935.3065, 936.788, 934.4085, 934.876, 935.69, 932.8945,
                         933.371, 935.1505, 934.476, 934.3815, 937.738, 937.5135,
                         936.416, 935.189, 938.811, 935.944, 933.615, 937.0245,
                         936.158, 936.392, 935.5185, 934.5445, 935.5995, 937.558,
                         934.818, 938.656],
                        [907.232, 906.837, 905.18, 905.9475, 906.899, 905.943,
                         910.2205, 906.6175, 907.0505, 906.7745, 905.617, 906.22,
                         905.618, 907.9585, 906.614, 906.1015, 909.1985, 904.1915,
                         907.5495, 905.473, 906.7305, 906.13, 906.6355, 908.777,
                         907.707, 908.632, 908.0305, 909.4205, 906.577, 908.915,
                         905.554, 908.4405],
                        [912.0525, 915.9255, 912.4155, 914.6165, 911.837, 912.972,
                         914.565, 911.3935, 914.436, 912.2305, 912.7245, 912.1845,
                         913.71, 910.72, 914.3115, 912.15, 913.904, 911.9215,
                         913.945, 915.6435, 912.1515, 911.706, 914.1635, 913.4045,
                         913.2215, 911.8105, 914.2615, 910.0545, 916.294, 912.275,
                         914.0825, 912.866]],
                       [[916.9535, 916.4515, 916.757, 914.8835, 917.5335, 913.3265,
                         917.247, 915.98, 916.913, 913.47, 916.226, 917.645,
                         918.456, 915.2805, 914.0765, 917.936, 917.4625, 917.5325,
                         915.2595, 919.59, 916.284, 916.469, 916.3455, 916.349,
                         915.1955, 914.9615, 916.2055, 917.8025, 917.3725, 917.6765,
                         919.8055, 915.673],
                        [930.426, 929.3605, 928.8635, 928.5815, 931.4165, 934.8425,
                         926.562, 929.8245, 927.698, 927.729, 931.3935, 929.712,
                         929.934, 931.1315, 927.5925, 928.56, 930.3495, 928.409,
                         928.722, 928.68, 928.9415, 928.724, 931.3785, 930.2515,
                         929.0015, 930.463, 930.073, 927.817, 932.591, 929.882,
                         930.525, 931.556],
                        [954.027, 954.6415, 955.039, 953.2225, 953.4165, 953.171,
                         953.271, 953.745, 953.117, 953.139, 953.0585, 953.095,
                         952.802, 953.839, 953.911, 953.355, 953.049, 953.5775,
                         953.2045, 952.9865, 951.6795, 950.117, 953.546, 951.0855,
                         952.7935, 951.641, 951.8025, 956.1455, 956.2475, 951.291,
                         951.101, 952.11],
                        [949.4705, 950.927, 950.724, 950.946, 950.4255, 950.7085,
                         948.5625, 949.682, 947.593, 948.0865, 948.8015, 948.042,
                         951.1615, 949.076, 946.883, 949.7255, 954.4525, 949.9105,
                         950.6765, 949.875, 950.1455, 949.602, 947.679, 952.5295,
                         947.855, 951.727, 949.167, 949.626, 951.3445, 948.1985,
                         953.204, 950.162]]])
def getCoincidences(lists):
    N_sources = len(lists) # the number of boards/petirocs
    indeces = np.zeros(N_sources,dtype=int)
    list_lenghts =  np.zeros(N_sources,dtype=int)
    for l in range(N_sources): # get the list lenghts
        list_lenghts[l] = len(lists[l])
    coincidence_indeces = np.zeros((max(list_lenghts),N_sources),dtype=int)
    c = 0
    elements_to_be_compared = np.zeros(N_sources,dtype=int)
    # loop until one array reaches the end
    while np.prod(indeces < list_lenghts).astype(bool) : #as long as there are no indeces overpassing maximum list lenght
        # memorize elements in the tuple
        for l in range(N_sources):
            elements_to_be_compared[l] = lists[l][indeces[l]]
        # sort elements
        sorted_indeces = np.argsort(elements_to_be_compared)
        # checking which elements are equal to the maximum
        equal_elements_mask = elements_to_be_compared[sorted_indeces[-1]] == elements_to_be_compared
        if np.prod(equal_elements_mask): # are all equal memorize and advance all indeces
            coincidence_indeces[c,:] = indeces
            c += 1
            indeces += 1
        else: # some elements are not
            #advance index of the smallest elements
            indeces += (~ equal_elements_mask)
    return coincidence_indeces[0:c,:]

for file in fileList:
    m +=1
    print("Processing " + file)
    RTvector_list  = [[],[]]
    STvector_list  = [[],[]]
    CTmatrix_list = [[],[]]
    FTmatrix_list = [[], []]
    Cmatrix_list = [[], []]
    Hmatrix_list = [[], []]

    with open(file, 'r') as currentFile:
        setNumber = int(file.split('\\')[-1].split('_')[-1].split(".")[0])
        setNumberList.append(setNumber)
        data = csv.reader(currentFile)
        headerRow = True
        for currEvent in data:
            if headerRow:
                headerRow = False #skip header
            else:
                dataEvent = currEvent
                if dataEvent[0] == 'NaN':
                    print('No data frame found')
                    # skipping
                else:
                    ID=int(dataEvent[0])
                    if ID >= 0 and  ID <= 3:
                        RT = int(dataEvent[1])
                        C = [int(c) for c in dataEvent[2:]]
                        Cmatrix_list[ID].append(C)
                        RTvector_list[ID].append(RT)
                    else:
                        #skipping
                        print('Corrupted data frame')
                #sorting
        sorted_0 = np.argsort(RTvector_list[0])
        RTvector_list[0] = np.asarray(RTvector_list[0])[sorted_0]
        Cmatrix_list[0] = np.asarray(Cmatrix_list[0])[sorted_0,:]
        sorted_1 = np.argsort(RTvector_list[1])
        RTvector_list[1] = np.asarray(RTvector_list[1])[sorted_1]
        Cmatrix_list[1] = np.asarray(Cmatrix_list[1])[sorted_1, :]

        indexes = getCoincidences(RTvector_list)
        Nevents =len(indexes[:,0])
        Cmatrix = np.zeros((Nevents,2,32))


        Cmatrix[:, 0, :] = np.asarray(Cmatrix_list[0])[indexes[:,0],:]
        Cmatrix[:, 1, :] = np.asarray(Cmatrix_list[1])[indexes[:,1],:]


        # remap values equal to -1 to 1023
        Cmatrix[Cmatrix == 0] = 1024
        CorrectedCmatrix = Cpedestal[board_indeces[m],0:2,:] - Cmatrix
        CorrectedCmatrix = CorrectedCmatrix + 1
        HalfSumCorrectedCmatrix = CorrectedCmatrix.sum(2)
        FullSumCorrectedCmatrix =HalfSumCorrectedCmatrix.sum(1)
        FullAverageCorrectedCmatrix = (CorrectedCmatrix.mean(2)).mean(1)

        corrected_histogram_charge_full_sum[m],bin_edges = np.histogram(FullSumCorrectedCmatrix, bins = autobins)
        corrected_histogram_bin_centers_full_sum[m] = (bin_edges[1:] + bin_edges[:-1])/2

        corrected_histogram_charge_full_avg[m], bin_edges = np.histogram(FullAverageCorrectedCmatrix, bins=autobins)
        corrected_histogram_bin_centers_full_avg[m] = (bin_edges[1:] + bin_edges[:-1]) / 2

        for a in range(2):
            for ch in range(32):
                corrected_bins = np.sort(Cpedestal[board_indeces[m],a, ch] - (np.arange(1025) - 0.5))
                corrected_bins_centers = (corrected_bins[1:] + corrected_bins[:-1]) / 2
                corrected_histogram_bin_centers_per_channel[a, ch, :, m] = corrected_bins_centers
                corrected_histogram_charge_per_channel[a, ch, :, m], _ = np.histogram(CorrectedCmatrix[:, a, ch], bins=corrected_bins)
                pixel_histogram_raw_peak_position[a,ch,m] = corrected_bins_centers[np.argmax(corrected_histogram_charge_per_channel[a, ch, :, m])]
                # fit a gaussian on top
                xdata = corrected_bins_centers
                ydata = corrected_histogram_charge_per_channel[a, ch, :, m]
                ydata =    np.convolve(ydata, np.ones(12) / 12, mode='same')
                peak_height = max(ydata)
                peak_pos = xdata[np.argmax(ydata)]
                pixel_histogram_improved_peak_position[a, ch, m] = peak_pos
                pixel_avg_deposed_energy[a, ch, m] = CorrectedCmatrix[:,a, ch].mean()

        avg_deposed_charge[m] = CorrectedCmatrix.mean()
        # flood map calculations
        CorrectedCmatrix_positive = CorrectedCmatrix - np.min(CorrectedCmatrix)
        # matrix has to be reshaped
        SiPM_Cmatrix = reshapeToSiPM(CorrectedCmatrix_positive)
        X_interactions[m] = np.sum(np.multiply(SiPM_Cmatrix,SiPM_XY_pos_rel[:,:,0]),axis=(1,2))/np.sum(SiPM_Cmatrix,axis=(1,2))
        Y_interactions[m] = np.sum(np.multiply(SiPM_Cmatrix,SiPM_XY_pos_rel[:,:,1]),axis=(1,2))/np.sum(SiPM_Cmatrix,axis=(1,2))

"""
fig, axes = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True,gridspec_kw={'hspace': 0, 'wspace': 0})
fig.suptitle('Corrected charge histogram per channel')
for m in range(Nexp):
    for i in range(8):
        for j in range(8):
            ax = axes[i, j]
            ax.plot(corrected_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m], corrected_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m], label=stringList[m])
            ax.text(.99, .99, 'A' + str(pixelToAsicMap[i,j]) + 'C' + str(pixelToChannelMap[i,j]), ha='right', va='top', transform=ax.transAxes, fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='x', labelsize=8)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(), loc='lower center',ncol=Nexp)
"""
# show single histograms and average with shaded area
"""
for m in range(Nexp):
    plt.figure()
    average_binning = np.arange(0,1024)
    average_histogram = np.zeros((1024,64))
    n = 0
    for i in range(8):
        for j in range(8):
            plt.plot(corrected_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m],
                    corrected_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m],
                    color='b',alpha=0.1,label='Histogram of single pixel')
            average_histogram[:,n] = linear_extrap(average_binning,corrected_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m],corrected_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m])
            n +=1
    plt.plot(average_binning,  average_histogram.mean(1), color='g',
             label='Average of histograms')
    plt.fill_between(average_binning, (average_histogram.mean(1) - average_histogram.std(1)), (average_histogram.mean(1) + average_histogram.std(1)),color='g',alpha=0.3)
    plt.plot(corrected_histogram_bin_centers_full_avg[m],corrected_histogram_charge_full_avg[m], color='r',label='Histogram of average')
    plt.title(stringList[m])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel('Charge ADCu')
    plt.ylabel('Counts')
"""






# for every histogram, find the peak and calculate its resolution


mu_vector = np.zeros(Nexp)
sigma_vector = np.zeros(Nexp)
fwhm_vector = np.zeros(Nexp)
fwhm_perc_vector = np.zeros(Nexp)
label_list = []
for m in range(Nexp):
    xdata = corrected_histogram_bin_centers_full_sum[m]
    ydata = corrected_histogram_charge_full_sum[m]
    peak_height = max(ydata)
    peak_pos = xdata[np.argmax(ydata)]
    res_Sgauss_fit, _ = scipy.optimize.curve_fit(Sgaussians, xdata, ydata, p0=[peak_height, peak_pos, peak_pos * 0.01])
    fit_ydata = Sgaussians(xdata, *[res_Sgauss_fit[0], abs(res_Sgauss_fit[1]), res_Sgauss_fit[2]])
    """
    plt.figure()
    plt.plot(xdata,ydata,'.b')
    plt.plot(xdata, fit_ydata, '-r')
    plt.title('Fitting of '+ stringList[setNumberList[m]])
    """
    mu_vector[m] = res_Sgauss_fit[1]
    fwhm_vector[m] = res_Sgauss_fit[2]
    sigma_vector[m] = res_Sgauss_fit[2]/(2*np.sqrt(2*np.log(2)))
    fwhm_perc_vector[m] = res_Sgauss_fit[2]/res_Sgauss_fit[1]*100
    label_list.append(stringList[setNumberList[m]])
if vdac_valid:
    vdac_input_list = [int(l.split(' ')[0].split("=")[-1]) for l in label_list]
else:
    try:
        vdac_input_list = [int(l.split(' ')[3].split("=")[-1]) for l in label_list]
    except Exception as e:
        vdac_input_list = [int(l.split(' ')[-2].split("=")[-1]) for l in label_list]
pixel_color_list = np.asarray( distinctipy.get_colors(64,n_attempts=100,rng=1 ))
pixel_color_matrix = np.dstack((pixel_color_list[:,0].reshape((8,8)),pixel_color_list[:,1].reshape((8,8)),pixel_color_list[:,2].reshape((8,8))))
fig, ax = plt.subplots()
for m in range(Nexp):
    for i in range(8):
        for j in range(8):
            ax.plot(vdac_input_list, pixel_histogram_raw_peak_position[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :],
                    label='A' + str(pixelToAsicMap[i,j]) + 'C' + str(pixelToChannelMap[i,j]), color=pixel_color_matrix[i,j])
ax.set_xlabel(label_list[0].split(' ')[0].split("=")[0])
ax.set_ylabel("Pixel raw peak [ADCu]")
ax1 = inset_axes(ax, width='35%', height='35%', loc='upper right')
ax1.imshow(pixel_color_matrix)
ax1.set_xticks(np.arange(0, 8, 1))
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])
ax1.set_yticks(np.arange(0, 8, 1))
ax1.set_xticks(np.arange(-.5, 8, 1), minor=True)
ax1.set_yticks(np.arange(-.5, 8, 1), minor=True)
ax1.grid(which='minor', color='w', linestyle='-', linewidth=1)
ax1.tick_params(bottom=False, left=False)
ax1.tick_params(which='minor',bottom=False, left=False)


fig, ax = plt.subplots()
for m in range(Nexp):
    for i in range(8):
        for j in range(8):
            ax.plot(vdac_input_list, pixel_histogram_improved_peak_position[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :],
                    label='A' + str(pixelToAsicMap[i,j]) + 'C' + str(pixelToChannelMap[i,j]), color=pixel_color_matrix[i,j])
ax.set_xlabel(label_list[0].split(' ')[0].split("=")[0])
ax.set_ylabel("Pixel peak (improved) [ADCu]")
ax1 = inset_axes(ax, width='35%', height='35%', loc='upper right')
ax1.imshow(pixel_color_matrix)
ax1.set_xticks(np.arange(0, 8, 1))
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])
ax1.set_yticks(np.arange(0, 8, 1))
ax1.set_xticks(np.arange(-.5, 8, 1), minor=True)
ax1.set_yticks(np.arange(-.5, 8, 1), minor=True)
ax1.grid(which='minor', color='w', linestyle='-', linewidth=1)
ax1.tick_params(bottom=False, left=False)
ax1.tick_params(which='minor',bottom=False, left=False)

fig, ax = plt.subplots()
for m in range(Nexp):
    for i in range(8):
        for j in range(8):
            ax.plot(np.asarray(vdac_input_list)*3.896833333 + 15.474888889, pixel_avg_deposed_energy[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :],
                    label='A' + str(pixelToAsicMap[i,j]) + 'C' + str(pixelToChannelMap[i,j]), color=pixel_color_matrix[i,j])
ax.set_xlabel(label_list[0].split(' ')[3].split("=")[0] + " [ns]")
ax.set_ylabel("Average deposed energy per pixel [ADCu]")
ax1 = inset_axes(ax, width='35%', height='35%', loc='upper right')
ax1.imshow(pixel_color_matrix)
ax1.set_xticks(np.arange(0, 8, 1))
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])
ax1.set_yticks(np.arange(0, 8, 1))
ax1.set_xticks(np.arange(-.5, 8, 1), minor=True)
ax1.set_yticks(np.arange(-.5, 8, 1), minor=True)
ax1.grid(which='minor', color='w', linestyle='-', linewidth=1)
ax1.tick_params(bottom=False, left=False)
ax1.tick_params(which='minor',bottom=False, left=False)

plt.figure()
plt.plot(np.asarray(vdac_input_list)*8.5 + 2,avg_deposed_charge)
plt.ylabel("Averaged deposed energy on matrix [ADCu]")
plt.xlabel(label_list[0].split(' ')[3].split("=")[0] + " [ns]")



## recycle information on the averaged deposed energy per pixel to find the correct setting for the VinputDAC
if vdac_valid:
    N_points = 100;
    max_energy = pixel_avg_deposed_energy.max()
    min_energy = pixel_avg_deposed_energy.min()

    energy_vector = np.linspace(min_energy, max_energy, num=N_points)
    vdac_vector = np.asarray(vdac_input_list)
    inverse_vdac_energy_map = np.zeros((2, 32, N_points))
    vdac_calibration = np.zeros((2, 32))
    print('VinputDAC calibration data')
    for m in range(Nexp):
        for i in range(8):
            for j in range(8):
                avg_deposed_energy = pixel_avg_deposed_energy[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :]
                avg_deposed_energy_mask = (avg_deposed_energy > min_energy) * (avg_deposed_energy < max_energy)

                interpolation = scipy.interpolate.CubicSpline(vdac_vector,avg_deposed_energy)
                better_energy_curve = interpolation(np.arange(256))

                inverse_vdac_energy_map[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :] = linear_extrap(energy_vector, avg_deposed_energy[avg_deposed_energy_mask], vdac_vector[avg_deposed_energy_mask])
                vdac_calibration[pixelToAsicMap[i, j], pixelToChannelMap[i, j]] = linear_extrap(target_energy, avg_deposed_energy[avg_deposed_energy_mask], vdac_vector[avg_deposed_energy_mask])
                vdac_calibration[pixelToAsicMap[i, j], pixelToChannelMap[i, j]] = np.argmin((better_energy_curve - target_energy)**2)

                #inverse_vdac_energy_map[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :] = interpolation(energy_vector)
    for a in range(2):
        if a == 0:
            asicN = 4
        else:
            asicN = a
        for c in range(32):
            print("ASIC_" + str(asicN) + " inputDAC_ch" + str(c) + "  " + str(int(vdac_calibration[a, c])))


    fig, ax = plt.subplots()
    for m in range(Nexp):
        for i in range(8):
            for j in range(8):
                ax.plot(vdac_input_list, pixel_avg_deposed_energy[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :],
                        color=pixel_color_matrix[i, j])
                ax.plot(vdac_calibration[pixelToAsicMap[i, j], pixelToChannelMap[i, j]],target_energy,color=pixel_color_matrix[i, j],linestyle="None",marker='o')
    ax.plot(vdac_input_list, np.ones(len(vdac_input_list)) * target_energy, color='r', linestyle='--',
            label='Target Energy')
    ax.set_xlabel(label_list[0].split(' ')[0].split("=")[0])
    ax.set_ylabel("Average deposed energy per pixel [ADCu]")
    ax.legend(loc='lower right')
    ax1 = inset_axes(ax, width='35%', height='35%', loc='lower left')
    ax1.imshow(pixel_color_matrix)
    ax1.set_xticks(np.arange(0, 8, 1))
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.set_yticks(np.arange(0, 8, 1))
    ax1.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax1.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax1.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax1.tick_params(bottom=False, left=False)
    ax1.tick_params(which='minor', bottom=False, left=False)


fig, ax = plt.subplots()
ax.bar(label_list,mu_vector)
ax.set_ylabel('Peak Positon [sum ADCu]')
plt.setp(ax.get_xticklabels(), ha="right", rotation=45)

fig, ax = plt.subplots()
ax.bar(label_list,sigma_vector)
ax.set_ylabel('Peak width [sum ADCu]')
plt.setp(ax.get_xticklabels(), ha="right", rotation=45)

fig, ax = plt.subplots()
ax.bar(label_list,fwhm_perc_vector)
ax.set_ylabel('Peak FWHM%')
plt.setp(ax.get_xticklabels(), ha="right", rotation=45)


# Get the Viridis colormap
cmap = matplotlib.colormaps['viridis']
# generate N visually distinct colours
colors = distinctipy.get_colors(Nexp)


plt.figure()
plt.title('Corrected charge histogram, sum on full matrix')
for m in range(Nexp):
    plt.plot(corrected_histogram_bin_centers_full_sum[m],corrected_histogram_charge_full_sum[m]/np.max(corrected_histogram_charge_full_sum[m]), label=stringList[setNumberList[m]], color = cmap(m / (Nexp - 1)))
plt.legend()
plt.xlabel('Sum ADCu')
plt.ylabel('Normalized Counts')

plt.figure()
plt.title('Corrected charge histogram, sum on full matrix')
for m in range(Nexp):
    plt.plot(corrected_histogram_bin_centers_full_sum[m],corrected_histogram_charge_full_sum[m], label=stringList[setNumberList[m]], color = cmap(m / (Nexp - 1)))
plt.legend()
plt.xlabel('Sum ADCu')
plt.ylabel('Counts')

plt.figure()
plt.title('Corrected charge histogram, sum on full matrix')
for m in range(Nexp):
    plt.plot(corrected_histogram_bin_centers_full_sum[m],corrected_histogram_charge_full_sum[m]/np.max(corrected_histogram_charge_full_sum[m]), label=stringList[setNumberList[m]], color = colors[m])
plt.legend()
plt.xlabel('Sum ADCu')
plt.ylabel('Normalized Counts')

plt.figure()
plt.title('Corrected charge histogram, sum on full matrix')
for m in range(Nexp):
    plt.plot(corrected_histogram_bin_centers_full_sum[m],corrected_histogram_charge_full_sum[m], label=stringList[setNumberList[m]], color = colors[m])
plt.legend()
plt.xlabel('Sum ADCu')
plt.ylabel('Counts')


binsPerSide = 8*64
subplot_grid = bestSubplotSize(Nexp)
indeces = np.arange(subplot_grid[0]*subplot_grid[1]).reshape(subplot_grid)

fig, axes = plt.subplots(nrows=subplot_grid[0], ncols=subplot_grid[1], sharex=True, sharey=True)
fig.suptitle('Flood histogram')
for i in range(subplot_grid[0]):
    for j in range(subplot_grid[1]):
        m = indeces[i,j]
        if m <= Nexp-1:
            H, xedges, yedges = np.histogram2d(X_interactions[m], Y_interactions[m], bins=binsPerSide, range=[[-26.68 / 2, +26.68 / 2], [-26.68 / 2, +26.68 / 2]])
            im = axes[i,j].imshow(H, extent=[-26.68 / 2, +26.68 / 2, -26.68 / 2, +26.68 / 2], interpolation='nearest', origin='lower',cmap='inferno',norm=matplotlib.colors.PowerNorm(gamma=0.2))
            axes[i,j].set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            plt.setp(axes[i,j].get_xticklabels(), ha="right", rotation=90)
            axes[i,j].set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            axes[i,j].grid()
            axes[i,j].set_title(stringList[setNumberList[m]])
            if (j == 0):
                axes[i, j].set_ylabel('[mm]')
            if (i == subplot_grid[1] - 1):
                axes[i, j].set_xlabel('[mm]')
        else:
            axes[i, j].remove()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Counts #')



fig, axes = plt.subplots(nrows=subplot_grid[0], ncols=subplot_grid[1], sharex=True, sharey=True)
fig.suptitle('Interaction positions')
for i in range(subplot_grid[0]):
    for j in range(subplot_grid[1]):
        m = indeces[i,j]
        if m <= Nexp-1:
            axes[i,j].plot(X_interactions[m], Y_interactions[m],marker='.',linestyle='None',markersize=.5)
            axes[i,j].set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            plt.setp(axes[i, j].get_xticklabels(), ha="right", rotation=90)
            axes[i,j].set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            axes[i,j].grid()
            axes[i,j].set_title(stringList[setNumberList[m]])
            if(j==0):
                axes[i, j].set_ylabel('[mm]')
            if(i==subplot_grid[1]-1):
                axes[i,j].set_xlabel('[mm]')
        else:
            axes[i, j].remove()

fig, axes = plt.subplots(nrows=subplot_grid[0], ncols=subplot_grid[1], sharex=True, sharey=True)
fig.suptitle('Corrected charge histogram, sum on full matrix')
for i in range(subplot_grid[0]):
    for j in range(subplot_grid[1]):
        m = indeces[i,j]
        if m <= Nexp-1:
            axes[i,j].plot(corrected_histogram_bin_centers_full_sum[m],corrected_histogram_charge_full_sum[m]/np.max(corrected_histogram_charge_full_sum[m]),color = colors[m])
            axes[i,j].set_title(stringList[setNumberList[m]])
            if(j==0):
                axes[i, j].set_ylabel('Normalized Counts')
            if(i==subplot_grid[1]):
                axes[i,j].set_xlabel('Sum ADCu')
        else:
            axes[i, j].remove()
"""
"""

plt.show()