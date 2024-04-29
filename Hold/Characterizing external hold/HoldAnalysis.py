
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

from matplotlib.table import CustomCell
from matplotlib.widgets import TextBox

from lmfit.models import PowerLawModel, ExponentialModel, GaussianModel, LinearModel

FTtimeConst = 37e-12
CTtimeConst = 25e-9
RTtimeConst = 12.775e-6
STtimeConst = 60.000010325

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
        self.tba = self.ax.figure.add_axes([0 ,0 ,.01 ,.01])
        self.tba.set_visible(False)
        self.tb = TextBox(self.tba, '', initial="")
        self.cid2 = self.tb.on_submit(self.on_submit)
        self.currentcell = celld[(1 ,0)]

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

# matplotlib.use('QtAgg')
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

def linear_extrap(x ,x_train ,y_train):
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
    middle = float(len(input_list)) / 2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        return (input_list[int(middle)], input_list[int(middle - 1)])


os.system('SETLOCAL EnableDelayedExpansion')

fileList = []
stringList = []
cwd = os.getcwd()
os.chdir(cwd)

Cpedestal0 = np.zeros(32)
Cpedestal1 = np.zeros(32)
STDpedestal0 = np.zeros(32)
STDpedestal1 = np.zeros(32)
pixelToChannelMap = np.reshape(np.floor(np.arange(0, 32, 0.5)).astype(int), (8, 8))
pixelToAsicMap = np.matlib.repmat([0, 1], 8, 4)
channelAsicToPixelMap = np.stack((np.arange(0, 64, 2), (np.arange(1, 64, 2))), axis=1) + 1


def reshapeToSiPM(OldStructure):
    newStructure = np.zeros((len(OldStructure), 8, 8))
    for i in range(8):
        for j in range(8):
            newStructure[:, i, j] = OldStructure[:, pixelToAsicMap[i, j], pixelToChannelMap[i, j]]
    return newStructure


# define spatial mapping SiPM
SiPM_X_pos = np.matlib.repmat(1.58 + np.arange(8) * (3.16 + 0.2), 8, 1)
SiPM_Y_pos = np.rot90(SiPM_X_pos)

SiPM_XY_pos = np.stack((SiPM_X_pos, SiPM_Y_pos), axis=2)
SiPM_offset = np.array([26.68 / 2, 26.68 / 2])
SiPM_XY_pos_rel = SiPM_XY_pos - SiPM_offset

distanceMeasures = False
currTestHasDistance = []
currTestDistance = []
currTestSource = []

for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith(".csv"):
            fileList.append(os.path.join(root, file))
            source_start_index = 0
            source_end_index = file.find('.csv')
            stringList.append(file[source_start_index:source_end_index])
            # currTestSource.append(stringList[-1][:stringList[-1].find('_')])
            currTestSource.append(stringList[-1])
plt.close('all')
m = -1
Nexp = len(fileList)

board_indeces = np.ones(Nexp, dtype=int)*3
asic_indeces = np.ones(Nexp, dtype=int)*1
ch_indeces = np.ones(Nexp, dtype=int)*3

Cpedestal = np.asarray([np.ones((4, 32)) * 1023,
                        np.ones((4, 32)) * 1023,  # DT1 Missing
                        np.ones((4, 32)) * 1023,  # DT2 Missing
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


internal_hold_delay = 27 + np.arange(256)*0.34
internal_hold_adc_values = np.zeros(256)
internal_hold_adc_values_std = np.zeros(256)

external_hold_adc_values_interpolated = np.zeros((1000,Nexp-1))
external_hold_dac_interpolated = np.linspace(0,255,1000)

external_hold_adc_values = np.zeros((256,Nexp-1))
external_hold_adc_values_std = np.zeros((256,Nexp-1))

external_hold_delay = np.zeros((256,Nexp-1))
external_hold_labels = []


#

colors = distinctipy.get_colors(Nexp,rng=1)
i=0
j=0
for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith(".csv"):
            hold_raw_data = pd.read_csv((os.path.join(root, file)))
            pedestal = Cpedestal[board_indeces[i],asic_indeces[i],ch_indeces[i]]
            hold_scan_mean = (pedestal - hold_raw_data.values).mean(1)
            hold_scan_std = (pedestal - hold_raw_data.values).std(1)
            if stringList[i] == 'Internal hold':
                internal_hold_adc_values = hold_scan_mean
                internal_hold_adc_values_std= hold_scan_std
            i += 1
i = 0
j = 0
plt.figure()
for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith(".csv"):
            hold_raw_data = pd.read_csv((os.path.join(root, file)))
            pedestal = Cpedestal[board_indeces[i], asic_indeces[i], ch_indeces[i]]
            hold_scan_mean = (pedestal - hold_raw_data.values).mean(1)
            hold_scan_std = (pedestal - hold_raw_data.values).std(1)
            if stringList[i] != 'Internal hold':
                #define the error
                def error(params):
                    x_trans = params[0] + params[1]*np.arange(256) # linear transform the range
                    y_trans = linear_extrap(np.arange(256),x_trans, hold_scan_mean)
                    #cs = scipy.interpolate.CubicSpline(x_trans, hold_scan_mean)
                    #error = np.sum((cs(np.arange(256)) - internal_hold_adc_values)**2)
                    error = np.sum((y_trans- internal_hold_adc_values) ** 2)
                    return error
                optimal_params = scipy.optimize.minimize(error,[-25,25],bounds=((-100,0),(0,100)))
                # Extract the optimized parameters
                A_optimal, B_optimal = optimal_params.x
                #A_optimal = -25
                #B_optimal = 25
                # Transform x2
                external_hold_adc_values[:,j] = hold_scan_mean
                external_hold_adc_values_std[:,j] = hold_scan_std
                external_hold_delay[:,j] = A_optimal + B_optimal * np.arange(256)
                external_hold_labels.append(stringList[i])


                # Print the optimal parameters
                print(stringList[i] + " - Optimal Parameters:")
                print("A_optimal = ", A_optimal + 27, " ns")
                print("B_optimal = ", B_optimal*0.34, " ns")
                j += 1
                #plt.plot(external_hold_dac_interpolated, cs(external_hold_dac_interpolated), label=stringList[i] + ' cs interp', color=colors[i],linestyle='--')

            plt.plot(np.arange(256), hold_scan_mean,  label=stringList[i], color=colors[i])
            plt.fill_between(np.arange(256),hold_scan_mean + hold_scan_std,hold_scan_mean - hold_scan_std,color=colors[i],alpha=0.2)

            i +=1
plt.xlabel('Hold DAC code')
plt.legend()
plt.ylabel("ADC value")


plt.figure()
plt.plot(internal_hold_delay, internal_hold_adc_values, label='Internal Hold scan',color = colors[0])
plt.fill_between(internal_hold_delay, internal_hold_adc_values + internal_hold_adc_values_std,internal_hold_adc_values - internal_hold_adc_values_std,color = colors[0],alpha = 0.1)
for j in range(Nexp-1):
    plt.plot(27 + 0.34*external_hold_delay[:,j], external_hold_adc_values[:,j], label=external_hold_labels[j],color=colors[j+1])
    plt.fill_between(27 + 0.34*external_hold_delay[:,j], external_hold_adc_values[:,j] +  external_hold_adc_values_std[:,j], external_hold_adc_values[:,j] - external_hold_adc_values_std[:,j], color=colors[j+1],
                     alpha=0.1)
plt.xlabel('DAC delay [ns]')
plt.legend()
plt.ylabel("ADC value")

plt.figure()
plt.plot(np.arange(256),internal_hold_delay, label='Internal Hold scan')
for j in range(Nexp-1):
    plt.plot(np.arange(256),27 + 0.34*external_hold_delay[:,j], label=external_hold_labels[j])
plt.ylabel('DAC delay [ns]')
plt.legend()
plt.xlabel("DAC code")



plt.show()


