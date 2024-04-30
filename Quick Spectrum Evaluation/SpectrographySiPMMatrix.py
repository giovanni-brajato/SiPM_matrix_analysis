

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

from matplotlib.colors import LinearSegmentedColormap
from lmfit.models import PowerLawModel, ExponentialModel, GaussianModel, LinearModel

from lmfit import Model
import pylustrator
#pylustrator.start()
FTtimeConst = 37e-12
CTtimeConst = 25e-9
RTtimeConst = 12.775e-6
STtimeConst = RTtimeConst*(2**30)

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

# Function to invert a color represented as a 4-tuple (RGBA)
def invert_color(color):
    inverted_color = tuple(1 - c for c in color[:3]) + (color[3],)
    return inverted_color


import colorsys
def edgesToCenters(bin_edges):
    return (bin_edges[1:] + bin_edges[:-1]) / 2
def centersToEdges(bin_centers):
    steps = np.diff(bin_centers)/2
    first = bin_centers[0]-steps[0]
    last = bin_centers[-1] + steps[-1]
    return np.append(np.insert(bin_centers[:-1] + steps,0,first),last)

def calculate_contrast_color(color):
    # Unpack the color tuple
    r, g, b, _ = color

    return (1-r, 1-g, 1-b, 1)  # Black for lighter colors

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


# function that looks for coincidences
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

def _1gaussian(x,amp1,cen1,sigma1):
    return amp1 * (np.exp((-1.0 / 2.0) * (((x - cen1) / sigma1) ** 2)))
def select_callback_crt(eclick, erelease,ax,plots,n_peaks,fit,fig):

    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    len_xmasked = np.zeros(len(plots))
    for m in range(len(plots)):
        if type(plots[m]) == list:
            plots[m] = plots[m][0]
        xdata = plots[m].get_xdata()
        ydata = plots[m].get_ydata()
        mask = (xdata > min(x1, x2)) & (xdata < max(x1, x2)) & \
               (ydata > min(y1, y2)) & (ydata < max(y1, y2))
        xmasked = xdata[mask]
        ymasked = ydata[mask]
        len_xmasked[m] = len(xmasked)
    if max(len_xmasked) > 0:
        index = np.argmax(len_xmasked)
        xdata = plots[index].get_xdata()
        ydata = plots[index].get_ydata()
        mask = (xdata > min(x1, x2)) & (xdata < max(x1, x2)) & \
               (ydata > min(y1, y2)) & (ydata < max(y1, y2))
        xmasked = xdata[mask]
        ymasked = ydata[mask]

        xmax = xmasked[np.argmax(ymasked)]
        ymax = ymasked.max()

        popt_gauss, _ = scipy.optimize.curve_fit(_1gaussian, xmasked, ymasked, p0=[ymax, xmax, np.std(xmasked)])



        #res_Sgauss_fit, _ = scipy.optimize.curve_fit(Sgaussians, xmasked, ymasked,  p0=[ymax, xmax, xmax * 0.05])
        if type(fit) == list:
            fit = fit[0]
        prev_xdata = fit.get_xdata()
        prev_ydata = fit.get_ydata()
        new_xdata = xmasked
        #new_ydata = Sgaussians(xmasked, *[res_Sgauss_fit[0], abs(res_Sgauss_fit[1]), res_Sgauss_fit[2]])
        new_ydata = _1gaussian(xmasked, *[popt_gauss[0], (popt_gauss[1]), popt_gauss[2]])
        fit.set_xdata(np.concatenate((prev_xdata,new_xdata)))
        fit.set_ydata(np.concatenate((prev_ydata,new_ydata)))
        #estimated_peak = result.params['peak_center'].value
        #estimated_fwhm = result.params['peak_fwhm'].value
        estimated_peak = popt_gauss[1]
        estimated_fwhm = 2*np.sqrt(2*np.log(2))*popt_gauss[2]

        #n_peaks[0] += 1
        ax.text(xmax, ymax,"FWHM="+ f'{estimated_fwhm:.3f}'+ " ns\ncenter=" + f'{estimated_peak:.3f}'+ " ns")

        estimated_fwhm_perc =estimated_fwhm/estimated_peak*100



        renderer = fig.canvas.renderer
        ax.draw(renderer)

os.system('SETLOCAL EnableDelayedExpansion')

fileList = []
stringList = []
cwd = os.getcwd()
os.chdir(cwd)

Cpedestal0 =np.zeros(32)
Cpedestal1 =np.zeros(32)
STDpedestal0 = np.zeros(32)
STDpedestal1 = np.zeros(32)
pixelToChannelMap = np.reshape(np.floor(np.arange(0,32,0.5)).astype(int),(8,8))
pixelToAsicMap = np.matlib.repmat([0,1],8,4)
channelAsicToPixelMap = np.stack((np.arange(0,64,2),(np.arange(1,64,2))),axis=1)+1

def reshapeToSiPM(OldStructure):
    shp = OldStructure.shape
    if len(shp) == 3:
        if shp[1] == 2 and shp[2] == 32:
            newStructure = np.zeros((len(OldStructure),8,8))
            for i in range(8):
                for j in range(8):
                    newStructure[:,i,j] = OldStructure[:,pixelToAsicMap[i,j],pixelToChannelMap[i,j]]
            return newStructure
        else:
            raise ValueError
    elif len(shp) == 2:
        if shp[0] == 2 and shp[1] == 32:
            newStructure = np.zeros((8, 8))
            for i in range(8):
                for j in range(8):
                    newStructure[i, j] = OldStructure[pixelToAsicMap[i, j], pixelToChannelMap[i, j]]
            return newStructure
        else:
            raise ValueError

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
currTestHasDualBoard = []
currTestCoincidenceIndex= []
currTestHasFastMatrix = []
currTestHasSlowMatrix = []

coincidence_index = -1
for root, dirs, files in os.walk(cwd):
    csv_files = [file for file in files if file.endswith('.csv')]
    if len(csv_files) == 2:
        deuxDT = True
        coincidence_index += 1
    else:
        deuxDT = False
    for file in files:
        if file.endswith(".csv"):
            # count N of CSV files inside here, and flag this current file for looking for a coincidence
            fileList.append(os.path.join(root, file))
            source_start_index = file.find('_')
            source_end_index = file.find('.csv')
            stringList.append(file[source_start_index + 1:source_end_index])
            #currTestSource.append(stringList[-1][:stringList[-1].find('_')])
            currTestSource.append(stringList[-1])
            currTestHasDualBoard.append(deuxDT)
            if deuxDT:
                currTestCoincidenceIndex.append(coincidence_index)
            else:
                currTestCoincidenceIndex.append(-1)


plt.close('all')
m = -1
Nexp = len(fileList)

board_indeces = np.zeros(Nexp,dtype=int)
# identify the types of boards in order to determine the correct charge pedestal
for i in range(len(stringList)):
    board_number = stringList[i].split('_')[0]
    if board_number == "DT1":
        board_indeces[i] = 1
    elif board_number == "DT2":
        board_indeces[i] = 2
    elif board_number == "DT3":
        board_indeces[i] = 3
    elif board_number == "DT4":
        board_indeces[i] = 4

Cpedestal = np.asarray([np.ones((4,32))*1023,# DT0 does not exist
                       np.ones((4,32))*1023, # DT1 data Missing
                       np.ones((4,32))*1023, # DT2 data Missing
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


Gain_correction = np.asarray([np.ones((2,32)),# DT0 does not exist
                       np.ones((2,32)), # DT1 data Missing
                       np.ones((2,32)), # DT2 data Missing
                              [[1.02406474, 1.02343222, 1.02500919, 1.00693699, 1.04884671,
                                1.05157957, 1.03516032, 1.01780513, 1.03989883, 1.03735276,
                                1.0291803, 1.03034582, 1.05698207, 1.04600944, 1.,
                                1.00896464, 1.0415657, 1.05137698, 1.03900043, 1.03903619,
                                1.05534226, 1.04357368, 1.04821664, 1.05827679, 1.04284375,
                                1.04353029, 1.01306081, 1.05420223, 1.05762644, 1.05380639,
                                1.06151031, 1.03828839],
                               [0.87812889, 0.87558939, 0.87505479, 0.8657089, 0.8742606,
                                0.86504405, 0.86842357, 0.88131983, 0.86780885, 0.85377955,
                                0.86638277, 0.8917598, 0.87628586, 0.89570281, 0.84631473,
                                0.89905824, 0.88649991, 0.85953348, 0.85637068, 0.88523372,
                                0.86593324, 0.8687055, 0.88174312, 0.89528882, 0.8798367,
                                0.85131827, 0.88131971, 0.90884327, 0.89279569, 0.8669091,
                                0.86128971, 0.86218428]], # DT3
                              [[1.06109904, 1.03286922, 1.05678824, 1.0204831, 1.04595601,
                                0.99311681, 0.97801609, 1.01303914, 0.99748738, 0.99956803,
                                0.98379383, 0.98777365, 0.98605637, 0.99128339, 1.,
                                0.99032922, 0.99887288, 0.99523494, 1.00450577, 0.98207919,
                                1.01509315, 1.00145504, 1.0120405, 1.03883821, 1.00171473,
                                1.00895757, 1.00255282, 0.99749456, 1.01368722, 0.9980258,
                                1.01508538, 1.0048101],
                               [0.95292889, 0.93797745, 0.94186441, 0.96827225, 0.91832105,
                                0.88449939, 0.92952764, 0.91398183, 0.98518031, 0.9165039,
                                0.91289901, 0.92520164, 0.91422276, 0.91845566, 0.92804053,
                                0.92639526, 0.92989288, 0.94108589, 0.92624155, 0.93209221,
                                0.94005576, 0.93051905, 0.91563218, 0.93935587, 0.91834948,
                                0.92589936, 0.93882634, 0.94942763, 0.91857794, 0.93058616,
                                0.9250318, 0.90665298]] # DT4
                              ])

PixelCalibrationLUT = np.asarray([np.repeat(np.repeat(np.expand_dims(np.linspace(1023,0,1024),axis=(1,2)),2,axis=1),32,axis=2), #DT0 non-existing
                                    np.repeat(np.repeat(np.expand_dims(np.linspace(1023,0,1024),axis=(1,2)),2,axis=1),32,axis=2), #DT1 missing
                                    np.repeat(np.repeat(np.expand_dims(np.linspace(1023,0,1024),axis=(1,2)),2,axis=1),32,axis=2), #DT2 missing
                                    np.repeat(np.repeat(np.expand_dims(np.linspace(1023,0,1024),axis=(1,2)),2,axis=1),32,axis=2), #DT3 missing
                                    np.repeat(np.repeat(np.expand_dims(np.linspace(1023,0,1024),axis=(1,2)),2,axis=1),32,axis=2) #DT4
                                  ]) 
calbins=200
raw_histogram_charge_per_channel = np.zeros((2,32,1024,Nexp))
corrected_histogram_charge_per_channel = np.zeros((2, 32, 1024, Nexp))
corrected_histogram_bin_centers_per_channel = np.zeros((2, 32, 1024, Nexp))
calibrated_histogram_charge_per_channel = np.zeros((2, 32, calbins, Nexp))
calibrated_histogram_bin_centers_per_channel = np.zeros((2, 32, calbins, Nexp))


corrected_histogram_charge_per_channel_highlights =  np.zeros((2, 32, 1024, Nexp))


corrected_histogram_charge_half_sum  = [[None for j in range(2)] for e in range(Nexp)]
corrected_histogram_bin_centers_half_sum = [[None for j in range(2)] for e in range(Nexp)]
calibrated_histogram_charge_half_sum  = [[None for j in range(2)] for e in range(Nexp)]
calibrated_histogram_bin_centers_half_sum = [[None for j in range(2)] for e in range(Nexp)]

corrected_histogram_charge_full_sum  = [[] for e in range(Nexp)]
corrected_histogram_bin_centers_full_sum = [[] for e in range(Nexp)]
calibrated_histogram_charge_full_sum  = [[] for e in range(Nexp)]
calibrated_histogram_bin_centers_full_sum = [[] for e in range(Nexp)]

corrected_histogram_charge_full_avg  = [[] for e in range(Nexp)]
corrected_histogram_bin_centers_full_avg = [[] for e in range(Nexp)]
Peak_energy = np.zeros(Nexp)
HeatmapAvgCSipm = np.zeros((8, 8, Nexp))
HeatmapAvgCSipmCal = np.zeros((8, 8, Nexp))

RT_possible_coincidences_vector = [] # looking for coincidences
ST_possible_coincidences_vector = [] # looking for coincidences
FT_possible_coincidences_matrix = []
H_possible_coincidences_matrix = []
CT_possible_coincidences_matrix = []
C_corrected_possible_coincidences_matrix = []
AT_with_nan_slow_possible_coincidence_matrix  = [[] for e in range(Nexp)]
AT_with_nan_fast_possible_coincidence_matrix  = [[] for e in range(Nexp)]

X_interactions = [None for e in range(Nexp)]
Y_interactions = [None for e in range(Nexp)]
X_interactions_Cal = [None for e in range(Nexp)]
Y_interactions_Cal = [None for e in range(Nexp)]

autobins= 1000

for file in fileList:
    m +=1
    print("Processing " + file)


    ID_list = []

    data = pd.read_csv(file,sep=';',dtype=int, usecols=[i for i in range(131)]).values
    ST = data[:,0]
    ID = data[:,1]
    RT = data[:, 2]
    C = data[:, 5::4]
    CT = data[:, 3::4]
    FT = data[:, 4::4]
    H = data[:, 6::4]

    possible_asics , event_count = np.unique(ID, return_counts=True)

    # Rearrange data for ID
    RTvector_list =[[], [], [], []]
    STvector_list = [[], [], [], []]
    CTmatrix_list = [[], [], [], []]
    FTmatrix_list = [[], [], [], []]
    Cmatrix_list = [[], [], [], []]
    Hmatrix_list = [[], [], [], []]

    if len(possible_asics) == 4:  # quad readout
        quad_readout = True
        for a in range(4):
            RTvector_list[a] = RT[ID == a]
            STvector_list[a] = ST[ID == a]
            CTmatrix_list[a] = CT[ID == a,:]
            FTmatrix_list[a] = FT[ID == a,:]
            Cmatrix_list[a] = C[ID == a,:]
            Hmatrix_list[a] = H[ID == a,:]
    elif len(possible_asics) == 2:  # dual readout
        quad_readout = False
        for a in range(2):
            RTvector_list[a] = RT[ID == a]
            STvector_list[a] = ST[ID == a]
            CTmatrix_list[a] = CT[ID == a,:]
            FTmatrix_list[a] = FT[ID == a,:]
            Cmatrix_list[a] = C[ID == a,:]
            Hmatrix_list[a] = H[ID == a,:]

    # FRAME MATCHING TECHNIQUE
    Cmatrix = [[],[],[],[]]
    RTvector = [[],[],[],[]]
    STvector =  [[],[],[],[]]
    Hmatrix = [[],[],[],[]]
    FTmatrix = [[],[],[],[]]
    CTmatrix = [[],[],[],[]]

    index_frame_0 = 0
    index_frame_1 = 0
    index_frame_2 = 0
    index_frame_3 = 0
    index_frame_0_temp = 0
    index_frame_1_temp = 0
    index_frame_2_temp = 0
    index_frame_3_temp = 0
    max_index_value_0 = len(RTvector_list[0])
    max_index_value_1 = len(RTvector_list[1])
    if quad_readout:
        max_index_value_2 = len(RTvector_list[2])
        max_index_value_3 = len(RTvector_list[3])
    else:
        max_index_value_2 = 1
        max_index_value_3 = 1
    # sorting elements for RT_vector (ST vector will be useless)
    sorted_ind_0 = np.argsort(RTvector_list[0])
    sorted_ind_1 = np.argsort(RTvector_list[1])
    RTvector_list[0] = RTvector_list[0][sorted_ind_0]
    RTvector_list[1] = RTvector_list[1][sorted_ind_1]
    """
    RTvector_list[0] = np.arange(len(RTvector_list[0]))
    RTvector_list[1] = np.arange(len(RTvector_list[1]))
    """
    Cmatrix_list[0] = (Cmatrix_list[0][sorted_ind_0,:])
    Cmatrix_list[1] = (Cmatrix_list[1][sorted_ind_1,:])
    Hmatrix_list[0] = (Hmatrix_list[0][sorted_ind_0, :])
    Hmatrix_list[1] = (Hmatrix_list[1][sorted_ind_1, :])
    CTmatrix_list[0] = (CTmatrix_list[0][sorted_ind_0, :])
    CTmatrix_list[1] = (CTmatrix_list[1][sorted_ind_1, :])
    FTmatrix_list[0] = (FTmatrix_list[0][sorted_ind_0, :])
    FTmatrix_list[1] = (FTmatrix_list[1][sorted_ind_1, :])


    if quad_readout:
        sorted_ind_2 = np.argsort(RTvector_list[2])
        sorted_ind_3 = np.argsort(RTvector_list[3])
        RTvector_list[2] = RTvector_list[2][sorted_ind_2]
        RTvector_list[3] = RTvector_list[3][sorted_ind_3]
        Cmatrix_list[2] = (Cmatrix_list[2][sorted_ind_2, :])
        Cmatrix_list[3] = (Cmatrix_list[3][sorted_ind_3, :])
        Hmatrix_list[2] = (Hmatrix_list[2][sorted_ind_2, :])
        Hmatrix_list[3] = (Hmatrix_list[3][sorted_ind_3, :])
        CTmatrix_list[2] = (CTmatrix_list[2][sorted_ind_2, :])
        CTmatrix_list[3] = (CTmatrix_list[3][sorted_ind_3, :])
        FTmatrix_list[2] = (FTmatrix_list[2][sorted_ind_2, :])
        FTmatrix_list[3] = (FTmatrix_list[3][sorted_ind_3, :])
    N_matched_frames = 0
    N_discarded_frames = 0
    if quad_readout:
        coincidence_index_list = getCoincidences(
            [RTvector_list[0], RTvector_list[1], RTvector_list[2],RTvector_list[3]])
        RTvector[0]=(RTvector_list[0][coincidence_index_list[:,0]])
        RTvector[1]=(RTvector_list[1][coincidence_index_list[:,1]])
        RTvector[2]=(RTvector_list[2][coincidence_index_list[:,2]])
        RTvector[3]=(RTvector_list[3][coincidence_index_list[:,3]])
        STvector[0]=(STvector_list[0][coincidence_index_list[:,0]])
        STvector[1]=(STvector_list[1][coincidence_index_list[:,1]])
        STvector[2]=(STvector_list[2][coincidence_index_list[:,2]])
        STvector[3]=(STvector_list[3][coincidence_index_list[:,3]])
        Cmatrix[0]=(Cmatrix_list[0][coincidence_index_list[:,0]])
        Cmatrix[1]=(Cmatrix_list[1][coincidence_index_list[:,1]])
        Cmatrix[2]=(Cmatrix_list[2][coincidence_index_list[:,2]])
        Cmatrix[3]=(Cmatrix_list[3][coincidence_index_list[:,3]])
        Hmatrix[0]=(Hmatrix_list[0][coincidence_index_list[:,0]])
        Hmatrix[1]=(Hmatrix_list[1][coincidence_index_list[:,1]])
        Hmatrix[2]=(Hmatrix_list[2][coincidence_index_list[:,2]])
        Hmatrix[3]=(Hmatrix_list[3][coincidence_index_list[:,3]])
        CTmatrix[0]=(CTmatrix_list[0][coincidence_index_list[:,0]])
        CTmatrix[1]=(CTmatrix_list[1][coincidence_index_list[:,1]])
        CTmatrix[2]=(CTmatrix_list[2][coincidence_index_list[:,2]])
        CTmatrix[3]=(CTmatrix_list[3][coincidence_index_list[:,3]])
        FTmatrix[0]=(FTmatrix_list[0][coincidence_index_list[:,0]])
        FTmatrix[1]=(FTmatrix_list[1][coincidence_index_list[:,1]])
        FTmatrix[2]=(FTmatrix_list[2][coincidence_index_list[:,2]])
        FTmatrix[3]=(FTmatrix_list[3][coincidence_index_list[:,3]])
    else:
        coincidence_index_list = getCoincidences([RTvector_list[0], RTvector_list[1]])
        RTvector[0]= (RTvector_list[0][coincidence_index_list[:, 0]])
        RTvector[1]=(RTvector_list[1][coincidence_index_list[:, 1]])
        STvector[0]=(STvector_list[0][coincidence_index_list[:, 0]])
        STvector[1]=(STvector_list[1][coincidence_index_list[:, 1]])
        Cmatrix[0]=(Cmatrix_list[0][coincidence_index_list[:, 0]])
        Cmatrix[1]=(Cmatrix_list[1][coincidence_index_list[:, 1]])
        Hmatrix[0]=(Hmatrix_list[0][coincidence_index_list[:, 0]])
        Hmatrix[1]=(Hmatrix_list[1][coincidence_index_list[:, 1]])
        CTmatrix[0]=(CTmatrix_list[0][coincidence_index_list[:, 0]])
        CTmatrix[1]=(CTmatrix_list[1][coincidence_index_list[:, 1]])
        FTmatrix[0]=(FTmatrix_list[0][coincidence_index_list[:, 0]])
        FTmatrix[1]=(FTmatrix_list[1][coincidence_index_list[:, 1]])
    if quad_readout:
        Hmatrix = np.moveaxis(np.asarray([Hmatrix[0], Hmatrix[1],Hmatrix[2],Hmatrix[3]]), 0, 2)
        FTmatrix = np.moveaxis(np.asarray([FTmatrix[0], FTmatrix[1],FTmatrix[2], FTmatrix[3]]), 0, 2)
        CTmatrix = np.moveaxis(np.asarray([CTmatrix[0], CTmatrix[1],CTmatrix[2], CTmatrix[3]]), 0, 2)
    else:
        Hmatrix = np.moveaxis(np.asarray([Hmatrix[0], Hmatrix[1]]), 0, 2)
        FTmatrix = np.moveaxis(np.asarray([FTmatrix[0], FTmatrix[1]]), 0, 2)
        CTmatrix = np.moveaxis(np.asarray([CTmatrix[0], CTmatrix[1]]), 0, 2)
    Cmatrix = np.swapaxes(np.asarray([Cmatrix[0], Cmatrix[1]]), 0, 1)
    RTvector = np.asarray(RTvector[0])
    STvector = np.asarray(STvector[0])

    # calculated discarded frames


    Nevents = len(RTvector)
    for a in range (len(possible_asics)):
        print("Unmatched frames asic: " + str(a) + " : " + str(event_count[a] - Nevents) + " events, " + str((event_count[a] - Nevents)/event_count[a]*100) + " % of total")

    # Hit percentage calculation
    Hit_perc = Hmatrix.sum(0)/len(Hmatrix)*100
    Hit_valid_mask_slow = Hmatrix == 1

    # check frame deserializer

    FT_invalid_hit_values = FTmatrix.copy()
    FT_invalid_hit_values[Hmatrix == 1] = -1
    # find values that are different than -1 and 4
    critical_values_locations = np.argwhere((FT_invalid_hit_values != -1)*(FT_invalid_hit_values != 4))
    critical_frames_index = np.unique(critical_values_locations[:,0])
    unique_invalid_ft_values = np.unique(FT_invalid_hit_values,return_index=True,axis=0)
    if len(critical_frames_index) == 0:
        print('No frames with inconsistencies')
    else:
        print('There are frames with inconsistencies')
    """
    plt.figure()
    im = plt.imshow(reshapeToSiPM(np.expand_dims(Hit_perc[:,0:2].T,0)).squeeze())
    plt.xlabel('SiPM pixel X')
    plt.ylabel('SiPM pixel Y')
    plt.title("Slow matrix Hit rate % - "+stringList[m])
    plt.colorbar(im, orientation='vertical')

    if quad_readout:
        plt.figure()
        im = plt.imshow(reshapeToSiPM(np.expand_dims(Hit_perc[:, 2:4].T, 0)).squeeze())
        plt.xlabel('SiPM pixel X')
        plt.ylabel('SiPM pixel Y')
        plt.title("Fast matrix Hit rate % - " + stringList[m])
        plt.colorbar(im, orientation='vertical')

    """






    Hit_valid_mask_slow = Hmatrix[:, :, 0:2] == 1
    Hit_valid_frame_slow = Hit_valid_mask_slow.sum(axis=(1,2)) > 0
    if quad_readout:
        Hit_valid_mask_fast = Hmatrix[:, :, 2:4] == 1
        Hit_valid_mask_common = Hmatrix[:, :, 0:2]*Hmatrix[:, :, 2:4]
        Hit_valid_frame_fast = Hit_valid_mask_fast.sum(axis=(1, 2)) > 0

    # delete events with no hits

    if quad_readout:
        FastValidHitEvents = np.sum(Hit_valid_frame_fast)
        SlowValidHitEvents = np.sum(Hit_valid_frame_slow)
        perc_rejected_hit_slow_events = 100 - SlowValidHitEvents / Nevents * 100
        print(
            "Invalid slow events: " + str(Nevents - SlowValidHitEvents) + " , " + str(perc_rejected_hit_slow_events) + " % of total")
        perc_rejected_hit_fast_events =100 - FastValidHitEvents / Nevents * 100
        print("Invalid fast events: " + str(Nevents - FastValidHitEvents) + " , " + str(perc_rejected_hit_fast_events) + " % of total")

        Hit_valid_frame_common = Hit_valid_frame_slow * Hit_valid_frame_fast
        Hit_valid_frame_at_least = Hit_valid_frame_slow + Hit_valid_frame_fast
        slow_matrix_has_time_data = sum(Hit_valid_frame_slow) > 1

        if slow_matrix_has_time_data:

            NvalidEvents = Nevents - sum(~Hit_valid_frame_common)
            Cmatrix = np.delete(Cmatrix, ~Hit_valid_frame_common, axis=0)
            FTmatrix = np.delete(FTmatrix, ~Hit_valid_frame_common, axis=0)
            CTmatrix = np.delete(CTmatrix, ~Hit_valid_frame_common, axis=0)
            Hmatrix = np.delete(Hmatrix, ~Hit_valid_frame_common, axis=0)
            RTvector = np.delete(RTvector, ~Hit_valid_frame_common, axis=0)
            STvector = np.delete(STvector, ~Hit_valid_frame_common, axis=0)
        else:
            NvalidEvents = Nevents - sum(~Hit_valid_frame_at_least)
            Cmatrix = np.delete(Cmatrix, ~Hit_valid_frame_at_least, axis=0)
            FTmatrix = np.delete(FTmatrix, ~Hit_valid_frame_at_least, axis=0)
            CTmatrix = np.delete(CTmatrix, ~Hit_valid_frame_at_least, axis=0)
            Hmatrix = np.delete(Hmatrix, ~Hit_valid_frame_at_least, axis=0)
            RTvector = np.delete(RTvector, ~Hit_valid_frame_at_least, axis=0)
            STvector = np.delete(STvector, ~Hit_valid_frame_at_least, axis=0)

        Hit_valid_mask_fast = Hmatrix[:, :, 2:4] == 1
        Hit_valid_mask_slow = Hmatrix[:, :, 0:2] == 1
        Hit_valid_mask_common = Hmatrix[:, :, 0:2]*Hmatrix[:, :, 2:4]



    else:
        SlowValidHitEvents = np.sum(Hit_valid_frame_slow)
        perc_rejected_hit_slow_events = 100- SlowValidHitEvents / Nevents * 100
        print(
            "Invalid slow events: " + str(Nevents - SlowValidHitEvents) + " , " + str(perc_rejected_hit_slow_events) + " % of total")
        NvalidEvents = Nevents - sum(~Hit_valid_frame_slow)
        Cmatrix = np.delete(Cmatrix, ~Hit_valid_frame_slow, axis=0)
        FTmatrix = np.delete(FTmatrix, ~Hit_valid_frame_slow, axis=0)
        CTmatrix = np.delete(CTmatrix, ~Hit_valid_frame_slow, axis=0)
        Hmatrix = np.delete(Hmatrix, ~Hit_valid_frame_slow, axis=0)
        RTvector = np.delete(RTvector, ~Hit_valid_frame_slow, axis=0)
        STvector = np.delete(STvector, ~Hit_valid_frame_slow, axis=0)
        Hit_valid_mask_slow = Hmatrix[:, :, 0:2] == 1

    slow_matrix_has_time_data = sum(Hit_valid_frame_slow) > 1

    if slow_matrix_has_time_data:
        currTestHasSlowMatrix.append(True)
    else:
        currTestHasSlowMatrix.append(False)
    if quad_readout:
        currTestHasFastMatrix.append(True)
    else:
        currTestHasFastMatrix.append(False)



    if slow_matrix_has_time_data:
        # interframe time
        ATmatrix_slow = np.mod(CTmatrix[:, :, 0:2] + 1, 512) * CTtimeConst - FTmatrix[:, :, 0:2] * FTtimeConst
        # insert NaN values when hit data is not valid
        ATmatrix_with_nan_slow = ATmatrix_slow
        ATmatrix_with_nan_slow[~Hit_valid_mask_slow] = np.nan


        CTmatrix_with_nan_slow = CTmatrix[:, :, 0:2].astype(float)
        CTmatrix_with_nan_slow[~Hit_valid_mask_slow] = np.nan

        FTmatrix_with_nan_slow = FTmatrix[:, :, 0:2].astype(float)
        FTmatrix_with_nan_slow[~Hit_valid_mask_slow] = np.nan

        min_interframe_FT_slow = np.min(CTmatrix_with_nan_slow, axis=(1, 2))
        min_interframe_CT_slow = np.nanmin(CTmatrix_with_nan_slow, axis=(1, 2))

        fastest_pixel_slow = np.nanargmin(np.reshape(ATmatrix_with_nan_slow, (-1, 64)), axis=1)
        """
        plt.figure()
        plt.hist(fastest_pixel_slow, bins=np.arange(-0.5, 63.5, 1))
        plt.xlabel('Pixel number')
        plt.ylabel('Counts')
        plt.title("Fastest pixel (triggering event) for slow matrix - " + stringList[m])
        """



        min_interframe_time_slow = np.nanmin(np.reshape(ATmatrix_with_nan_slow, (NvalidEvents, 64)), axis=1)
        min_interframe_time_pixel_slow = np.nanargmin(np.reshape(ATmatrix_with_nan_slow, (NvalidEvents, 64)), axis=1)
        FT_min_slow = np.reshape(FTmatrix_with_nan_slow, (NvalidEvents, 64))[np.arange(NvalidEvents),min_interframe_time_pixel_slow]
        CT_min_slow = np.reshape(CTmatrix_with_nan_slow, (NvalidEvents, 64))[np.arange(NvalidEvents),min_interframe_time_pixel_slow]

        delta_AT_slow = np.mod((np.reshape(CTmatrix_with_nan_slow, (NvalidEvents, 64)).T - CT_min_slow).T, 512) * CTtimeConst - ((np.reshape(FTmatrix_with_nan_slow, (NvalidEvents, 64)).T - FT_min_slow).T) * FTtimeConst
        delta_AT_per_ASIC_slow = np.mod(((CTmatrix_with_nan_slow).T - CT_min_slow).T, 512) * CTtimeConst - (((FTmatrix_with_nan_slow).T - FT_min_slow).T) * FTtimeConst

        sameCT_mask_slow = np.mod(((CTmatrix_with_nan_slow).T - CT_min_slow).T, 512) == 0
        oneDiffCT_mask_slow = np.mod(((CTmatrix_with_nan_slow).T - CT_min_slow).T, 512) == 1

        """
        plt.figure()
        plt.hist(delta_AT_per_ASIC_slow[sameCT_mask_slow], bins=np.arange(0, 100e-9, FTtimeConst), alpha=0.5,
                 label='Identical Coarse Time')
        plt.hist(delta_AT_per_ASIC_slow[oneDiffCT_mask_slow], bins=np.arange(0, 100e-9, FTtimeConst), alpha=0.5,
                 label='1 unit coarse time difference')
        plt.xlabel('Time after the first trigger [s]')
        plt.ylabel('Counts')
        plt.title(stringList[m] + ' - Histogram distribution of arrival times after first trigger')
        plt.legend()
        """
        """
        plt.figure()
        plt.hist(delta_AT_per_ASIC_slow[:, :, 0].flatten(), bins=np.arange(0, 100e-9, FTtimeConst), color='b', alpha=0.5, label='ASIC 0')
        plt.hist(delta_AT_per_ASIC_slow[:, :, 1].flatten(), bins=np.arange(0, 100e-9, FTtimeConst), color='r', alpha=0.5, label='ASIC 1')
        plt.xlabel('Time after the first trigger [s]')
        plt.ylabel('Counts')
        plt.title(stringList[m] + ' - Histogram distribution of arrival times after first trigger (slow matrix)')
        plt.legend()
        """
        max_interframe_time_slow = np.nanmax(ATmatrix_with_nan_slow, axis=(1, 2))

        mean_interframe_time_slow = np.nanmean(ATmatrix_with_nan_slow, axis=(1, 2))
        std_interframe_time_slow = np.nanstd(ATmatrix_with_nan_slow, axis=(1, 2))

        acq_time_slow = STvector * STtimeConst + RTvector * RTtimeConst + min_interframe_time_slow

    if quad_readout:
        ATmatrix_fast = np.mod(CTmatrix[:, :, 2:4] + 1, 512) * CTtimeConst - FTmatrix[:, :, 2:4] * FTtimeConst
        # insert NaN values when hit data is not valid
        ATmatrix_with_nan_fast = ATmatrix_fast
        ATmatrix_with_nan_fast[~Hit_valid_mask_fast] = np.nan
        if slow_matrix_has_time_data:
            fast_slow_delay = ATmatrix_with_nan_fast[Hit_valid_mask_common.astype(bool)] - ATmatrix_with_nan_slow[Hit_valid_mask_common.astype(bool)]

        CTmatrix_with_nan_fast = CTmatrix[:, :, 2:4].astype(float)
        CTmatrix_with_nan_fast[~Hit_valid_mask_fast] = np.nan

        FTmatrix_with_nan_fast = FTmatrix[:, :, 2:4].astype(float)
        FTmatrix_with_nan_fast[~Hit_valid_mask_fast] = np.nan

        min_interframe_FT_fast = np.min(CTmatrix_with_nan_fast, axis=(1, 2))
        min_interframe_CT_fast = np.nanmin(CTmatrix_with_nan_fast, axis=(1, 2))

        fastest_pixel_fast = np.nanargmin(np.reshape(ATmatrix_with_nan_fast, (NvalidEvents, 64)), axis=1)
        Hmatrix_fastest_pixel_fast = np.zeros((64,NvalidEvents),dtype=bool)
        Hmatrix_fastest_pixel_slow = np.zeros((64, NvalidEvents), dtype=bool)

        for i in range(NvalidEvents):
            Hmatrix_fastest_pixel_fast[fastest_pixel_fast[i],i] = True
            if slow_matrix_has_time_data:
                Hmatrix_fastest_pixel_slow[fastest_pixel_slow[i],i] = True
        """
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
        axes[0].imshow(Hmatrix_fastest_pixel_fast, aspect='auto', interpolation='none', origin="lower")
        axes[0].set_xlabel('Frame number')
        axes[0].set_ylabel('Pixel number')
        axes[0].set_title("First fast pixel - " + stringList[m])

        axes[1].imshow(Hmatrix_fastest_pixel_slow, aspect='auto', interpolation='none', origin="lower")
        axes[1].set_xlabel('Frame number')
        axes[1].set_ylabel('Pixel number')
        axes[1].set_title("First slow pixel - " + stringList[m])

        
        plt.figure()
        plt.imshow(Hit_valid_mask_common.reshape((NvalidEvents, 64)).T, aspect='auto', interpolation='none', origin="lower")
        plt.ylabel('Pixel number')
        plt.xlabel('Event number')
        plt.title('Common activated pixels fast and slow matrix')

        plt.figure()
        plt.hist(fastest_pixel_fast, bins=np.arange(-0.5, 63.5, 1))
        plt.xlabel('Pixel number')
        plt.ylabel('Counts')
        plt.title("Fastest pixel (triggering event) for fast matrix - " + stringList[m])

        plt.figure()
        plt.hist(fast_slow_delay, bins=1000)
        plt.xlabel('Delay [s]')
        plt.ylabel('Counts')
        plt.title("Activation delay : fast pixel and slow pixel - " + stringList[m])
        """
        min_interframe_time_fast = np.nanmin(np.reshape(ATmatrix_with_nan_fast, (NvalidEvents, 64)), axis=1)
        min_interframe_time_pixel_fast = np.nanargmin(np.reshape(ATmatrix_with_nan_fast, (NvalidEvents, 64)), axis=1)
        FT_min_fast = np.reshape(FTmatrix_with_nan_fast, (NvalidEvents, 64))[
            np.arange(NvalidEvents), min_interframe_time_pixel_fast]
        CT_min_fast = np.reshape(CTmatrix_with_nan_fast, (NvalidEvents, 64))[
            np.arange(NvalidEvents), min_interframe_time_pixel_fast]

        delta_AT_fast = np.mod((np.reshape(CTmatrix_with_nan_fast, (NvalidEvents, 64)).T - CT_min_fast).T,
                               512) * CTtimeConst - ((np.reshape(FTmatrix_with_nan_fast, (
        NvalidEvents, 64)).T - FT_min_fast).T) * FTtimeConst
        delta_AT_per_ASIC_fast = np.mod(((CTmatrix_with_nan_fast).T - CT_min_fast).T, 512) * CTtimeConst - (
            ((FTmatrix_with_nan_fast).T - FT_min_fast).T) * FTtimeConst

        sameCT_mask_fast = np.mod(((CTmatrix_with_nan_fast).T - CT_min_fast).T, 512) == 0
        oneDiffCT_mask_fast = np.mod(((CTmatrix_with_nan_fast).T - CT_min_fast).T, 512) == 1

        """
        plt.figure()
        plt.hist(delta_AT_per_ASIC_slow[sameCT_mask_slow], bins=np.arange(0, 100e-9, FTtimeConst), alpha=0.5,
                 label='Identical Coarse Time')
        plt.hist(delta_AT_per_ASIC_slow[oneDiffCT_mask_slow], bins=np.arange(0, 100e-9, FTtimeConst), alpha=0.5,
                 label='1 unit coarse time difference')
        plt.xlabel('Time after the first trigger [s]')
        plt.ylabel('Counts')
        plt.title(stringList[m] + ' - Histogram distribution of arrival times after first trigger')
        plt.legend()
        """
        """
        plt.figure()
        plt.hist(delta_AT_per_ASIC_fast[:, :, 0].flatten(), bins=np.arange(0, 100e-9, FTtimeConst), color='b',
                 alpha=0.5, label='ASIC 0')
        plt.hist(delta_AT_per_ASIC_fast[:, :, 1].flatten(), bins=np.arange(0, 100e-9, FTtimeConst), color='r',
                 alpha=0.5, label='ASIC 1')
        plt.xlabel('Time after the first trigger [s]')
        plt.ylabel('Counts')
        plt.title(stringList[m] + ' - Histogram distribution of arrival times after first trigger (fast matrix)')
        plt.legend()
        """
        max_interframe_time_fast = np.nanmax(ATmatrix_with_nan_fast, axis=(1, 2))

        mean_interframe_time_fast = np.nanmean(ATmatrix_with_nan_fast, axis=(1, 2))
        std_interframe_time_fast = np.nanstd(ATmatrix_with_nan_fast, axis=(1, 2))

        acq_time_fast = STvector * STtimeConst + RTvector * RTtimeConst + min_interframe_time_fast

    """
    plt.figure()
    plt.plot(STvector,RTvector[:,0],marker='o')
    plt.ylabel("Rough time (timestamp) [a.u.]")
    plt.xlabel("Soft time (timestamp ramp detector) [a.u.]")

    plt.figure()
    plt.plot(RTvector[:, 0], marker='o',color='b',label='ASIC 0')
    plt.plot(RTvector[:, 1], marker='o', color='r',label='ASIC 1')
    plt.ylabel("Rough time (timestamp) [a.u.]")
    plt.xlabel("Frame index #")
    plt.legend()

    plt.figure()
    plt.plot(acq_time_slow, (np.mod(CTmatrix_with_nan_slow[:, :, 0].T - min_interframe_CT_slow,512)).T, linestyle='None', marker='.',
             color='b', markersize=0.25, label='ASIC 0')
    plt.plot(acq_time_slow, (np.mod(CTmatrix_with_nan_slow[:, :, 1].T - min_interframe_CT_slow,512)).T, linestyle='None', marker='.',
             color='r', markersize=0.25, label='ASIC 1')
    plt.xlabel("Acquisition time [s]")
    plt.ylabel("Interarrival CT difference [a.u.]")
    plt.title(stringList[m] + " pixel interarrival CT difference")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.figure()
    plt.plot(acq_time_slow, (FTmatrix_with_nan_slow[:, :, 0].T - min_interframe_FT_slow).T, linestyle='None', marker='.',
             color='b', markersize=0.25, label='ASIC 0')
    plt.plot(acq_time_slow, (FTmatrix_with_nan_slow[:, :, 1].T - min_interframe_FT_slow).T, linestyle='None', marker='.',
             color='r', markersize=0.25, label='ASIC 1')
    plt.xlabel("Acquisition time [s]")
    plt.ylabel("Interarrival TDC difference [TDCu]")
    plt.title(stringList[m] + " pixel interarrival TDC difference")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.figure()
    plt.plot(acq_time_slow, delta_AT_slow, linestyle='None', marker='.',
             color='b', markersize=0.25)
    plt.xlabel("Acquisition time [s]")
    plt.ylabel("Interarrival time difference [s]")
    plt.yscale('log')
    plt.title(stringList[m] + " pixel interarrival time difference")


    plt.figure()
    plt.plot(acq_time_slow,(ATmatrix_with_nan_slow[:,:,0].T - min_interframe_time_slow).T,linestyle='None',marker='.',color='b',markersize=0.25,label='ASIC 0')
    plt.plot(acq_time_slow,(ATmatrix_with_nan_slow[:,:,1].T - min_interframe_time_slow).T,linestyle='None',marker='.',color='r',markersize=0.25,label='ASIC 1')
    plt.xlabel("Acquisition time [s]")
    plt.ylabel("Interarrival time difference [s]")
    plt.title(stringList[m] + " pixel interarrival time difference")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.figure()
    plt.plot(acq_time_slow, (ATmatrix_with_nan_slow[:, :, 0].T - min_interframe_time_slow).T, linestyle='None', marker='.',
             color='b', markersize=0.25, label='ASIC 0')
    plt.plot(acq_time_slow, (ATmatrix_with_nan_slow[:, :, 1].T - min_interframe_time_slow).T, linestyle='None', marker='.',
             color='r', markersize=0.25, label='ASIC 1')
    plt.xlabel("Acquisition time [s]")
    plt.ylabel("Interarrival time difference [s]")
    plt.yscale('log')
    plt.title(stringList[m] + " pixel interarrival time difference")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


    plt.figure()
    plt.plot(acq_time_slow,max_interframe_time_slow - min_interframe_time_slow,color='r',label="Interframe time difference (last-first)")
    plt.plot(acq_time_slow, mean_interframe_time_slow - min_interframe_time_slow, color='g', label='Mean interframe time difference')
    plt.fill_between(acq_time_slow,mean_interframe_time_slow -min_interframe_time_slow - std_interframe_time_slow,mean_interframe_time_slow - min_interframe_time_slow + std_interframe_time_slow,color='g',alpha=0.5)
    plt.xlabel("Acquisition time [s]")
    plt.ylabel("Interarrival time difference [s]")
    plt.title(stringList[m] + " interarrival time difference statistics")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    """



    CalibratedCmatrix = np.zeros(shape=np.shape(Cmatrix))
    for a in range(2):
        for ch in range(32):
            CalibratedCmatrix[:,a,ch] = PixelCalibrationLUT[board_indeces[m],Cmatrix[:,a,ch],a,ch]


    # remap values equal to 0 to 1024
    Cmatrix[Cmatrix == 0] = 1024
    CorrectedCmatrix = Cpedestal[board_indeces[m],0:2,:] - Cmatrix
    CorrectedCmatrix = CorrectedCmatrix + 1 #add one
    # gain correction
    CorrectedCmatrix = Gain_correction[board_indeces[m],:,:]*CorrectedCmatrix
     #CorrectedCmatrix=CalibratedCmatrix


    ATmatrix = np.mod(CTmatrix+1,512)*CTtimeConst - FTtimeConst*FTmatrix

    # memorize timestamp information
    if currTestHasDualBoard[m]:
        RT_possible_coincidences_vector.append(RTvector)
        ST_possible_coincidences_vector.append(STvector)
        FT_possible_coincidences_matrix.append(FTmatrix)
        CT_possible_coincidences_matrix.append(CTmatrix)
        H_possible_coincidences_matrix.append(Hmatrix)
        C_corrected_possible_coincidences_matrix.append(CorrectedCmatrix)
        if slow_matrix_has_time_data:
            AT_with_nan_slow_possible_coincidence_matrix[m] = (ATmatrix_with_nan_slow)
        if quad_readout:
            AT_with_nan_fast_possible_coincidence_matrix[m] = (ATmatrix_with_nan_fast)

    HalfSumCorrectedCmatrix = CorrectedCmatrix.sum(2)
    FullSumCorrectedCmatrix =HalfSumCorrectedCmatrix.sum(1)
    FullAverageCorrectedCmatrix = (CorrectedCmatrix.mean(2)).mean(1)

    HalfSumCalibratedCmatrix = CalibratedCmatrix.sum(2)
    FullSumCalibratedCmatrix = HalfSumCalibratedCmatrix.sum(1)
    FullAverageCalibratedCmatrix = (CalibratedCmatrix.mean(2)).mean(1)

    """
    if quad_readout:
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
        axes[0].imshow(Hmatrix[:,:,0:2].reshape((NvalidEvents, 64)).T, aspect='auto', interpolation='none', origin="lower")
        axes[0].set_xlabel('Frame number')
        axes[0].set_ylabel('Pixel number')
        axes[0].set_title("Slow Hit map over time - " + stringList[m])

        axes[1].imshow(Hmatrix[:,:,2:4].reshape((NvalidEvents, 64)).T, aspect='auto', interpolation='none', origin="lower")
        axes[1].set_xlabel('Frame number')
        axes[1].set_ylabel('Pixel number')
        axes[1].set_title("Fast Hit map over time - " + stringList[m])

        axes[2].imshow(CorrectedCmatrix.reshape((NvalidEvents, 64)).T, aspect='auto', interpolation='none',
                       origin="lower")
        axes[2].set_xlabel('Frame number')
        axes[2].set_ylabel('Pixel number')
        axes[2].set_title("Energy map over time - " + stringList[m])

    else:
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
        axes[0].imshow(Hmatrix[:,:,0:2].reshape((NvalidEvents, 64)).T, aspect='auto', interpolation='none', origin="lower")
        axes[0].set_xlabel('Frame number')
        axes[0].set_ylabel('Pixel number')
        axes[0].set_title("Slow Hit map over time - " + stringList[m])

        axes[1].imshow(CorrectedCmatrix.reshape((NvalidEvents,64)).T,aspect='auto', interpolation='none',origin="lower")
        axes[1].set_xlabel('Frame number')
        axes[1].set_ylabel('Pixel number')
        axes[1].set_title("Energy map over time - " + stringList[m])
    """




    # flood map calculations
    CorrectedCmatrix_positive = CorrectedCmatrix - np.min(CorrectedCmatrix)
    CalibratedCmatrix_positive = CalibratedCmatrix - np.min(CalibratedCmatrix)
    #matrix has to be reshaped
    SiPM_Cmatrix = reshapeToSiPM(CorrectedCmatrix_positive)
    SiPM_CmatrixCalibrated = reshapeToSiPM(CalibratedCmatrix_positive)

    X_interactions[m] = np.sum(np.multiply(SiPM_Cmatrix,SiPM_XY_pos_rel[:,:,0]),axis=(1,2))/np.sum(SiPM_Cmatrix,axis=(1,2))
    Y_interactions[m] = np.sum(np.multiply(SiPM_Cmatrix,SiPM_XY_pos_rel[:,:,1]),axis=(1,2))/np.sum(SiPM_Cmatrix,axis=(1,2))

    X_interactions_Cal[m] = np.sum(np.multiply(SiPM_CmatrixCalibrated,SiPM_XY_pos_rel[:,:,0]),axis=(1,2))/np.sum(SiPM_CmatrixCalibrated,axis=(1,2))
    Y_interactions_Cal[m] = np.sum(np.multiply(SiPM_CmatrixCalibrated,SiPM_XY_pos_rel[:,:,1]),axis=(1,2))/np.sum(SiPM_CmatrixCalibrated,axis=(1,2))

    HeatmapAvgCSipm[:, :, m] = reshapeToSiPM(CorrectedCmatrix).mean(0)
    HeatmapAvgCSipmCal[:, :, m] = reshapeToSiPM(CalibratedCmatrix).mean(0)

    # highlight spectra within a range
    fwhm_perc = 28.517
    mu = 2265.135
    fwhm = fwhm_perc/100*mu
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    highlight_events_mask = (FullSumCorrectedCmatrix < (mu+ sigma)) & (FullSumCorrectedCmatrix > (mu - sigma))

    for a in range(2):
        corrected_histogram_charge_half_sum[m][a], bin_edges = np.histogram(HalfSumCorrectedCmatrix[:,a], bins = autobins)
        corrected_histogram_bin_centers_half_sum[m][a] = (bin_edges[1:] + bin_edges[:-1])/2

        calibrated_histogram_charge_half_sum[m][a], bin_edges = np.histogram(HalfSumCalibratedCmatrix[:, a],
                                                                            bins=autobins)
        calibrated_histogram_bin_centers_half_sum[m][a] = edgesToCenters(bin_edges)
        for ch in range(32):
            raw_histogram_charge_per_channel[a,ch,:,m], _ = np.histogram(Cmatrix[:,a,ch],bins=np.arange(1025)-0.5)
            corrected_bins =np.sort( Cpedestal[board_indeces[m],a,ch]-(np.arange(1025) - 0.5))
            corrected_bins_centers =(corrected_bins[1:] + corrected_bins[:-1])/2
            corrected_histogram_bin_centers_per_channel[a, ch, :, m]  = corrected_bins_centers
            corrected_histogram_charge_per_channel[a, ch, :, m], _ = np.histogram(CorrectedCmatrix[:, a, ch], bins=corrected_bins)

            calibrated_histogram_charge_per_channel[a, ch, :, m], binE = np.histogram(CalibratedCmatrix[:, a, ch],
                                                                                  bins=calbins)
            calibrated_histogram_bin_centers_per_channel[a, ch, :, m] = edgesToCenters(binE)

            corrected_histogram_charge_per_channel_highlights[a,ch,:,m], _ = np.histogram(CorrectedCmatrix[highlight_events_mask, a, ch], bins=corrected_bins)


    corrected_histogram_charge_full_sum[m],bin_edges = np.histogram(FullSumCorrectedCmatrix, bins = autobins)
    corrected_histogram_bin_centers_full_sum[m] = (bin_edges[1:] + bin_edges[:-1])/2

    calibrated_histogram_charge_full_sum[m], bin_edges = np.histogram(FullSumCalibratedCmatrix, bins=autobins)
    calibrated_histogram_bin_centers_full_sum[m] = (bin_edges[1:] + bin_edges[:-1]) / 2

    corrected_histogram_charge_full_avg[m], bin_edges = np.histogram(FullAverageCorrectedCmatrix, bins=autobins)
    corrected_histogram_bin_centers_full_avg[m] = (bin_edges[1:] + bin_edges[:-1]) / 2

    if quad_readout:
        # calculate two histograms: one for events with common activated pixel, the other for events with different activated pixel
        N_fast_activated_pixels = np.sum(Hmatrix[:, :, 2:4], axis=(1, 2))
        sum_adc_edges = centersToEdges(np.linspace(min(FullSumCorrectedCmatrix), max(FullSumCorrectedCmatrix), 1000))
        N_active_pixel_edges = centersToEdges(np.arange(max(N_fast_activated_pixels)))
        x = FullSumCorrectedCmatrix
        y = N_fast_activated_pixels
        H, xedges, yedges = np.histogram2d(x, y, bins=(sum_adc_edges, N_active_pixel_edges))
        # Histogram does not follow Cartesian convention (see Notes),
        # therefore transpose H for visualization purposes.
        H = H.T
        plt.figure()
        X, Y = np.meshgrid(sum_adc_edges, N_active_pixel_edges)
        plt.pcolormesh(X, Y, H, norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1, vmin=H.min(), vmax=H.max()),
                       cmap='inferno')
        plt.colorbar(label='# counts', extend='max')
        plt.xlabel('Energy [sum ADCu]')
        plt.ylabel('Number of fast activated pixels per event')
        plt.xlim([-100, 30000])
        plt.title(stringList[m])


    if slow_matrix_has_time_data:
        if quad_readout:
            # calculate two histograms: one for events with common activated pixel, the other for events with different activated pixel
            N_common_activated_pixels = np.sum(Hmatrix[:,:,0:2]*Hmatrix[:,:,2:4],axis=(1,2))
            sum_adc_edges = centersToEdges(np.linspace(min(FullSumCorrectedCmatrix),max(FullSumCorrectedCmatrix),1000))
            N_active_pixel_edges = centersToEdges(np.arange(max(N_common_activated_pixels)))
            x = FullSumCorrectedCmatrix
            y = N_common_activated_pixels
            H, xedges, yedges = np.histogram2d(x, y, bins=(sum_adc_edges, N_active_pixel_edges))
            # Histogram does not follow Cartesian convention (see Notes),
            # therefore transpose H for visualization purposes.
            H = H.T
            plt.figure()
            X, Y = np.meshgrid(sum_adc_edges, N_active_pixel_edges)
            plt.pcolormesh(X, Y, H, norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1,vmin=H.min(), vmax=H.max()),cmap='inferno')
            plt.colorbar(label='# counts', extend='max')
            plt.xlabel('Energy [sum ADCu]')
            plt.ylabel('Number of common activated pixels per event')
            plt.xlim([-100,30000])
            plt.title(stringList[m])
        else:
            N_activated_pixels =  np.sum(Hmatrix[:,:,0:2],axis=(1,2))
            sum_adc_edges = centersToEdges(np.linspace(min(FullSumCorrectedCmatrix),max(FullSumCorrectedCmatrix),1000))
            N_active_pixel_edges = centersToEdges(np.arange(max(N_activated_pixels)))
            x = FullSumCorrectedCmatrix
            y = N_activated_pixels
            H, xedges, yedges = np.histogram2d(x, y, bins=(sum_adc_edges, N_active_pixel_edges))
            # Histogram does not follow Cartesian convention (see Notes),
            # therefore transpose H for visualization purposes.
            H = H.T
            plt.figure()
            X, Y = np.meshgrid(sum_adc_edges, N_active_pixel_edges)
            plt.pcolormesh(X, Y, H, norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1,vmin=H.min(), vmax=H.max()),cmap='inferno')
            plt.colorbar(label='# counts', extend='max')
            plt.xlabel('Energy [sum ADCu]')
            plt.ylabel('Number of activated pixels per event')
            plt.xlim([-100, 30000])
            plt.title(stringList[m])

    Energy_per_event = CorrectedCmatrix.sum(axis=(1, 2))
    # calculate with help of the histogram what are the events closest to the peak

    peak_energy_location = np.argmax(corrected_histogram_charge_full_sum[m])
    peak_energy = corrected_histogram_bin_centers_full_sum[m][peak_energy_location]

    energy_difference_per_event = (Energy_per_event - peak_energy)**2
    most_energetic_events = np.argsort(energy_difference_per_event)
    N_displayed_energetic_events = 1

    # Define a custom colormap that maps -1 to white
    cmap = plt.cm.viridis
    cmap_colors_CT = cmap(np.linspace(0, 511, cmap.N))
    cmap_colors_CT[0] = [1, 1, 1, 1]  # Set the first color to white
    custom_cmap_CT = LinearSegmentedColormap.from_list('CustomCmap', cmap_colors_CT, cmap.N)

    cmap_colors_FT = cmap(np.linspace(0, 1023, cmap.N))
    cmap_colors_FT[0] = [1, 1, 1, 1]  # Set the first color to white
    custom_cmap_FT = LinearSegmentedColormap.from_list('CustomCmap', cmap_colors_FT, cmap.N)

    Peak_energy[m] = Energy_per_event[most_energetic_events[0]]

    for i in range(N_displayed_energetic_events):
        Gpix = reshapeToSiPM(CorrectedCmatrix[most_energetic_events[i],:,:])
        xx, yy = np.meshgrid(np.arange(0, 8), np.arange(0, 8))
        xpos = np.sum(np.multiply(Gpix,xx))/np.sum(Gpix)
        ypos = np.sum(np.multiply(Gpix,yy))/np.sum(Gpix)

        if quad_readout:
            ATmatrix_with_nan_fast = ATmatrix[most_energetic_events[i], :, 2:4]
            ATmatrix_with_nan_fast[~Hit_valid_mask_fast[most_energetic_events[i], :, :]] = np.nan
            Gdie_fast = reshapeToSiPM((ATmatrix_with_nan_fast - np.nanmin(ATmatrix_with_nan_fast)).T) / 1e-9
            CT_fast = reshapeToSiPM(CTmatrix_with_nan_fast[most_energetic_events[i], :, :].T)
            CT_fast[np.isnan(CT_fast)] = -1
            FT_fast = reshapeToSiPM(FTmatrix_with_nan_fast[most_energetic_events[i], :, :].T)
            FT_fast[np.isnan(FT_fast)] = -1

        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,sharey=True)
        fig.suptitle(stringList[m] + " - Event #"+str(i) + " @" + str(Energy_per_event[most_energetic_events[i]]) + ' sum ADCu')
        if slow_matrix_has_time_data:
            ATmatrix_with_nan_slow = ATmatrix[most_energetic_events[i], :, 0:2]
            ATmatrix_with_nan_slow[~Hit_valid_mask_slow[most_energetic_events[i], :, :]] = np.nan
            Gdie_slow = reshapeToSiPM((ATmatrix_with_nan_slow - np.nanmin(ATmatrix_with_nan_slow)).T) / 1e-9
            CT_slow = reshapeToSiPM(CTmatrix_with_nan_slow[most_energetic_events[i], :, :].T)
            CT_slow[np.isnan(CT_slow)] = -1
            FT_slow = reshapeToSiPM(FTmatrix_with_nan_slow[most_energetic_events[i], :, :].T)
            FT_slow[np.isnan(FT_slow)] = -1

            im3 = axes[1,0].imshow(CT_slow, cmap =custom_cmap_CT)
            for y in range(8):
                for x in range(8):
                    if (CT_slow[y, x] > -1):
                        axes[1,0].text(x, y, '%d' % CT_slow[y, x],
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    size=8,color=calculate_contrast_color(custom_cmap_CT(CT_slow[y, x]/CT_slow.max())))
            axes[1,0].set_xlabel('SiPM pixel X')
            axes[1,0].set_ylabel('SiPM pixel Y')
            axes[1,0].set_title('Coarse time (slow pixels)')
            divider3 = make_axes_locatable(axes[1,0])
            cax3 = divider3.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im3, cax=cax3, orientation='vertical')

            im4 = axes[1, 1].imshow(FT_slow,  cmap=custom_cmap_FT)
            for y in range(8):
                for x in range(8):
                    if (FT_slow[y, x] > -1):
                        axes[1, 1].text(x, y, '%d' % FT_slow[y, x],
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    size=8,color=calculate_contrast_color(custom_cmap_FT(FT_slow[y, x]/FT_slow.max())))
            axes[1, 1].set_xlabel('SiPM pixel X')
            axes[1, 1].set_ylabel('SiPM pixel Y')
            axes[1, 1].set_title('Fine time (slow pixels)')
            divider4 = make_axes_locatable(axes[1, 1])
            cax4 = divider4.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im4, cax=cax4, orientation='vertical')



        im = axes[0,0].imshow(Gpix)

        for y in range(Gpix.shape[0]):
            for x in range(Gpix.shape[1]):
                axes[0,0].text(x, y, '%d' % Gpix[y, x],
                         horizontalalignment='center',
                         verticalalignment='center',
                         size = 8,color=calculate_contrast_color(im.cmap(Gpix[y, x]/Gpix.max())))
        axes[0,0].plot(xpos,ypos,marker='+',markersize=15, linestyle='None',color='r')
        axes[0,0].set_xlabel('SiPM pixel X')
        axes[0,0].set_ylabel('SiPM pixel Y')
        axes[0,0].set_title('Gpix [ADCu]')
        divider = make_axes_locatable(axes[0,0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        if slow_matrix_has_time_data:
            im2 = axes[0,1].imshow(Gdie_slow)
            for y in range(Gdie_slow.shape[0]):
                for x in range(Gdie_slow.shape[1]):
                    if not(np.isnan(Gdie_slow[y,x])):
                        axes[0,1].text(x, y, '%.2f' % Gdie_slow[y, x],
                                       horizontalalignment='center',
                                       verticalalignment='center',
                                       size = 8, color=calculate_contrast_color(im2.cmap(Gdie_slow[y, x])))
            axes[0,1].set_xlabel('SiPM pixel X')
            axes[0,1].set_ylabel('SiPM pixel Y')
            axes[0,1].set_title('Gdie Slow Pixels [ns]')
            divider2 = make_axes_locatable(axes[0,1])
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im2, cax=cax2, orientation='vertical')

        if quad_readout:
            fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
            fig.suptitle(stringList[m] + " - Event #" + str(i) + " @" + str(
                Energy_per_event[most_energetic_events[i]]) + ' sum ADCu')
            im3 = axes[1, 0].imshow(CT_fast, cmap=custom_cmap_CT)
            for y in range(8):
                for x in range(8):
                    if (CT_fast[y, x] > -1):
                        axes[1, 0].text(x, y, '%d' % CT_fast[y, x],
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        size=8,
                                        color=calculate_contrast_color(custom_cmap_CT(CT_fast[y, x] / CT_fast.max())))
            axes[1, 0].set_xlabel('SiPM pixel X')
            axes[1, 0].set_ylabel('SiPM pixel Y')
            axes[1, 0].set_title('Coarse time (fast pixels)')
            divider3 = make_axes_locatable(axes[1, 0])
            cax3 = divider3.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im3, cax=cax3, orientation='vertical')

            im4 = axes[1, 1].imshow(FT_fast, cmap=custom_cmap_FT)
            for y in range(8):
                for x in range(8):
                    if (FT_fast[y, x] > -1):
                        axes[1, 1].text(x, y, '%d' % FT_fast[y, x],
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        size=8,
                                        color=calculate_contrast_color(custom_cmap_FT(FT_fast[y, x] / FT_fast.max())))
            axes[1, 1].set_xlabel('SiPM pixel X')
            axes[1, 1].set_ylabel('SiPM pixel Y')
            axes[1, 1].set_title('Fine time (fast pixels)')
            divider4 = make_axes_locatable(axes[1, 1])
            cax4 = divider4.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im4, cax=cax4, orientation='vertical')

            im = axes[0, 0].imshow(Gpix)

            for y in range(Gpix.shape[0]):
                for x in range(Gpix.shape[1]):
                    axes[0, 0].text(x, y, '%d' % Gpix[y, x],
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    size=8, color=calculate_contrast_color(im.cmap(Gpix[y, x] / Gpix.max())))
            axes[0, 0].plot(xpos, ypos, marker='+', markersize=15, linestyle='None', color='r')
            axes[0, 0].set_xlabel('SiPM pixel X')
            axes[0, 0].set_ylabel('SiPM pixel Y')
            axes[0, 0].set_title('Gpix [ADCu]')
            divider = make_axes_locatable(axes[0, 0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            im2 = axes[0, 1].imshow(Gdie_fast)
            for y in range(Gdie_fast.shape[0]):
                for x in range(Gdie_fast.shape[1]):
                    if not (np.isnan(Gdie_fast[y, x])):
                        axes[0, 1].text(x, y, '%.2f' % Gdie_fast[y, x],
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        size=8, color=calculate_contrast_color(im2.cmap(Gdie_fast[y, x])))
            axes[0, 1].set_xlabel('SiPM pixel X')
            axes[0, 1].set_ylabel('SiPM pixel Y')
            axes[0, 1].set_title('Gdie Fast Pixels [ns]')
            divider2 = make_axes_locatable(axes[0, 1])
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im2, cax=cax2, orientation='vertical')

# finding coincidences among several boards
Coincidences = np.unique(np.array(currTestCoincidenceIndex)[np.array(currTestCoincidenceIndex) != -1])


corrected_histogram_charge_full_sum_coincidence = [[[],[]] for m in range(len(Coincidences))]
corrected_histogram_bin_centers_full_sum_coincidence = [[[],[]] for m in range(len(Coincidences))]
corrected_histogram_charge_full_sum_coincidence_time_masked = [[[],[]] for m in range(len(Coincidences))]
corrected_histogram_bin_centers_full_sum_coincidence_time_masked = [[[],[]] for m in range(len(Coincidences))]
histogram_crt_fast = [[] for m in range(len(Coincidences))]
histogram_crt_bin_edges_fast =  [[] for m in range(len(Coincidences))]
histogram_crt_slow = [[] for m in range(len(Coincidences))]
histogram_crt_bin_edges_slow =  [[] for m in range(len(Coincidences))]
histogram_crt_fast_best = [[] for m in range(len(Coincidences))]
histogram_crt_bin_edges_fast_best =  [[] for m in range(len(Coincidences))]
histogram_crt_slow_best = [[] for m in range(len(Coincidences))]
histogram_crt_bin_edges_slow_best =  [[] for m in range(len(Coincidences))]

selectors_crt_fast = []
selectors_crt_slow = []
selectors_crt_fast_best = []
selectors_crt_slow_best = []

Npeaks_crt_slow = [[] for m in (Coincidences)]
fit_plot_crt_slow = [[] for m in (Coincidences)]
curr_plot_crt_slow = [[] for m in (Coincidences)]
Npeaks_crt_fast = [[] for m in (Coincidences)]
fit_plot_crt_fast = [[] for m in (Coincidences)]
curr_plot_crt_fast = [[] for m in (Coincidences)]

if len(Coincidences) > 0:
    for m in Coincidences:
        exp_index = np.squeeze(np.argwhere(m == currTestCoincidenceIndex))
        timestamps = [[],[]]
        timestamps[0]=((2 ** 30) * ST_possible_coincidences_vector[exp_index[0]] + RT_possible_coincidences_vector[exp_index[0]])
        timestamps[1] = ((2 ** 30) * ST_possible_coincidences_vector[exp_index[1]] + RT_possible_coincidences_vector[
            exp_index[1]])
        timestamps[0] = (RT_possible_coincidences_vector[
            exp_index[0]])
        timestamps[1] = (RT_possible_coincidences_vector[
            exp_index[1]])
        dual_board_coincidence_index_list = getCoincidences(timestamps)
        N_coincidence_found = np.size(dual_board_coincidence_index_list,axis=0)
        perc_coincidence_found = N_coincidence_found/np.asarray([len(timestamps[m]) for m in range(2)])
        print("Number of coincidence events: " + str(N_coincidence_found))
        print("Percentage of coincidence events: " + str(perc_coincidence_found*100) + " %")
        STvector_coincidence = np.asarray([ST_possible_coincidences_vector[exp_index[0]][dual_board_coincidence_index_list[:,0]],ST_possible_coincidences_vector[exp_index[1]][dual_board_coincidence_index_list[:,1]]])
        RTvector_coincidence = np.asarray([RT_possible_coincidences_vector[exp_index[0]][dual_board_coincidence_index_list[:, 0]],
                                           RT_possible_coincidences_vector[exp_index[1]][dual_board_coincidence_index_list[:, 1]]])
        Hmatrix_coincidence = np.asarray([H_possible_coincidences_matrix[exp_index[0]][dual_board_coincidence_index_list[:, 0], :, :],
                                           H_possible_coincidences_matrix[exp_index[1]][dual_board_coincidence_index_list[:, 1], :, :]])
        FTmatrix_coincidence = np.asarray([FT_possible_coincidences_matrix[exp_index[0]][dual_board_coincidence_index_list[:, 0], :, :],
                                           FT_possible_coincidences_matrix[exp_index[1]][dual_board_coincidence_index_list[:, 1], :, :]])
        CTmatrix_coincidence = np.asarray([CT_possible_coincidences_matrix[exp_index[0]][dual_board_coincidence_index_list[:, 0], :, :],
                                           CT_possible_coincidences_matrix[exp_index[1]][dual_board_coincidence_index_list[:, 1], :, :]])
        H_valid_mask = Hmatrix_coincidence == 1



        CTmatrix_coincidence_with_nan = CTmatrix_coincidence.astype(float)
        CTmatrix_coincidence_with_nan[~H_valid_mask] = np.nan

        AT_matrix_coincidence = np.mod(CTmatrix_coincidence[0,:,:,:] - CTmatrix_coincidence[1,:,:,:],512)

        Cmatrix_corrected_coincidence = np.asarray(
            [C_corrected_possible_coincidences_matrix[exp_index[0]][dual_board_coincidence_index_list[:, 0], :, :],
             C_corrected_possible_coincidences_matrix[exp_index[1]][dual_board_coincidence_index_list[:, 1], :, :]])
        # coincidence charge histogram
        FullSumCorrectedCmatrix_coincidence = Cmatrix_corrected_coincidence.sum(axis=(2,3))
        corrected_histogram_charge_full_sum_coincidence[m][0], bin_edges = np.histogram(FullSumCorrectedCmatrix_coincidence[0,:], bins=autobins)
        corrected_histogram_bin_centers_full_sum_coincidence[m][0] = (bin_edges[1:] + bin_edges[:-1]) / 2
        corrected_histogram_charge_full_sum_coincidence[m][1], bin_edges = np.histogram(
            FullSumCorrectedCmatrix_coincidence[1, :], bins=autobins)
        corrected_histogram_bin_centers_full_sum_coincidence[m][1] = (bin_edges[1:] + bin_edges[:-1]) / 2
        # masking energy
        energy_mask_1 = (FullSumCorrectedCmatrix_coincidence[0, :] < (Peak_energy[0] * (1 + 0.2))) * (
                    FullSumCorrectedCmatrix_coincidence[0, :] > (Peak_energy[0] * (1 - 0.2)))
        energy_mask_2 = (FullSumCorrectedCmatrix_coincidence[1, :] < (Peak_energy[1] * (1 + 0.2))) * (
                    FullSumCorrectedCmatrix_coincidence[1, :] > (Peak_energy[1] * (1 - 0.2)))

        # coincidence arrival time

        dot_size = 2
        histbins = 50000
        # masking time
        time_threshold = 50e-9

        if np.array(currTestHasSlowMatrix)[exp_index].all():
            ATmatrix_slow_coincidence = np.asarray(
            [AT_with_nan_slow_possible_coincidence_matrix[exp_index[0]][dual_board_coincidence_index_list[:, 0], :, :],
             AT_with_nan_slow_possible_coincidence_matrix[exp_index[1]][dual_board_coincidence_index_list[:, 1], :, :]])
            # calculate pixel pairs that activates
            pixel_0 = np.nanargmin(ATmatrix_slow_coincidence[0].reshape(-1,64),axis=1)
            pixel_1 = np.nanargmin(ATmatrix_slow_coincidence[1].reshape(-1,64),axis=1)
            H, xedges, yedges = np.histogram2d(pixel_0,pixel_1, bins=np.arange(65)-0.5)

            X, Y = np.meshgrid(xedges, yedges)
            fig, ax = plt.subplots(ncols=1, nrows=1)
            fig.suptitle("2D histogram of activated pixel pair - " + ' '.join(stringList[exp_index[0]].split('_')[1:]))
            im0 = ax.pcolormesh(X, Y, H.T, norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1, vmin=H.min(),
                                                                               vmax=H.max()),
                                   cmap='inferno')
            plt.colorbar(im0, ax=ax, label='# counts', extend='max')
            ax.set_xlabel('Slow pixel on ' + stringList[exp_index[0]].split('_')[0])
            ax.set_ylabel('Slow pixel on ' + stringList[exp_index[1]].split('_')[0])
            im1 = ax.pcolormesh(X, Y, H.T,
                                   norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1, vmin=H.min(),
                                                                     vmax=H.max()),
                                   cmap='inferno')
            #calculate winning pair
            best_pixels = np.unravel_index(np.argmax(H, axis=None), H.shape)
            delta_AT_slow_best = ATmatrix_slow_coincidence[0].reshape(-1,64)[:,best_pixels[0]] - ATmatrix_slow_coincidence[1].reshape(-1,64)[:,best_pixels[1]]
            delta_AT_slow_best = np.delete(delta_AT_slow_best,np.isnan(delta_AT_slow_best))
            histogram_crt_slow_best[m], histogram_crt_bin_edges_slow_best[m] = np.histogram(delta_AT_slow_best, bins=histbins)



            # event time = find the fastest absolute time for every event
            fastest_AT_slow = np.nanmin(ATmatrix_slow_coincidence, axis=(2, 3))
            # subtract the event time from both coincidences to get the coincidence time
            delta_AT_slow = fastest_AT_slow[0, :] - fastest_AT_slow[1, :]
            # represent the histogram delta T for the coincidences
            time_mask_slow = (delta_AT_slow < time_threshold) * (delta_AT_slow > - time_threshold)

            # coincidence with time mask
            FullSumCorrectedCmatrix_coincidence_time_masked = Cmatrix_corrected_coincidence.sum(axis=(2, 3))
            corrected_histogram_charge_full_sum_coincidence_time_masked[m][0], bin_edges = np.histogram(
                FullSumCorrectedCmatrix_coincidence[0, time_mask_slow], bins=autobins)
            corrected_histogram_bin_centers_full_sum_coincidence_time_masked[m][0] = (bin_edges[1:] + bin_edges[
                                                                                                      :-1]) / 2
            corrected_histogram_charge_full_sum_coincidence_time_masked[m][1], bin_edges = np.histogram(
                FullSumCorrectedCmatrix_coincidence[1, time_mask_slow], bins=autobins)
            corrected_histogram_bin_centers_full_sum_coincidence_time_masked[m][1] = (bin_edges[1:] + bin_edges[
                                                                                                      :-1]) / 2
            # CRT histogram
            histogram_crt_slow[m], histogram_crt_bin_edges_slow[m] = np.histogram(delta_AT_slow, bins=histbins)

            # 2D histogram of coincidence time vs energy
            # feature 1 : energy of coincidences (FullSumCorrectedCmatrix_coincidence)
            # feature 2 : deltaT of coincidences (deltaAT slow)
            sum_adc_edges = centersToEdges(
                np.linspace(np.min(FullSumCorrectedCmatrix_coincidence), np.max(FullSumCorrectedCmatrix_coincidence),
                            1000))
            deltaT_edges = centersToEdges(np.linspace(np.min(delta_AT_slow), np.max(delta_AT_slow), 1000))

            x0 = FullSumCorrectedCmatrix_coincidence[0, :]
            x1 = FullSumCorrectedCmatrix_coincidence[1, :]
            y = delta_AT_slow

            H0, xedges, yedges = np.histogram2d(x0, y, bins=(sum_adc_edges, deltaT_edges))
            H1, _, _ = np.histogram2d(x1, y, bins=(sum_adc_edges, deltaT_edges))
            # Histogram does not follow Cartesian convention (see Notes),
            # therefore transpose H for visualization purposes.
            H0 = H0.T
            H1 = H1.T
            fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
            fig.suptitle("2D histogram of coincidence time vs energy")
            X, Y = np.meshgrid(sum_adc_edges, deltaT_edges)
            im0 = ax[0].pcolormesh(X, Y, H0, norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1, vmin=H0.min(),
                                                                               vmax=H0.max()),
                                   cmap='inferno')
            plt.colorbar(im0, ax=ax[0], label='# counts', extend='max')
            ax[0].set_xlabel('Energy [sum ADCu]')
            ax[0].set_ylabel('Time [s]')
            im1 = ax[1].pcolormesh(X, Y, H1,
                                   norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1, vmin=H1.min(),
                                                                     vmax=H1.max()),
                                   cmap='inferno')
            plt.colorbar(im1, ax=ax[1], label='# counts', extend='max')
            ax[1].set_xlabel('Energy [sum ADCu]')
            ax[1].set_ylabel('Time [s]')

            # map of coincidences position vs arrival time (use symmetric colormap)
            x_pos_0 = np.sum(np.multiply(reshapeToSiPM(Cmatrix_corrected_coincidence[0, :]), SiPM_XY_pos_rel[:, :, 0]),
                             axis=(1, 2)) / np.sum(
                reshapeToSiPM(Cmatrix_corrected_coincidence[0, :]), axis=(1, 2))
            y_pos_0 = np.sum(np.multiply(reshapeToSiPM(Cmatrix_corrected_coincidence[0, :]), SiPM_XY_pos_rel[:, :, 1]),
                             axis=(1, 2)) / np.sum(
                reshapeToSiPM(Cmatrix_corrected_coincidence[0, :]), axis=(1, 2))
            x_pos_1 = np.sum(
                np.multiply(reshapeToSiPM(Cmatrix_corrected_coincidence[1, :]), SiPM_XY_pos_rel[:, :, 0]),
                axis=(1, 2)) / np.sum(
                reshapeToSiPM(Cmatrix_corrected_coincidence[1, :]), axis=(1, 2))
            y_pos_1 = np.sum(
                np.multiply(reshapeToSiPM(Cmatrix_corrected_coincidence[1, :]), SiPM_XY_pos_rel[:, :, 1]),
                axis=(1, 2)) / np.sum(
                reshapeToSiPM(Cmatrix_corrected_coincidence[1, :]), axis=(1, 2))

            fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
            fig.suptitle("Interaction positions of coincidences with time - slow pixels")
            s0 = ax[0].scatter(x_pos_0, y_pos_0, c=delta_AT_slow,
                               cmap="inferno", lw=0, s=dot_size)
            ax[0].set_xlabel("[mm]")
            ax[0].set_ylabel("[mm]")
            fig.colorbar(s0, ax=ax[0], label="DeltaT [s]")
            ax[0].set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[0].set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[0].grid()
            s1 = ax[1].scatter(x_pos_1, y_pos_1, c=delta_AT_slow,
                               cmap="inferno", lw=0, s=dot_size)
            ax[1].set_xlabel("[mm]")
            ax[1].set_ylabel("[mm]")
            fig.colorbar(s1, ax=ax[1], label="DeltaT [s]")
            ax[1].set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[1].set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[1].grid()

            fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
            fig.suptitle("Interaction positions of coincidences with charge - slow pixels")
            s0 = ax[0].scatter(x_pos_0, y_pos_0, c=FullSumCorrectedCmatrix_coincidence[0, :],
                               cmap="plasma", lw=0, s=10)
            ax[0].set_xlabel("[mm]")
            ax[0].set_ylabel("[mm]")
            fig.colorbar(s0, ax=ax[0], label="Energy [sum ADCu]")
            ax[0].set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[0].set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[0].grid()
            s1 = ax[1].scatter(x_pos_1, y_pos_1, c=FullSumCorrectedCmatrix_coincidence[1, :],
                               cmap="plasma", lw=0, s=10)
            ax[1].set_xlabel("[mm]")
            ax[1].set_ylabel("[mm]")
            fig.colorbar(s1, ax=ax[1], label="Energy [sum ADCu]")
            ax[1].set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[1].set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[1].grid()

        
        if np.array(currTestHasFastMatrix)[exp_index].all():  # quad readout
            ATmatrix_fast_coincidence = np.asarray(
                [AT_with_nan_fast_possible_coincidence_matrix[exp_index[0]][dual_board_coincidence_index_list[:, 0], :, :],
                 AT_with_nan_fast_possible_coincidence_matrix[exp_index[1]][dual_board_coincidence_index_list[:, 1], :, :]])
            # calculate pixel pairs that activates
            pixel_0 = np.nanargmin(ATmatrix_fast_coincidence[0].reshape(-1,64),axis=1)
            pixel_1 = np.nanargmin(ATmatrix_fast_coincidence[1].reshape(-1,64),axis=1)
            H, xedges, yedges = np.histogram2d(pixel_0,pixel_1, bins=np.arange(65)-0.5)
            X, Y = np.meshgrid(xedges, yedges)
            fig, ax = plt.subplots(ncols=1, nrows=1)
            fig.suptitle("2D histogram of activated pixel pair - " + ' '.join(stringList[exp_index[0]].split('_')[1:]))
            im0 = ax.pcolormesh(X, Y, H.T, norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1, vmin=H.min(),
                                                                               vmax=H.max()),
                                   cmap='inferno')
            plt.colorbar(im0, ax=ax, label='# counts', extend='max')
            ax.set_xlabel('Fast pixel on ' + stringList[exp_index[0]].split('_')[0])
            ax.set_ylabel('Fast pixel on ' + stringList[exp_index[1]].split('_')[0])
            im1 = ax.pcolormesh(X, Y, H.T,
                                   norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1, vmin=H.min(),
                                                                     vmax=H.max()),
                                   cmap='inferno')
            # calculate winning pair
            best_pixels = np.unravel_index(np.argmax(H, axis=None), H.shape)
            delta_AT_fast_best = ATmatrix_fast_coincidence[0].reshape(-1, 64)[:, best_pixels[0]] - \
                                 ATmatrix_fast_coincidence[1].reshape(-1, 64)[:, best_pixels[1]]
            delta_AT_fast_best = np.delete(delta_AT_fast_best, np.isnan(delta_AT_fast_best))
            histogram_crt_fast_best[m], histogram_crt_bin_edges_fast_best[m] = np.histogram(delta_AT_fast_best,
                                                                                            bins=histbins)





            # event time = find the fastest absolute time for every event
            fastest_AT_fast = np.nanmin(ATmatrix_fast_coincidence, axis=(2, 3))
            # subtract the event time from both coincidences to get the coincidence time
            delta_AT_fast = fastest_AT_fast[0, :] - fastest_AT_fast[1, :]
            # masking time
            time_mask_fast = (delta_AT_fast < time_threshold) * (delta_AT_fast > - time_threshold)

            # coincidence with time mask
            FullSumCorrectedCmatrix_coincidence_time_masked = Cmatrix_corrected_coincidence.sum(axis=(2, 3))
            corrected_histogram_charge_full_sum_coincidence_time_masked[m][0], bin_edges = np.histogram(
                FullSumCorrectedCmatrix_coincidence[0, time_mask_fast], bins=autobins)
            corrected_histogram_bin_centers_full_sum_coincidence_time_masked[m][0] = (bin_edges[1:] + bin_edges[
                                                                                                      :-1]) / 2
            corrected_histogram_charge_full_sum_coincidence_time_masked[m][1], bin_edges = np.histogram(
                FullSumCorrectedCmatrix_coincidence[1, time_mask_fast], bins=autobins)
            corrected_histogram_bin_centers_full_sum_coincidence_time_masked[m][1] = (bin_edges[1:] + bin_edges[
                                                                                                      :-1]) / 2
            # CRT histogram
            histogram_crt_fast[m], histogram_crt_bin_edges_fast[m] = np.histogram(delta_AT_fast, bins=histbins)

            sum_adc_edges = centersToEdges(
                np.linspace(np.min(FullSumCorrectedCmatrix_coincidence), np.max(FullSumCorrectedCmatrix_coincidence),
                            1000))
            deltaT_edges = centersToEdges(np.linspace(np.min(delta_AT_fast), np.max(delta_AT_fast), 1000))

            x0 = FullSumCorrectedCmatrix_coincidence[0, :]
            x1 = FullSumCorrectedCmatrix_coincidence[1, :]
            y = delta_AT_fast

            H0, xedges, yedges = np.histogram2d(x0, y, bins=(sum_adc_edges, deltaT_edges))
            H1, _, _ = np.histogram2d(x1, y, bins=(sum_adc_edges, deltaT_edges))
            # Histogram does not follow Cartesian convention (see Notes),
            # therefore transpose H for visualization purposes.
            H0 = H0.T
            H1 = H1.T
            fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
            X, Y = np.meshgrid(sum_adc_edges, deltaT_edges)
            im0 = ax[0].pcolormesh(X, Y, H0, norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1, vmin=H0.min(),
                                                                               vmax=H0.max()),
                                   cmap='inferno')
            plt.colorbar(im0, ax=ax[0], label='# counts', extend='max')
            ax[0].set_xlabel('Energy [sum ADCu]')
            ax[0].set_ylabel('Time [s]')
            im1 = ax[1].pcolormesh(X, Y, H1,
                                   norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1, vmin=H1.min(),
                                                                     vmax=H1.max()),
                                   cmap='inferno')
            plt.colorbar(im1, ax=ax[1], label='# counts', extend='max')
            ax[1].set_xlabel('Energy [sum ADCu]')
            ax[1].set_ylabel('Time [s]')

            # map of coincidences position vs arrival time (use symmetric colormap)
            x_pos_0 = np.sum(np.multiply(reshapeToSiPM(Cmatrix_corrected_coincidence[0, :]), SiPM_XY_pos_rel[:, :, 0]),
                             axis=(1, 2)) / np.sum(
                reshapeToSiPM(Cmatrix_corrected_coincidence[0, :]), axis=(1, 2))
            y_pos_0 = np.sum(np.multiply(reshapeToSiPM(Cmatrix_corrected_coincidence[0, :]), SiPM_XY_pos_rel[:, :, 1]),
                             axis=(1, 2)) / np.sum(
                reshapeToSiPM(Cmatrix_corrected_coincidence[0, :]), axis=(1, 2))
            x_pos_1 = np.sum(
                np.multiply(reshapeToSiPM(Cmatrix_corrected_coincidence[1, :]), SiPM_XY_pos_rel[:, :, 0]),
                axis=(1, 2)) / np.sum(
                reshapeToSiPM(Cmatrix_corrected_coincidence[1, :]), axis=(1, 2))
            y_pos_1 = np.sum(
                np.multiply(reshapeToSiPM(Cmatrix_corrected_coincidence[1, :]), SiPM_XY_pos_rel[:, :, 1]),
                axis=(1, 2)) / np.sum(
                reshapeToSiPM(Cmatrix_corrected_coincidence[1, :]), axis=(1, 2))

            fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
            fig.suptitle("Interaction positions of coincidences with time - fast pixels")
            s0 = ax[0].scatter(x_pos_0, y_pos_0, c=delta_AT_fast,
                               cmap="inferno", lw=0, s=dot_size)
            ax[0].set_xlabel("[mm]")
            ax[0].set_ylabel("[mm]")
            fig.colorbar(s0, ax=ax[0], label="DeltaT [s]")
            ax[0].set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[0].set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[0].grid()
            s1 = ax[1].scatter(x_pos_1, y_pos_1, c=delta_AT_fast,
                               cmap="inferno", lw=0, s=dot_size)
            ax[1].set_xlabel("[mm]")
            ax[1].set_ylabel("[mm]")
            fig.colorbar(s1, ax=ax[1], label="DeltaT [s]")
            ax[1].set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[1].set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[1].grid()

            fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
            fig.suptitle("Interaction positions of coincidences with charge - fast pixels")
            s0 = ax[0].scatter(x_pos_0, y_pos_0, c=FullSumCorrectedCmatrix_coincidence[0, :],
                               cmap="plasma", lw=0, s=10)
            ax[0].set_xlabel("[mm]")
            ax[0].set_ylabel("[mm]")
            fig.colorbar(s0, ax=ax[0], label="Energy [sum ADCu]")
            ax[0].set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[0].set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[0].grid()
            s1 = ax[1].scatter(x_pos_1, y_pos_1, c=FullSumCorrectedCmatrix_coincidence[1, :],
                               cmap="plasma", lw=0, s=10)
            ax[1].set_xlabel("[mm]")
            ax[1].set_ylabel("[mm]")
            fig.colorbar(s1, ax=ax[1], label="Energy [sum ADCu]")
            ax[1].set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[1].set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
            ax[1].grid()
        

       


        """
        fig, ax = plt.subplots(ncols=3, nrows=1)
        ax[0].hist(delta_AT_slow, bins=histbins,label='All coincidences',alpha=0.25)
        ax[0].hist(delta_AT_slow[energy_mask_1], bins=histbins, label='Around peak board 1',alpha=0.25)
        ax[0].hist(delta_AT_slow[energy_mask_2], bins=histbins, label='Around peak board 2', alpha=0.25)
        ax[0].set_xlabel("Time [s]")
        ax[0].set_ylabel("Counts")
        ax[0].set_title("Coincidence time histogram distribution slow pixels")
        ax[0].legend()
        s1 = ax[1].scatter(fastest_AT_slow[0,:],fastest_AT_slow[1,:],c=FullSumCorrectedCmatrix_coincidence[0,:],cmap="jet",lw=0 ,s=dot_size)
        ax[1].set_xlabel("Arrival time [s]")
        ax[1].set_ylabel("Arrival time [s]")
        ax[1].set_title("2D distribution slow pixels, all coincidences")
        fig.colorbar(s1, ax=ax[1],label="Energy board 1 [sum ADCu]")
        s2 = ax[2].scatter(fastest_AT_slow[0, :], fastest_AT_slow[1, :], c=FullSumCorrectedCmatrix_coincidence[1, :], cmap="jet",
                      lw=0,s=dot_size)
        ax[2].set_xlabel("Arrival time [s]")
        ax[2].set_ylabel("Arrival time [s]")
        ax[2].set_title("2D distribution slow pixels, all coincidences")
        fig.colorbar(s2, ax=ax[2],label="Energy board 2 [sum ADCu]")

        fig, ax = plt.subplots(ncols=2, nrows=2)
        ax[0,0].hist(delta_AT_slow[time_mask_slow], bins=histbins, label='Coincidences within +-' + str(time_threshold/1e-9) + ' ns', alpha=0.25)
        ax[0,0].set_xlabel("Time [s]")
        ax[0,0].set_ylabel("Counts")
        ax[0,0].set_title("Coincidence time histogram distribution slow pixels")
        ax[0,0].legend()
        s1 = ax[0,1].scatter(fastest_AT_slow[0, time_mask_slow], fastest_AT_slow[1, time_mask_slow], c=FullSumCorrectedCmatrix_coincidence[0, time_mask_slow],
                           cmap="jet", lw=0, s=dot_size)
        ax[0,1].set_xlabel("Arrival time [s]")
        ax[0,1].set_ylabel("Arrival time [s]")
        ax[0,1].set_title("2D distribution slow pixels, selected coincidences")
        fig.colorbar(s1, ax=ax[0,1], label="Energy board 1 [sum ADCu]")

        s2 = ax[1,1].scatter(fastest_AT_slow[0, time_mask_slow], fastest_AT_slow[1, time_mask_slow], c=FullSumCorrectedCmatrix_coincidence[1, time_mask_slow],
                           cmap="jet",
                           lw=0, s=dot_size)
        ax[1,1].set_xlabel("Arrival time [s]")
        ax[1,1].set_ylabel("Arrival time [s]")
        ax[1,1].set_title("2D distribution slow pixels, all coincidences")
        fig.colorbar(s2, ax=ax[1,1], label="Energy board 2 [sum ADCu]")

        ax[1, 0].set_title('Corrected charge histogram, sum on full matrix')
        for i in range(Nexp):
            ax[1, 0].plot(corrected_histogram_bin_centers_full_sum[i],
                     corrected_histogram_charge_full_sum[i] / corrected_histogram_charge_full_sum[i].max(),
                     label=stringList[i])


        ax[1, 0].legend()
        ax[1, 0].set_xlabel('Sum ADCu')
        ax[1, 0].set_ylabel('Relative Counts')


        fig, ax = plt.subplots(ncols=2, nrows=1)
        s1 = ax[0].scatter(fastest_AT_slow[0, energy_mask_1], fastest_AT_slow[1, energy_mask_1], c=FullSumCorrectedCmatrix_coincidence[0, energy_mask_1],
                           cmap="jet", lw=0, s=dot_size)
        ax[0].set_xlabel("Arrival time [s]")
        ax[0].set_ylabel("Arrival time [s]")
        ax[0].set_title("2D distribution slow pixels, peak coincidences")
        fig.colorbar(s1, ax=ax[0], label="Energy board 1 [sum ADCu]")
        s2 = ax[1].scatter(fastest_AT_slow[0, energy_mask_2], fastest_AT_slow[1, energy_mask_2], c=FullSumCorrectedCmatrix_coincidence[1, energy_mask_2],
                           cmap="jet",
                           lw=0, s=dot_size)
        ax[1].set_xlabel("Arrival time [s]")
        ax[1].set_ylabel("Arrival time [s]")
        ax[1].set_title("2D distribution slow pixels, peak coincidences")
        fig.colorbar(s2, ax=ax[1], label="Energy board 2 [sum ADCu]")
        """
        
        """
        # represent the histogram delta T for the coincidences
        fig, ax = plt.subplots(ncols=3, nrows=1)
        ax[0].hist(delta_AT_fast, bins=histbins, label='All coincidences', alpha=0.25)
        ax[0].hist(delta_AT_fast[energy_mask_1], bins=histbins, label='Around peak board 1', alpha=0.25)
        ax[0].hist(delta_AT_fast[energy_mask_2], bins=histbins, label='Around peak board 2', alpha=0.25)
        ax[0].set_xlabel("Time [s]")
        ax[0].set_ylabel("Counts")
        ax[0].set_title("Coincidence time histogram distribution fast pixels")
        ax[0].legend()
        s1 = ax[1].scatter(fastest_AT_fast[0, :], fastest_AT_fast[1, :], c=FullSumCorrectedCmatrix_coincidence[0, :],
                           cmap="jet", lw=0, s=dot_size)
        ax[1].set_xlabel("Arrival time [s]")
        ax[1].set_ylabel("Arrival time [s]")
        ax[1].set_title("2D distribution fast pixels, all coincidences")
        fig.colorbar(s1, ax=ax[1], label="Energy board 1 [sum ADCu]")
        s2 = ax[2].scatter(fastest_AT_fast[0, :], fastest_AT_fast[1, :], c=FullSumCorrectedCmatrix_coincidence[1, :],
                           cmap="jet",
                           lw=0, s=dot_size)
        ax[2].set_xlabel("Arrival time [s]")
        ax[2].set_ylabel("Arrival time [s]")
        ax[2].set_title("2D distribution fast pixels, all coincidences")
        fig.colorbar(s2, ax=ax[2], label="Energy board 2 [sum ADCu]")

        fig, ax = plt.subplots(ncols=2, nrows=1)
        s1 = ax[0].scatter(fastest_AT_fast[0, energy_mask_1], fastest_AT_fast[1, energy_mask_1],
                           c=FullSumCorrectedCmatrix_coincidence[0, energy_mask_1],
                           cmap="jet", lw=0, s=dot_size)
        ax[0].set_xlabel("Arrival time [s]")
        ax[0].set_ylabel("Arrival time [s]")
        ax[0].set_title("2D distribution fast pixels, peak coincidences")
        fig.colorbar(s1, ax=ax[0], label="Energy board 1 [sum ADCu]")
        s2 = ax[1].scatter(fastest_AT_fast[0, energy_mask_2], fastest_AT_fast[1, energy_mask_2],
                           c=FullSumCorrectedCmatrix_coincidence[1, energy_mask_2],
                           cmap="jet",
                           lw=0, s=dot_size)
        ax[1].set_xlabel("Arrival time [s]")
        ax[1].set_ylabel("Arrival time [s]")
        ax[1].set_title("2D distribution fast pixels, peak coincidences")
        fig.colorbar(s2, ax=ax[1], label="Energy board 2 [sum ADCu]")
        """
    # CRT fitting model
    def triangular(x,offset,slope,maxvalue):
        return maxvalue/slope*(0.5*np.abs(x-(offset + slope)) + 0.5*np.abs(x-(offset-slope))-np.abs(x-offset))
    def gaussian(x,offset,fwhm,maxvalue):
        return maxvalue*np.exp(-4*np.log(2)*((x-offset)/fwhm)**2)
    mod_triang = Model(triangular,prefix='bkg_')
    #mod_gauss = GaussianModel(prefix='crt_')
    mod_gauss = Model(gaussian,prefix='crt_')
    model = mod_gauss+mod_triang



    # compare coincidences
    # plot each one crt and selector to calculate CRT (gaussian fit + triangular distribution?)
    # compare all CRT + selector to calculate CRT
    fig_crt, axes = plt.subplots(ncols=2, nrows=len(Coincidences), sharex=True)
    fig_crt.suptitle('CRT for coincidences')
    if len(Coincidences) == 1:
        if currTestHasSlowMatrix[currTestCoincidenceIndex[0]]:
            max_pos = np.argmax(histogram_crt_slow[0])
            ymax = histogram_crt_slow[0][max_pos]
            xmax = edgesToCenters(histogram_crt_bin_edges_slow[0])[max_pos]
            # fitting peak
            # model.set_param_hint('crt_maxvalue', value=ymax, min=ymax * 0.9, max=ymax * 1.1)
            # model.set_param_hint('crt_offset', value=xmax, min=xmax * 0.8, max=xmax * 1.2)
            #             model.set_param_hint('crt_fwhm', value=xmax * 0.05, min=xmax * 0.05, max=1)
            # model.set_param_hint('bkg_maxvalue', value=ymax, min=1, max=ymax * 1.1)
            # model.set_param_hint('bkg_offset', value=0, min=-100e-9, max=100e-9)
            # model.set_param_hint('bkg_slope', value=ymax/6e-6, min=1/6e-6, max=ymax*1.1/6e-6)
            #params = model.make_params(crt_maxvalue=ymax - (histogram_crt_slow[0].mean()), crt_offset=0, crt_fwhm=1e-9,
#                                       bkg_maxvalue=(histogram_crt_slow[0].mean()), bkg_offset=0, bkg_slope=1.5e-7)
            # gauss_params = mod_gauss.make_params(crt_amplitude=ymax, crt_center=xmax, crt_sigma=xmax * 0.05)
            # triang_params = mod_triang.make_params(bkg_maxvalue=ymax, bkg_offset=0,bkg_slope=ymax/6e-6)
            # do fit
            #result = model.fit(histogram_crt_slow[0], params, x=edgesToCenters(histogram_crt_bin_edges_slow[0]))
            # result2 = mod_gauss.fit(histogram_crt[m], gauss_params, x=edgesToCenters(histogram_crt_bin_edges[m]))
            # result3 = mod_triang.fit(histogram_crt[m], triang_params, x=edgesToCenters(histogram_crt_bin_edges[m]))
            # print out fitting statistics, best-fit parameters, uncertainties,....
            #print(result.params)

            axes.stairs(histogram_crt_slow[0], histogram_crt_bin_edges_slow[0],
                        label=stringList[currTestCoincidenceIndex[0]][4:] + ' slow pixels')
            #axes.plot(edgesToCenters(histogram_crt_bin_edges_slow[0]), result.best_fit, label='Gauss + Triangular fit',
            #          linestyle='--', color='r')
            axes.legend()
            axes.set_ylabel('# counts')
            axes.set_xlabel('Delta T [s]')
        if currTestHasFastMatrix[currTestCoincidenceIndex[0]]:
            max_pos = np.argmax(histogram_crt_fast[0])
            ymax = histogram_crt_fast[0][max_pos]
            xmax = edgesToCenters(histogram_crt_bin_edges_fast[0])[max_pos]
            # fitting peak
            # model.set_param_hint('crt_maxvalue', value=ymax, min=ymax * 0.9, max=ymax * 1.1)
            # model.set_param_hint('crt_offset', value=xmax, min=xmax * 0.8, max=xmax * 1.2)
            #             model.set_param_hint('crt_fwhm', value=xmax * 0.05, min=xmax * 0.05, max=1)
            # model.set_param_hint('bkg_maxvalue', value=ymax, min=1, max=ymax * 1.1)
            # model.set_param_hint('bkg_offset', value=0, min=-100e-9, max=100e-9)
            # model.set_param_hint('bkg_slope', value=ymax/6e-6, min=1/6e-6, max=ymax*1.1/6e-6)
           # params = model.make_params(crt_maxvalue=ymax - (histogram_crt_fast[0].mean()), crt_offset=0,
#                                       crt_fwhm=1e-9,
     #                                  bkg_maxvalue=(histogram_crt_fast[0].mean()), bkg_offset=0, bkg_slope=1.5e-7)
            # gauss_params = mod_gauss.make_params(crt_amplitude=ymax, crt_center=xmax, crt_sigma=xmax * 0.05)
            # triang_params = mod_triang.make_params(bkg_maxvalue=ymax, bkg_offset=0,bkg_slope=ymax/6e-6)
            # do fit
            #result = model.fit(histogram_crt_fast[0], params, x=edgesToCenters(histogram_crt_bin_edges_fast[m]))
            # result2 = mod_gauss.fit(histogram_crt[m], gauss_params, x=edgesToCenters(histogram_crt_bin_edges[m]))
            # result3 = mod_triang.fit(histogram_crt[m], triang_params, x=edgesToCenters(histogram_crt_bin_edges[m]))
            # print out fitting statistics, best-fit parameters, uncertainties,....
           # print(result.params)

            axes.stairs(histogram_crt_fast[0], histogram_crt_bin_edges_fast[0],
                        label=stringList[currTestCoincidenceIndex[0]][4:]+ ' fast pixels')
          #  axes.plot(edgesToCenters(histogram_crt_bin_edges_fast[0]), result.best_fit,
            #          label='Gauss + Triangular fit',
            #          linestyle='--', color='r')
            axes.legend()
            axes.set_ylabel('# counts')
            axes.set_xlabel('Delta T [s]')
    else:
        for m in Coincidences:
            if np.sum((currTestHasSlowMatrix*(np.array(currTestCoincidenceIndex) == m)))> 0:
                max_pos = np.argmax(histogram_crt_slow[m])
                ymax = histogram_crt_slow[m][max_pos]
                xmax = edgesToCenters(histogram_crt_bin_edges_slow[m])[max_pos]
                #fitting peak
                #model.set_param_hint('crt_maxvalue', value=ymax, min=ymax * 0.9, max=ymax * 1.1)
                #model.set_param_hint('crt_offset', value=xmax, min=xmax * 0.8, max=xmax * 1.2)
                #model.set_param_hint('crt_fwhm', value=xmax * 0.05, min=xmax * 0.05, max=1)
                #model.set_param_hint('bkg_maxvalue', value=ymax, min=1, max=ymax * 1.1)
                #model.set_param_hint('bkg_offset', value=0, min=-100e-9, max=100e-9)
                #model.set_param_hint('bkg_slope', value=ymax/6e-6, min=1/6e-6, max=ymax*1.1/6e-6)
               # params = model.make_params(crt_maxvalue=ymax - (histogram_crt_slow[m].mean()), crt_offset=0, crt_fwhm=1e-9,
          #                                 bkg_maxvalue=(histogram_crt_slow[m].mean()), bkg_offset=0,bkg_slope=1.5e-7)
                #gauss_params = mod_gauss.make_params(crt_amplitude=ymax, crt_center=xmax, crt_sigma=xmax * 0.05)
                #triang_params = mod_triang.make_params(bkg_maxvalue=ymax, bkg_offset=0,bkg_slope=ymax/6e-6)
                # do fit
               # result = model.fit(histogram_crt_slow[m], params, x=edgesToCenters(histogram_crt_bin_edges_slow[m]))
                #result2 = mod_gauss.fit(histogram_crt[m], gauss_params, x=edgesToCenters(histogram_crt_bin_edges[m]))
                #result3 = mod_triang.fit(histogram_crt[m], triang_params, x=edgesToCenters(histogram_crt_bin_edges[m]))
                # print out fitting statistics, best-fit parameters, uncertainties,....
               # print(result.params)
                axes[m,0].stairs(histogram_crt_slow[m], histogram_crt_bin_edges_slow[m]/1e-9, label=stringList[np.argwhere(currTestCoincidenceIndex == m).squeeze()[0]][4:] + ' - slow pixels')
                axes[m,1].stairs(histogram_crt_slow_best[m], histogram_crt_bin_edges_slow_best[m]/1e-9, label=stringList[np.argwhere(currTestCoincidenceIndex == m).squeeze()[0]][4:] + ' - slow pixels best pair')

                curr_plot_crt_slow[m] = axes[m,0].plot(edgesToCenters(histogram_crt_bin_edges_slow[m])/1e-9,histogram_crt_slow[m],linestyle='None')
                curr_plot_crt_slow_2 = axes[m, 1].plot(edgesToCenters(histogram_crt_bin_edges_slow_best[m]) / 1e-9,
                                                        histogram_crt_slow_best[m], linestyle='None')
                fit_plot_crt_slow[m] = axes[m,0].plot([],[],label='Gauss fit', color='r', linestyle='--', marker='None')
                fit_plot_crt_slow_2 = axes[m, 1].plot([], [], label='Gauss fit', color='r', linestyle='--',
                                                       marker='None')
                axes[m,0].legend()
                axes[m, 1].legend()
                axes[m,0].set_ylabel('# counts')
                axes[m,0].set_xlabel('Delta T [ns]')
                axes[m, 1].set_ylabel('# counts')
                axes[m, 1].set_xlabel('Delta T [ns]')

                Npeaks_crt_slow[m] = [0]
                selectors_crt_slow.append(RectangleSelector( axes[m,0], partial(select_callback_crt, plots=curr_plot_crt_slow[m], fig=fig_crt, ax=axes[m,0],
                                            fit=fit_plot_crt_slow[m]
                                            , n_peaks=Npeaks_crt_slow[m]), useblit=True, button=[1, 3],
                                    spancoords='pixels', interactive=True))
                selectors_crt_slow_best.append(RectangleSelector(axes[m, 1],
                                                        partial(select_callback_crt, plots=curr_plot_crt_slow_2,
                                                                fig=fig_crt, ax=axes[m, 1],
                                                                fit=fit_plot_crt_slow_2
                                                                , n_peaks=[0]), useblit=True,
                                                        button=[1, 3],
                                                        spancoords='pixels', interactive=True))

            if np.sum((currTestHasFastMatrix*(np.array(currTestCoincidenceIndex) == m)))> 0:
                max_pos = np.argmax(histogram_crt_fast[m])
                ymax = histogram_crt_fast[m][max_pos]
                xmax = edgesToCenters(histogram_crt_bin_edges_fast[m])[max_pos]
                # fitting peak
                # model.set_param_hint('crt_maxvalue', value=ymax, min=ymax * 0.9, max=ymax * 1.1)
                # model.set_param_hint('crt_offset', value=xmax, min=xmax * 0.8, max=xmax * 1.2)
                # model.set_param_hint('crt_fwhm', value=xmax * 0.05, min=xmax * 0.05, max=1)
                # model.set_param_hint('bkg_maxvalue', value=ymax, min=1, max=ymax * 1.1)
                # model.set_param_hint('bkg_offset', value=0, min=-100e-9, max=100e-9)
                # model.set_param_hint('bkg_slope', value=ymax/6e-6, min=1/6e-6, max=ymax*1.1/6e-6)
                #params = model.make_params(crt_maxvalue=ymax - (histogram_crt_fast[m].mean()), crt_offset=0,
              #                             crt_fwhm=1e-9,
             #                              bkg_maxvalue=(histogram_crt_fast[m].mean()), bkg_offset=0, bkg_slope=1.5e-7)
                # gauss_params = mod_gauss.make_params(crt_amplitude=ymax, crt_center=xmax, crt_sigma=xmax * 0.05)
                # triang_params = mod_triang.make_params(bkg_maxvalue=ymax, bkg_offset=0,bkg_slope=ymax/6e-6)
                # do fit
                #result = model.fit(histogram_crt_fast[m], params, x=edgesToCenters(histogram_crt_bin_edges_fast[m]))
                # result2 = mod_gauss.fit(histogram_crt[m], gauss_params, x=edgesToCenters(histogram_crt_bin_edges[m]))
                # result3 = mod_triang.fit(histogram_crt[m], triang_params, x=edgesToCenters(histogram_crt_bin_edges[m]))
                # print out fitting statistics, best-fit parameters, uncertainties,....
                #print(result.params)

                axes[m,0].stairs(histogram_crt_fast[m], histogram_crt_bin_edges_fast[m]/1e-9,
                               label=stringList[np.argwhere(currTestCoincidenceIndex == m).squeeze()[0]][
                                     4:] + ' - fast pixels')
                axes[m,1].stairs(histogram_crt_fast_best[m], histogram_crt_bin_edges_fast_best[m] / 1e-9,
                               label=stringList[np.argwhere(currTestCoincidenceIndex == m).squeeze()[0]][
                                     4:] + ' - fast pixels best pair')
                curr_plot_crt_fast[m] = axes[m,0].plot(edgesToCenters(histogram_crt_bin_edges_fast[m]) / 1e-9,
                                                     histogram_crt_fast[m],
                                                     linestyle='None')
                fit_plot_crt_fast[m] = axes[m,0].plot([], [],
                             label='Gauss fit',  color='r', linestyle='--', marker='None')
                curr_plot_crt_fast_2 = axes[m, 1].plot(edgesToCenters(histogram_crt_bin_edges_fast_best[m]) / 1e-9,
                                                        histogram_crt_fast_best[m],
                                                        linestyle='None')
                fit_plot_crt_fast_2 = axes[m, 1].plot([], [],
                                                       label='Gauss fit', color='r', linestyle='--', marker='None')

                axes[m,0].legend()
                axes[m,0].set_ylabel('# counts')
                axes[m,0].set_xlabel('Delta T [ns]')
                axes[m, 1].legend()
                axes[m, 1].set_ylabel('# counts')
                axes[m, 1].set_xlabel('Delta T [ns]')
                Npeaks_crt_fast[m]= [0]
                selectors_crt_fast.append(
                    RectangleSelector(axes[m,0], partial(select_callback_crt, plots=curr_plot_crt_fast[m], fig=fig_crt, ax=axes[m,0],
                                                       fit=fit_plot_crt_fast[m]
                                                       , n_peaks=Npeaks_crt_fast[m]), useblit=True, button=[1, 3],
                                      spancoords='pixels', interactive=True))
                selectors_crt_fast_best.append(RectangleSelector(axes[m, 1], partial(select_callback_crt, plots=curr_plot_crt_fast_2, fig=fig_crt, ax=axes[m, 1],
                                                      fit=fit_plot_crt_fast_2
                                                      , n_peaks=[0]), useblit=True, button=[1, 3],
                                  spancoords='pixels', interactive=True))



# compare spectra with associated CRT
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    fig.suptitle('Spectra for coincidences')
    for m in Coincidences:
        axes[0].plot(corrected_histogram_bin_centers_full_sum_coincidence[m][0],corrected_histogram_charge_full_sum_coincidence[m][0]/max(corrected_histogram_charge_full_sum_coincidence[m][0]),label=stringList[np.argwhere(currTestCoincidenceIndex == m).squeeze()[0]])
        axes[1].plot(corrected_histogram_bin_centers_full_sum_coincidence[m][1],corrected_histogram_charge_full_sum_coincidence[m][1]/max(corrected_histogram_charge_full_sum_coincidence[m][1]),label=stringList[np.argwhere(currTestCoincidenceIndex == m).squeeze()[1]])
    axes[0].legend()
    axes[1].legend()
    axes[0].set_ylabel('Normalized Counts')
    axes[1].set_ylabel('Normalized Counts')
    axes[0].set_xlabel('Energy [sum ADCu]')
    axes[1].set_xlabel('Energy [sum ADCu]')


fig, axes = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True,gridspec_kw={'hspace': 0, 'wspace': 0,'left':0,'bottom':0,'right':1,'top':0.97})
fig.suptitle('Raw charge histogram per channel')
for m in range(Nexp):
    for i in range(8):
        for j in range(8):
            ax = axes[i, j]
            ax.plot(np.arange(1024),raw_histogram_charge_per_channel[pixelToAsicMap[i, j],pixelToChannelMap[i, j],:, m], label=stringList[m])
            ax.text(.99, .99, 'A' + str(pixelToAsicMap[i,j]) + 'C' + str(pixelToChannelMap[i,j]), ha='right', va='top', transform=ax.transAxes)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(), loc='lower center',ncol=Nexp)

fig, axes = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True,gridspec_kw={'hspace': 0, 'wspace': 0})
fig.suptitle('Corrected charge histogram per channel')
for m in range(Nexp):
    for i in range(8):
        for j in range(8):
            ax = axes[i, j]
            ax.plot(corrected_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m], corrected_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m], label=stringList[m])
            #ax.plot(calibrated_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m], calibrated_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m], label=stringList[m] + ' calibrated')

            ax.text(.99, .99, 'A' + str(pixelToAsicMap[i,j]) + 'C' + str(pixelToChannelMap[i,j]), ha='right', va='top', transform=ax.transAxes, fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='x', labelsize=8)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(), loc='lower center',ncol=Nexp)

# show single histograms and average with shaded area
for m in range(Nexp):
    plt.figure()
    average_binning = np.arange(0,1024)
    average_histogram = np.zeros((1024,64))
    n = 0
    for i in range(8):
        for j in range(8):
            plt.plot(corrected_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m],
                    corrected_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m]/corrected_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m].max(),
                    color='b',alpha=0.1,label='Histogram of single pixel')
            #plt.plot(calibrated_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m],
            #         calibrated_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m]/calibrated_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m].max(),
            #         color='m', alpha=0.1, label='Histogram of single pixel, calibrated')

            average_histogram[:,n] = linear_extrap(average_binning,corrected_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m],corrected_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m])
            n +=1
    plt.plot(average_binning,  average_histogram.mean(1)/average_histogram.mean(1).max(), color='g',
             label='Average of histograms')
    plt.fill_between(average_binning, (average_histogram.mean(1) - average_histogram.std(1))/average_histogram.mean(1).max(), (average_histogram.mean(1) + average_histogram.std(1))/average_histogram.mean(1).max(),color='g',alpha=0.3)
    plt.plot(corrected_histogram_bin_centers_full_avg[m],corrected_histogram_charge_full_avg[m]/corrected_histogram_charge_full_avg[m].max(), color='r',label='Histogram of average')
    plt.title(stringList[m])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel('Charge ADCu')
    plt.ylabel('Relative counts')




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
            #plt.plot(calibrated_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m],
            #         calibrated_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m],
            #         color='m', alpha=0.1, label='Histogram of single pixel, calibrated')

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
fig, axes = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True,gridspec_kw={'hspace': 0, 'wspace': 0,'left':0,'bottom':0,'right':1,'top':0.97})
fig.suptitle('Corrected charge histogram per channel')
for m in range(Nexp):
    for i in range(8):
        for j in range(8):
            ax = axes[i, j]
            ax.plot(corrected_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m], corrected_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m], label=stringList[m])
            ax.plot(corrected_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m], corrected_histogram_charge_per_channel_highlights[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m], label=stringList[m] + " peak +-2sigma")
            ax.text(.99, .99, 'A' + str(pixelToAsicMap[i,j]) + 'C' + str(pixelToChannelMap[i,j]), ha='right', va='top', transform=ax.transAxes)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(), loc='lower center',ncol=Nexp)

fig, axes = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True,gridspec_kw={'hspace': 0, 'wspace': 0,'left':0,'bottom':0,'right':1,'top':0.97})
fig.suptitle('Corrected charge histogram per channel, normalized')
for m in range(Nexp):
    for i in range(8):
        for j in range(8):
            ax = axes[i, j]
            ax.plot(corrected_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m], corrected_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m]/max(corrected_histogram_charge_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m]), label=stringList[m])
            ax.plot(corrected_histogram_bin_centers_per_channel[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m], corrected_histogram_charge_per_channel_highlights[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m]/max(corrected_histogram_charge_per_channel_highlights[pixelToAsicMap[i, j], pixelToChannelMap[i, j], :, m]), label=stringList[m] + " peak +-2sigma")
            ax.text(.99, .99, 'A' + str(pixelToAsicMap[i,j]) + 'C' + str(pixelToChannelMap[i,j]), ha='right', va='top', transform=ax.transAxes)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(), loc='lower center',ncol=Nexp)
"""
fig,axes = plt.subplots(nrows=1, ncols=Nexp)
fig.suptitle('SiPM average deposed energy (All events)')
if Nexp>1:
    for m in range(Nexp):
        im=axes[m].imshow(HeatmapAvgCSipm[:, :, m],vmin=HeatmapAvgCSipm.min(),vmax=HeatmapAvgCSipm.max())
        axes[m].set_xlabel('SiPM pixel X')
        axes[m].set_ylabel('SiPM pixel Y')
        axes[m].set_title(stringList[m])
        divider = make_axes_locatable(axes[m])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
else:
    m = 0
    im=axes.imshow(HeatmapAvgCSipm[:, :, m],vmin=HeatmapAvgCSipm.min(),vmax=HeatmapAvgCSipm.max())
    axes.set_xlabel('SiPM pixel X')
    axes.set_ylabel('SiPM pixel Y')
    axes.set_title(stringList[m])
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
plt.tight_layout()

"""
fig,axes = plt.subplots(nrows=1, ncols=Nexp)
fig.suptitle('SiPM average deposed energy (All events) - Calibrated')
if Nexp>1:
    for m in range(Nexp):
        im=axes[m].imshow(HeatmapAvgCSipmCal[:, :, m],vmin=HeatmapAvgCSipmCal.min(),vmax=HeatmapAvgCSipmCal.max())
        axes[m].set_xlabel('SiPM pixel X')
        axes[m].set_ylabel('SiPM pixel Y')
        axes[m].set_title(stringList[m])
        divider = make_axes_locatable(axes[m])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
else:
    m = 0
    im=axes.imshow(HeatmapAvgCSipmCal[:, :, m],vmin=HeatmapAvgCSipmCal.min(),vmax=HeatmapAvgCSipmCal.max())
    axes.set_xlabel('SiPM pixel X')
    axes.set_ylabel('SiPM pixel Y')
    axes.set_title(stringList[m])
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
plt.tight_layout()
"""
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,gridspec_kw={'hspace': 0, 'wspace': 0,'left':0,'bottom':0,'right':1,'top':0.97})
fig.suptitle('Corrected charge histogram, sum on half matrix')
for m in range(Nexp):
    axes[0].plot(corrected_histogram_bin_centers_half_sum[m][0], corrected_histogram_charge_half_sum[m][0], label=stringList[m])
    #axes[0].plot(calibrated_histogram_bin_centers_half_sum[m][0], calibrated_histogram_charge_half_sum[m][0],
    #             label=stringList[m] + ' calibrated')
    axes[0].text(.99, .99, 'ASIC 4', ha='right', va='top', transform=ax.transAxes)

    axes[1].plot(corrected_histogram_bin_centers_half_sum[m][1], corrected_histogram_charge_half_sum[m][1], label=stringList[m])
    #axes[1].plot(calibrated_histogram_bin_centers_half_sum[m][1], calibrated_histogram_charge_half_sum[m][1],
    #             label=stringList[m]+ ' calibrated')
    axes[1].text(.99, .99, 'ASIC 1', ha='right', va='top', transform=ax.transAxes)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(), loc='lower center',ncol=Nexp)




"""
plt.figure()
fig.suptitle('Corrected charge histogram, sum on full matrix')
for m in range(Nexp):
    plt.plot(corrected_histogram_bin_centers_full_sum[m],corrected_histogram_charge_full_sum[m]/np.max(corrected_histogram_charge_full_sum[m]), label=stringList[m])
plt.legend()
plt.xlabel('Sum ADCu')
plt.ylabel('Normalized Counts')
"""
"""
plt.figure()
fig.suptitle('Corrected charge histogram, average on full matrix')
min_adc = 1023
max_adc = 0
for m in range(Nexp):
    # calculate the limits of spectrum for a nice visualization
    min_adc = min(min_adc,np.min(corrected_histogram_bin_centers_full_avg[m][corrected_histogram_charge_full_avg[m]> 5]))
    max_adc = max(max_adc,np.max(corrected_histogram_bin_centers_full_avg[m][corrected_histogram_charge_full_avg[m]> 5]))
    plt.plot(corrected_histogram_bin_centers_full_avg[m],corrected_histogram_charge_full_avg[m], label=stringList[m])
plt.legend()
plt.xlabel('Avg. ADCu')
plt.ylabel('Counts')
plt.xlim([min_adc,max_adc])
"""
plt.figure()
fig.suptitle('Corrected charge histogram, sum on full matrix')
min_adc = 0
max_adc = 64*1023
for m in range(Nexp):
    # calculate the limits of spectrum for a nice visualization
    """
    min_adc = min(min_adc,np.min(corrected_histogram_bin_centers_full_sum[m][corrected_histogram_charge_full_sum[m]> 5]))
    max_adc = max(max_adc,np.max(corrected_histogram_bin_centers_full_sum[m][corrected_histogram_charge_full_sum[m]> 5]))
    """
    plt.plot(corrected_histogram_bin_centers_full_sum[m],corrected_histogram_charge_full_sum[m], label=stringList[m])
    if currTestHasDualBoard[m]:
        plt.plot(corrected_histogram_bin_centers_full_sum_coincidence[currTestCoincidenceIndex[m]][0],corrected_histogram_charge_full_sum_coincidence[currTestCoincidenceIndex[m]][0], label=stringList[m] + ' Coinc.')
        plt.plot(corrected_histogram_bin_centers_full_sum_coincidence[currTestCoincidenceIndex[m]][1],
                 corrected_histogram_charge_full_sum_coincidence[currTestCoincidenceIndex[m]][1],
                 label=stringList[m] + ' Coinc.')

plt.legend()
plt.xlabel('Sum ADCu')
plt.ylabel('Counts')
plt.xlim([min_adc,max_adc])


plt.figure()
fig.suptitle('Corrected charge histogram, sum on full matrix')
min_adc = 0
max_adc = 64*1023
for m in range(Nexp):
    # calculate the limits of spectrum for a nice visualization
    """
    min_adc = min(min_adc,np.min(corrected_histogram_bin_centers_full_sum[m][corrected_histogram_charge_full_sum[m]> 5]))
    max_adc = max(max_adc,np.max(corrected_histogram_bin_centers_full_sum[m][corrected_histogram_charge_full_sum[m]> 5]))
    """
    plt.plot(corrected_histogram_bin_centers_full_sum[m],corrected_histogram_charge_full_sum[m]/np.max(corrected_histogram_charge_full_sum[m]), label=stringList[m])
    if currTestHasDualBoard[m]:
        plt.plot(corrected_histogram_bin_centers_full_sum_coincidence[currTestCoincidenceIndex[m]][0],corrected_histogram_charge_full_sum_coincidence[currTestCoincidenceIndex[m]][0]/np.max(corrected_histogram_charge_full_sum_coincidence[currTestCoincidenceIndex[m]][0]), label=stringList[m] + ' Coinc.')
        plt.plot(corrected_histogram_bin_centers_full_sum_coincidence[currTestCoincidenceIndex[m]][1],
                 corrected_histogram_charge_full_sum_coincidence[currTestCoincidenceIndex[m]][1]/np.max(corrected_histogram_charge_full_sum_coincidence[currTestCoincidenceIndex[m]][1]),
                 label=stringList[m] + ' Coinc.')

plt.legend()
plt.xlabel('Sum ADCu')
plt.ylabel('Counts')
plt.xlim([min_adc,max_adc])


plt.figure()
fig.suptitle('Corrected charge histogram, sum on full matrix')
min_adc = 0
max_adc = 64*1023
for m in range(Nexp):
    # calculate the limits of spectrum for a nice visualization
    """
    min_adc = min(min_adc,np.min(corrected_histogram_bin_centers_full_sum[m][corrected_histogram_charge_full_sum[m]> 5]))
    max_adc = max(max_adc,np.max(corrected_histogram_bin_centers_full_sum[m][corrected_histogram_charge_full_sum[m]> 5]))
    """
    plt.plot(corrected_histogram_bin_centers_full_sum[m],corrected_histogram_charge_full_sum[m]/corrected_histogram_charge_full_sum[m].max(), label=stringList[m])
    #plt.plot(calibrated_histogram_bin_centers_full_sum[m],calibrated_histogram_charge_full_sum[m]/calibrated_histogram_charge_full_sum[m].max(), label=stringList[m] + ' calibrated')

plt.legend()
plt.xlabel('Sum ADCu')
plt.ylabel('Relative Counts')
plt.xlim([min_adc,max_adc])

binsPerSide = 8*64


for m in range(Nexp):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    fig.suptitle('Flood histogram ' + stringList[m])
    H, xedges, yedges = np.histogram2d(X_interactions[m], Y_interactions[m], bins=binsPerSide, range=[[-26.68 / 2, +26.68 / 2], [-26.68 / 2, +26.68 / 2]])
    im = axes.imshow(H, extent=[-26.68 / 2, +26.68 / 2, -26.68 / 2, +26.68 / 2], interpolation='nearest', origin='lower',cmap='inferno',norm=matplotlib.colors.PowerNorm(gamma=0.2))
    axes.set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
    axes.set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
    axes.grid()
    axes.set_title(stringList[m])
    axes.set_xlabel('[mm]')
    axes.set_ylabel('[mm]')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('Counts #')
"""
for m in range(Nexp):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    fig.suptitle('Flood histogram calibrated ' + stringList[m])
    H, xedges, yedges = np.histogram2d(X_interactions_Cal[m], Y_interactions_Cal[m], bins=binsPerSide,
                                       range=[[-26.68 / 2, +26.68 / 2], [-26.68 / 2, +26.68 / 2]])
    im = axes.imshow(H, extent=[-26.68 / 2, +26.68 / 2, -26.68 / 2, +26.68 / 2], interpolation='nearest',
                     origin='lower', cmap='inferno', norm=matplotlib.colors.PowerNorm(gamma=0.2))
    axes.set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
    axes.set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
    axes.grid()
    axes.set_title(stringList[m])
    axes.set_xlabel('[mm]')
    axes.set_ylabel('[mm]')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('Counts #')


for m in range(Nexp):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    fig.suptitle('Interaction Positions ' + stringList[m])
    axes.plot(X_interactions[m], Y_interactions[m],marker='.',linestyle='None',markersize=.5)
    axes.set_xticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
    axes.set_yticks(np.arange(9) * (3.16 + 0.2) - 26.7 / 2)
    axes.grid()
    axes.set_title(stringList[m])
    axes.set_xlabel('[mm]')
    axes.set_ylabel('[mm]')
"""





def Sgaussians(adc,*params):
    amp = params[0:(len(params) // 3)]
    mu = params[((len(params)//3)):(2 * (len(params)//3))]
    fwhm = params[(2*(len(params)//3)):(3 * (len(params)//3))]
    return (np.exp(-4*np.log(2)*((adc.reshape(-1, 1) - mu)/fwhm)**2)*amp).sum(1)

# make models for individual components
mod_lin = LinearModel(prefix='bkg_')
mod_gauss = GaussianModel(prefix='peak_')
model = mod_lin + mod_gauss


def select_callback(eclick, erelease,ax,plots,n_peaks,fit,table,fig):

    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    len_xmasked = np.zeros(len(plots))
    for m in range(len(plots)):
        if type(plots[m]) == list:
            plots[m] = plots[m][0]
        xdata = plots[m].get_xdata()
        ydata = plots[m].get_ydata()
        mask = (xdata > min(x1, x2)) & (xdata < max(x1, x2)) & \
               (ydata > min(y1, y2)) & (ydata < max(y1, y2))
        xmasked = xdata[mask]
        ymasked = ydata[mask]
        len_xmasked[m] = len(xmasked)
    if max(len_xmasked) > 0:
        index = np.argmax(len_xmasked)
        xdata = plots[index].get_xdata()
        ydata = plots[index].get_ydata()
        mask = (xdata > min(x1, x2)) & (xdata < max(x1, x2)) & \
               (ydata > min(y1, y2)) & (ydata < max(y1, y2))
        xmasked = xdata[mask]
        ymasked = ydata[mask]

        xmax = xmasked[np.argmax(ymasked)]
        ymax = ymasked.max()

        # guess parameters for linear fit
        # point A, left of gaussian
        x_a = min(xmasked)
        y_a = ymasked[xmasked == x_a]
        # point B, right of gaussian
        x_b = max(xmasked)
        y_b = ymasked[xmasked == x_b]

        m_est =  (y_b - y_a) / (x_b - x_a)
        q_est = y_b - m_est*x_b

        model.set_param_hint('peak_height',value=ymax,min=ymax*0.9,max=ymax*1.1)
        model.set_param_hint('peak_center', value=xmax, min=xmax * 0.8, max=xmax * 1.2)
        model.set_param_hint('peak_sigma', value=xmax* 0.05, min=xmax * 0.001, max=xmax*2)
        model.set_param_hint('bkg_slope', value=m_est, min=-1, max=+1)

        params = model.make_params(peak_amplitude=ymax, peak_center=xmax, peak_sigma=xmax * 0.05,
                                   bkg_intercept=q_est, bkg_slope=m_est)

        # do fit
        result = model.fit(ymasked, params, x=xmasked)

        # print out fitting statistics, best-fit parameters, uncertainties,....
        print(result.params)

        res_Sgauss_fit, _ = scipy.optimize.curve_fit(Sgaussians, xmasked, ymasked,  p0=[ymax, xmax, xmax * 0.05])
        if type(fit) == list:
            fit = fit[0]
        prev_xdata = fit.get_xdata()
        prev_ydata = fit.get_ydata()
        new_xdata = xmasked
        #new_ydata = Sgaussians(xmasked, *[res_Sgauss_fit[0], abs(res_Sgauss_fit[1]), res_Sgauss_fit[2]])
        new_ydata = result.best_fit
        fit.set_xdata(np.concatenate((prev_xdata,new_xdata)))
        fit.set_ydata(np.concatenate((prev_ydata,new_ydata)))

        n_peaks[0] += 1
        ax.text(xmax, ymax, "#"+str(n_peaks[0]))
        estimated_peak = result.params['peak_center'].value
        estimated_fwhm = result.params['peak_fwhm'].value
        estimated_fwhm_perc =estimated_fwhm/estimated_peak*100

        table.get_celld()[(n_peaks[0], 0)].get_text().set_text("{:.3f}".format(estimated_peak))
        table.get_celld()[(n_peaks[0] , 1)].get_text().set_text("{:.3f}".format(estimated_fwhm_perc))

        renderer = fig.canvas.renderer
        ax.draw(renderer)



def reset_spectra(event,fit,table,n_peaks,ax):


        print('Reset fits')
        n_peaks[0] = 0 # resent number of peaks
        for i in range( 1,6): # reset table
            table.get_celld()[(i, 0)].get_text().set_text("0")
            table.get_celld()[(i, 1)].get_text().set_text("0")
        # clean figure
        fit.set_xdata([])
        fit.set_ydata([])
        # clean labels
        for j in range(len(ax.texts)):
            Artist.remove(ax.texts[0])

def draw_spectra(event,tables,n_peaks,final_plots): #it needs to update the three curves


        print('Updated Spectra')
        # collect data from tables
        adc_peaks = []
        kev_energies = []
        for m in range(len(tables)):
            for n in range(n_peaks[m][0]):
                adc_peaks.append(float(tables[m].get_celld()[(n+1, 0)].get_text().get_text()))
                kev_energies.append(float(tables[m].get_celld()[(n+1, 2)].get_text().get_text()))
        print("ADC peaks : " + str(adc_peaks))
        print("kEv energies : " + str(kev_energies))
        M, A = np.polyfit(np.asarray(adc_peaks), np.asarray(kev_energies), 1)
        print("Coefficients for ADCu-kEv conversions of type kev=A + M*adc are:")
        print("M : " + str(M))
        print("A : " + str(A))
        for m in range(len(final_plots)): # convert every scale to Kev
            adc_data = final_plots[m][0].get_xdata()
            kev_data = adc_data*M + A
            final_plots[m][0].set_xdata(kev_data)




selectors = []
final_selectors = []
fit = [[] for i in range(Nexp)]

final_plots = [[] for i in range(Nexp)]
peak_tables = [[] for i in range(Nexp)]
editables_tables = [[] for i in range(Nexp)]
N_peaks = [[0] for i in range(Nexp)]
final_peaks = [[0]]
mode = True
spectra_to_be_fit_figures = [[] for m in range(Nexp)]
final_spectra, (final_ax, final_table_ax) =  plt.subplots(nrows=2)
final_table_ax.axis("off")
final_table = final_table_ax.table(cellText=[[0, 0] for i in range(5)],colLabels=('Peak kEv','FWHM% kEv'), rowLabels=('#1','#2','#3','#4','#5'), cellLoc='center', loc='best')
column_labels = ('Peak ADCu', 'FWHM% ADCu', 'Peak kEv')
row_labels = ['#1', '#2', '#3', '#4', '#5']

final_ax.grid(which='both')
final_ax.set_xlim([0,2000])
final_ax.set_ylabel('Counts')
final_ax.set_xlabel('Energy kEv')



final_figures =[ []for m in range(Nexp)]
final_tables = [[]for m in range(Nexp)]
final_fits= [[]for m in range(Nexp)]
final_peaks = [[0] for i in range(Nexp)]
final_plots_individuals  = [[] for i in range(Nexp)]
for m in range(Nexp):
    final_figures[m] = plt.subplots(nrows=2)
    final_figures[m][1][0].grid(which='both')
    final_figures[m][1][0].set_xlim([0,2000])
    final_figures[m][1][0].set_ylabel('Counts')
    final_figures[m][1][0].set_xlabel('Energy kEv')
    final_figures[m][1][0].set_title(stringList[m])
    final_figures[m][1][1].axis("off")
    final_tables[m] = final_figures[m][1][1].table(cellText=[[0, 0] for i in range(5)],colLabels=('Peak kEv','FWHM% kEv'), rowLabels=('#1','#2','#3','#4','#5'), cellLoc='center', loc='best')
    final_fits[m] = final_figures[m][1][0].plot([],[],label='Gauss fit',color='r',linestyle='--')

peak_labels = [[] for i in range(Nexp)]
update_buttons = [[] for i in range(Nexp)]
reset_buttons = [[] for i in range(Nexp)]
reset_buttons_finals = [[] for i in range(Nexp)]
final_peak_labels = []
final_fit = final_ax.plot([],[],label='Gauss fit',color='r',linestyle='--')
for m in range(Nexp):
    spectra_to_be_fit_figures[m], (ax, tabax) = plt.subplots(nrows=2)
    xdata = corrected_histogram_bin_centers_full_sum[m]
    ydata = corrected_histogram_charge_full_sum[m]
    curr_plot = ax.plot(xdata, ydata,label='Spectra')  # plot spectra
    final_plots[m] = final_ax.plot(xdata, ydata,label=stringList[m])
    final_plots_individuals[m] = final_figures[m][1][0].plot(xdata, ydata,label=stringList[m])
    fit[m] = ax.plot([],[],label='Gauss fit',color='r',linestyle='--')
    selector_class = RectangleSelector
    ax.set_xlim([min_adc,max_adc])
    ax.set_title(stringList[m])
    ax.legend()
    tabax.axis("off")
    peak_tables[m] = tabax.table(cellText=[[0, 0, 0] for i in range(5)], colLabels=column_labels, rowLabels=row_labels,
                 cellLoc='center', loc='best')
    ax_reset_button = spectra_to_be_fit_figures[m].add_axes([0.7, 0.05, 0.1, 0.075])
    ax_update_button = spectra_to_be_fit_figures[m].add_axes([0.81, 0.05, 0.1, 0.075])
    update_buttons[m] = Button(ax_update_button, 'Update')
    update_buttons[m].on_clicked(partial(draw_spectra,tables=peak_tables,n_peaks=N_peaks,final_plots= final_plots))
    update_buttons[m].on_clicked(partial(draw_spectra, tables=peak_tables, n_peaks=N_peaks, final_plots=final_plots_individuals))
    reset_buttons[m] = Button(ax_reset_button, 'Reset')
    reset_buttons[m].on_clicked(partial(reset_spectra,fit=fit[m][0],table=peak_tables[m] ,ax=ax,n_peaks=N_peaks[m]))

    editables_tables[m] = EditableTable(peak_tables[m])
    selectors.append(selector_class(ax, partial(select_callback,plots=curr_plot,fig=spectra_to_be_fit_figures[m],ax=ax,fit=fit[m][0]
                                                ,n_peaks=N_peaks[m],table=peak_tables[m] ),useblit=True,button=[1, 3],spancoords='pixels',interactive=True))
    final_selectors.append(selector_class( final_figures[m][1][0], partial(select_callback,plots=final_plots_individuals[m],fig=final_figures[m][0],ax=final_figures[m][1][0],fit=final_fits[m][0]
                                                ,n_peaks=final_peaks[m],table=final_tables[m] ),useblit=True,button=[1, 3],spancoords='pixels',interactive=True))

final_selector = selector_class(final_ax,partial(select_callback,ax=final_ax,plots=final_plots,fit=final_fit,
                                                n_peaks=final_peaks[0],table=final_table,fig=final_spectra),useblit=True,button=[1, 3],spancoords='pixels',interactive=True)
final_ax.legend()




ax_reset_button = final_spectra.add_axes([0.7, 0.05, 0.1, 0.075])
reset_button = Button(ax_reset_button, 'Reset')
reset_button.on_clicked(partial(reset_spectra,fit=final_fit[0],ax=final_ax,table=final_table ,n_peaks=final_peaks[0]))

plt.show()