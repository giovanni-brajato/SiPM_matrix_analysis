import itertools
import os
import csv
import numpy as np
import pandas as pd
from distinctipy import distinctipy
import matplotlib
import scipy
import re
from matplotlib.ticker import FuncFormatter

import sklearn
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler


def format_binary(value, _):
    #binary_representation = format(int(value), '0:010b')
    return '{0:10b}'.format(value)
def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


#from __future__ import print_function
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





def select_callback(eclick, erelease,ax,plots,n_peaks,fit,fig):

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

        # make models for individual components
        mod_lin = LinearModel(prefix='bkg_')
        mod_gauss = GaussianModel(prefix='peak_')
        model_temp = mod_gauss
        model_temp.set_param_hint('peak_height',value=ymax,min=ymax*0.9,max=ymax*1.1)
        model_temp.set_param_hint('peak_center', value=xmax, min=xmax * 0.8, max=xmax * 1.2)
        model_temp.set_param_hint('peak_sigma', value=xmax * 0.05, min=0.05, max=1)
        #model.set_param_hint('bkg_slope', value=m_est, min=-1, max=0)

        params = model_temp.make_params(peak_amplitude=ymax, peak_center=xmax, peak_sigma=xmax * 0.05,
                                   bkg_intercept=q_est, bkg_slope=m_est)

        # do fit
        result = model_temp.fit(ymasked, params, x=xmasked)

        # print out fitting statistics, best-fit parameters, uncertainties,....
        print(result.params)

        #res_Sgauss_fit, _ = scipy.optimize.curve_fit(Sgaussians, xmasked, ymasked,  p0=[ymax, xmax, xmax * 0.05])
        if type(fit) == list:
            fit = fit[0]
        prev_xdata = fit.get_xdata()
        prev_ydata = fit.get_ydata()
        new_xdata = xmasked
        #new_ydata = Sgaussians(xmasked, *[res_Sgauss_fit[0], abs(res_Sgauss_fit[1]), res_Sgauss_fit[2]])
        new_ydata = result.best_fit
        fit.set_xdata(np.concatenate((prev_xdata,new_xdata)))
        fit.set_ydata(np.concatenate((prev_ydata,new_ydata)))
        estimated_peak = result.params['peak_center'].value
        estimated_fwhm = result.params['peak_fwhm'].value
        n_peaks[0] += 1
        ax.text(xmax, ymax, "#"+str(n_peaks[0]) + " FWHM="+ f'{estimated_fwhm*1e3:.3f}'+ " ps\ncenter=" + f'{estimated_peak:.3f}'+ " ns")

        estimated_fwhm_perc =estimated_fwhm/estimated_peak*100



        renderer = fig.canvas.renderer
        ax.draw(renderer)



def reset_spectra(event,fit,table,n_peaks,ax):


        print('Reset fits')
        n_peaks[0] = 0 # resent number of peaks
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





fileList = []
stringList = []
cwd = os.getcwd()
os.chdir(cwd)
startAsic = []
startChannels = []
stopAsic = []
stopChannels = []
globalDelay = []
board_id_list = []
tdc_data_list = []
calibration_data_index = np.ones((5,4,32),dtype=int)*-1
calibration_data = []
c_index = 0
for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith(".csv"):
            if file == 'ParamScanSet.csv':
                param_data = pd.read_csv(root+'\\'+file, sep=',')
            elif file.endswith("TDC_calibration_data.csv"):
                IDs = file.split('_')
                board_id = int(IDs[0][-1])
                asic_id = int(IDs[1][-1])
                channel_id = int(IDs[2][1:])
                calibration_data_index[board_id,asic_id,channel_id] = c_index
                c_index +=1
                # read csv file
                tdc_cal_data = np.loadtxt(root + '\\' + file,delimiter=',')
                # memorize matrix into the list
                calibration_data.append(tdc_cal_data)
            else:
                board = int(file.split('_')[-1][2])
                board_id_list.append(board)
                tdc_data_list.append(pd.read_csv(root+'\\'+file, sep=','))

N_boards = len(board_id_list)
total_curves = 0

asic_channels_board_list = []
possible_channels_board_list = []
total_points = np.Inf

# identification
Ch_list = []
asic_list = []
board_list = []

# data
ST_vector_list = []
FT_vector_list = []
CT_vector_list = []
RT_vector_list = []

N_comparisons = 0

b = 0
for tdc_data in tdc_data_list:
    # find the channels
    ct_values = []
    ft_values = []
    possible_boards = []
    possible_channels = []
    # Define a regular expression pattern to match numbers
    number_pattern = re.compile(r'\d+')
    set_delay = param_data.values[:, :]
    for index, entry in enumerate(tdc_data.columns.values, start=0) :
        elements = entry.split('_')
        if len(elements) > 1:
            channel_str = elements[0]
            channel_ind = int(number_pattern.findall(channel_str)[0])
            data_type = elements[1]
            possible_channels.append(channel_ind)
            if data_type == 'CT':
                ct_values.append(tdc_data.values[:, index].astype(int))
            elif data_type == 'FT':
                ft_values.append(tdc_data.values[:, index].astype(int))
        else:
            if elements[0] == 'SET #':
                set_number = tdc_data.values[:, index].astype(int)
            elif elements[0] == 'ID':
                petiroc_id = tdc_data.values[:, index].astype(int)
            elif elements[0] == 'RT':
                rt_values = tdc_data.values[:, index].astype(int)
    possible_channels = np.unique(np.asarray(possible_channels))
    possible_asics = np.unique(np.asarray((petiroc_id)))
    ft_values = np.asarray(ft_values)
    ct_values = np.asarray(ct_values)
    valid_values = ft_values != 4
    for a in range(len(possible_asics)):
        for c in range(len(possible_channels)):

            ft_values_filtered = ft_values[c,valid_values[c,:]*(petiroc_id == possible_asics[a])]
            ct_values_filtered = ct_values[c,valid_values[c,:]*(petiroc_id == possible_asics[a])]
            rt_values_filtered = rt_values[petiroc_id == possible_asics[a]]
            # st_values_filtered = np.diff(rt_values_filtered) > 0 # not now but maybe in future
            if len(ft_values_filtered) > 1:
                # Identification
                Ch_list.append(possible_channels[c])
                asic_list.append(possible_asics[a])
                board_list.append(board_id_list[b])

                # data
                #ST_vector_list.append()
                FT_vector_list.append(ft_values_filtered)
                CT_vector_list.append(ct_values_filtered)
                RT_vector_list.append(rt_values_filtered)
    b += 1

N_exp = len(board_list)
N_comparisons = int(N_exp*(N_exp-1)/2)
index_list = [] # we need to build it for more elements
for pair in itertools.combinations(np.arange(N_exp),2):
    index_list.append(pair)


#limiting comparisons to 16
N_comparisons = min(16,N_comparisons)

FTtimeConst = 37e-12
CTtimeConst = 25e-9
RTtimeConst = 12.775e-6
STtimeConst = ((2**30)-1)*RTtimeConst

selectors = []
selectors_cal = []
fit = [[] for i in range(N_comparisons)]
fit_cal = [[] for i in range(N_comparisons)]

final_plots = [[] for i in range(N_comparisons)]
peak_tables = [[] for i in range(N_comparisons)]
editables_tables = [[] for i in range(N_comparisons)]
N_peaks = [[0] for i in range(N_comparisons)]
final_peaks = [[0]]
mode = True
final_plots_cal = [[] for i in range(N_comparisons)]
peak_tables_cal = [[] for i in range(N_comparisons)]
editables_tables_cal = [[] for i in range(N_comparisons)]
N_peaks_cal = [[0] for i in range(N_comparisons)]
final_peaks_cal = [[0]]



spectra_to_be_fit_figures = [[] for m in range(N_comparisons)]
spectra_to_be_fit_figures_cal = [[] for m in range(N_comparisons)]

peak_labels = [[] for i in range(N_comparisons)]
update_buttons = [[] for i in range(N_comparisons)]
reset_buttons = [[] for i in range(N_comparisons)]
final_peak_labels = []
peak_labels_cal = [[] for i in range(N_comparisons)]
update_buttons_cal = [[] for i in range(N_comparisons)]
reset_buttons_cal = [[] for i in range(N_comparisons)]
final_peak_labels_cal = []
#final_fit = final_ax.plot([],[],label='Gauss fit',color='r',linestyle='None',marker='x')



for ne in range(N_comparisons):
    index_a, index_b = index_list[ne]
    # looking for frame match
    RT_a = RT_vector_list[index_a]
    RT_b = RT_vector_list[index_b]
    FT_a_sorted = []
    FT_b_sorted = []
    RT_sorted = []
    CT_a_sorted = []
    CT_b_sorted = []
    for rt_ind in range(len(RT_a)):
        match_index = np.argwhere(RT_a[rt_ind] == RT_b)
        if len(match_index) > 0: # match found
            RT_sorted.append(RT_a[rt_ind])
            FT_a_sorted.append(FT_vector_list[index_a][rt_ind])
            FT_b_sorted.append(FT_vector_list[index_b][match_index[0]])
            CT_a_sorted.append(CT_vector_list[index_a][rt_ind])
            CT_b_sorted.append(CT_vector_list[index_b][match_index[0]])
    same_CT_mask = (np.asarray(CT_a_sorted).squeeze() - np.asarray(CT_b_sorted).squeeze()) == 0
    deltaT_a_sorted = CTtimeConst * np.mod(np.asarray(CT_a_sorted).squeeze() - np.asarray(CT_b_sorted).squeeze(), 512) - FTtimeConst * (np.asarray(FT_a_sorted).squeeze() - np.asarray(FT_b_sorted).squeeze())

    AT_a_sorted = CTtimeConst*np.mod((np.asarray(CT_a_sorted).squeeze() +1),512) - FTtimeConst * (np.asarray(FT_a_sorted).squeeze())
    AT_b_sorted = CTtimeConst * np.mod((np.asarray(CT_b_sorted).squeeze() + 1),512) - FTtimeConst * (
        np.asarray(FT_b_sorted).squeeze())


    AT_a_sorted_rescaled = 24.821331570243144e-9 * np.mod((np.asarray(CT_a_sorted).squeeze() + 1), 512) - FTtimeConst * (
        np.asarray(FT_a_sorted).squeeze())
    AT_b_sorted_rescaled = 24.821331570243144e-9 * np.mod((np.asarray(CT_b_sorted).squeeze() + 1), 512) - FTtimeConst * (
        np.asarray(FT_b_sorted).squeeze())

    #deltaT_a_sorted = AT_a_sorted - AT_b_sorted # alternative way to calculate deltaT, should be equivalent to the previous one
    # selecting final histogram and perform a fit
    ToFstep = 37e-12
    ToF = deltaT_a_sorted
    ToF_sameCT = deltaT_a_sorted[same_CT_mask]


    binEdgesToF_sameCT = np.arange(ToF_sameCT.min(), ToF_sameCT.max() + 2 * ToFstep, ToFstep) - ToFstep / 2

    ToFhist_sameCT, binEdgesToF_sameCT = np.histogram(ToF_sameCT, bins=np.arange(ToF.min(), ToF.max() + 2 * ToFstep, ToFstep) - ToFstep / 2)

    ToFhist, binEdgesToF = np.histogram(ToF, bins=np.arange(ToF.min(), ToF.max() + 2 * ToFstep, ToFstep) - ToFstep / 2)

    # check if there is calibration data, in affirmative case, attempt to perform a crt calculation using the calibration
    board_id_a = board_list[index_a]
    board_id_b = board_list[index_b]
    asic_id_a = asic_list[index_a]
    asic_id_b = asic_list[index_b]
    channel_id_a = Ch_list[index_a]
    channel_id_b = Ch_list[index_b]

    cal_index_a = calibration_data_index[board_id_a,asic_id_a,channel_id_a]
    cal_index_b = calibration_data_index[board_id_b,asic_id_b,channel_id_b]

    if cal_index_a != -1 and cal_index_b != -1:
        # access calibration matrices
        calibration_data_a = calibration_data[cal_index_a]
        calibration_data_b = calibration_data[cal_index_b]

        #extra step: Reduce coarse time into minimum value
        CT_ab_differences = np.mod(np.asarray(CT_a_sorted).squeeze() - np.asarray(CT_b_sorted).squeeze(),512)
        CT_a_sorted_relative = np.zeros(len(CT_a_sorted),dtype=int)
        CT_b_sorted_relative = np.mod(CT_a_sorted_relative + CT_ab_differences,512)

        # third calibration attempt: re-estimate period
        # extract period
        period_a = np.diff(calibration_data_a, axis=0).mean()
        period_b = np.diff(calibration_data_b, axis=0).mean()
        # remove the period
        calibration_data_a_without_period = (calibration_data_a.T - period_a * np.arange(512)).T
        calibration_data_b_without_period = (calibration_data_b.T - period_b * np.arange(512)).T
        # convert coarse time and rough time into absolute time values
        #AT_a_calibrated = calibration_data_a[np.asarray(CT_a_sorted).squeeze(),np.asarray(FT_a_sorted).squeeze()]*1e-9
        #AT_b_calibrated = calibration_data_b[np.asarray(CT_b_sorted).squeeze(),np.asarray(FT_b_sorted).squeeze()]*1e-9
        optimal_periods = [25, 25]
        CD_a_temp = (calibration_data_a_without_period.T + optimal_periods[0] * np.arange(512)).T
        CD_b_temp = (calibration_data_b_without_period.T + optimal_periods[1] * np.arange(512)).T
        AT_a_temp = CD_a_temp[np.asarray(CT_a_sorted).squeeze(), np.asarray(FT_a_sorted).squeeze()] * 1e-9
        AT_b_temp = CD_b_temp[np.asarray(CT_b_sorted).squeeze(), np.asarray(FT_b_sorted).squeeze()] * 1e-9
        AT_a_temp = (CD_a_temp[0, np.asarray(FT_a_sorted).squeeze()] + optimal_periods[0] * np.asarray(
            CT_a_sorted).squeeze()) * 1e-9
        AT_b_temp = (CD_b_temp[0, np.asarray(FT_b_sorted).squeeze()] + optimal_periods[1] * np.asarray(
            CT_b_sorted).squeeze()) * 1e-9

        x = AT_a_temp
        y = AT_a_temp - AT_b_temp
        AT_a_calibrated = AT_a_temp
        AT_b_calibrated = AT_b_temp



        delta_AT_calibrated = (AT_a_calibrated - AT_b_calibrated)
        delta_at_calibrated_mask = np.abs(delta_AT_calibrated) < 10e-9
        ToFhistCal, binEdgesToFCal = np.histogram(delta_AT_calibrated[delta_at_calibrated_mask], bins=1000)

        AT_a_calibrated_alt = calibration_data_a[CT_a_sorted_relative,np.asarray(FT_a_sorted).squeeze()]*1e-9
        AT_b_calibrated_alt = calibration_data_a[CT_b_sorted_relative, np.asarray(FT_b_sorted).squeeze()]*1e-9





        """
        def error(periods):
            CD_a_temp = (calibration_data_a_without_period.T + periods[0] * np.arange(512)).T
            CD_b_temp = (calibration_data_b_without_period.T + periods[1] * np.arange(512)).T
            AT_a_temp = CD_a_temp[np.asarray(CT_a_sorted).squeeze(),np.asarray(FT_a_sorted).squeeze()]*1e-9
            AT_b_temp = CD_b_temp[np.asarray(CT_b_sorted).squeeze(), np.asarray(FT_b_sorted).squeeze()]*1e-9
            x = AT_a_temp
            y = AT_a_temp - AT_b_temp
            a, b = np.polyfit(x, y, 1)
            return a


        optimal_periods= scipy.optimize.minimize(error,x0=np.array([period_a,period_b]))
        """

        #optimal_periods = [24.921331570243144,24.91942604578018]


        # standardize
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        x_train = x_scaler.fit_transform(x[..., None])
        y_train = y_scaler.fit_transform(y[..., None])
        """
        # fit model
        model = HuberRegressor(epsilon=1)
        model.fit(x_train, y_train.ravel())
        # do some predictions
        test_x = np.array([min(x), max(x)]).reshape(-1, 1)
        predictions = y_scaler.inverse_transform(
            model.predict(x_scaler.transform(test_x)).reshape(-1, 1)
        )
        m_better = np.diff(predictions.squeeze())/np.diff(test_x.squeeze())

        m, q = np.polyfit(x, y, 1)
        
        fig, ax = plt.subplots(nrows=2,sharex=True)
        #ax[0].scatter(AT_a_calibrated_alt, AT_a_calibrated_alt - AT_b_calibrated_alt, label='Calibrated v2', s=2)
        ax[0].scatter(AT_a_calibrated, AT_a_calibrated - AT_b_calibrated, label='Calibrated', s=2)
        ax[0].scatter(AT_a_sorted, AT_a_sorted - AT_b_sorted, label='Conventional', s=2)
        #ax[0].scatter(AT_a_sorted_rescaled, AT_a_sorted_rescaled - AT_b_sorted_rescaled, label='Conventional rescaled', s=2)
        #ax[0].scatter(x, y,label='Calibrated v3', s=2)
        ax[0].plot(test_x, predictions, label='Better fitted line, m = '+str(m_better))
        ax[1].scatter(AT_a_sorted,np.asarray(CT_a_sorted).squeeze(),label='Coarse time A',s=2)
        ax[1].scatter(AT_a_sorted, np.asarray(CT_b_sorted).squeeze(), label='Coarse time B',s=2)
        ax[1].set_xlabel('Arrival time A')
        ax[0].set_xlabel('Arrival time A')
        ax[0].set_ylabel('Arrival time A - Arrival time B')
        ax[0].legend()
        ax[1].legend()
        """


        """
        plt.figure()
        plt.scatter(AT_a_calibrated_alt,AT_b_calibrated_alt,label='Calibrated v2',s=2)
        plt.scatter(AT_a_calibrated, AT_b_calibrated, label='Calibrated',s=2)
        plt.scatter(AT_a_sorted, AT_b_sorted, label='Conventional',s=2)
        plt.scatter(x, y, label='Calibrated v3', s=2)
        plt.xlabel('Arrival time A')
        plt.ylabel('Arrival time B')
        plt.legend()
        """


        delta_AT_calibrated_alt = (AT_a_calibrated_alt - AT_b_calibrated_alt)
        delta_at_calibrated_alt_mask = np.abs(delta_AT_calibrated_alt) < 10e-9
        ToFhistCalAlt, binEdgesToFCalAlt = np.histogram(delta_AT_calibrated_alt[delta_at_calibrated_alt_mask], bins=1000)

    spectra_to_be_fit_figures[ne], ax = plt.subplots(nrows=1)
    xdata = (binEdgesToF[:-1] + binEdgesToF[1:])/2
    ydata = ToFhist
    curr_plot = ax.plot(xdata / 1e-9, ydata,linestyle='None')
    ax.stairs(ToFhist, binEdgesToF / 1e-9,
              label='DT' + str(board_list[index_a]) + 'A' + str(asic_list[index_a]) + 'C' + str(
                  Ch_list[index_a]) + ' vs DT' + str(board_list[index_b]) + 'A' + str(asic_list[index_b]) + 'C' + str(
                  Ch_list[index_b]) + ' , all events CT')
    ax.stairs(ToFhist_sameCT, binEdgesToF_sameCT / 1e-9,
               label='DT' + str(board_list[index_a]) + 'A' + str(asic_list[index_a]) + 'C' + str(
                   Ch_list[index_a]) + ' vs DT' + str(board_list[index_b]) + 'A' + str(asic_list[index_b]) + 'C' + str(
                   Ch_list[index_b]) + ' , same CT')
    # final_plots[ne] = final_ax.plot(xdata, ydata, label=stringList[m])
    fit[ne] = ax.plot([], [], label='Gauss fit', color='r', linestyle='--', marker='None')
    selector_class = RectangleSelector
    ax.set_xlabel("delay [ns]")
    ax.set_ylabel("counts")
    ax.legend()
    """
    ax_reset_button = spectra_to_be_fit_figures[ne].add_axes([0.7, 0.05, 0.1, 0.075])
    reset_buttons[ne] = Button(ax_reset_button, 'Reset')
    reset_buttons[ne].on_clicked(
        partial(reset_spectra, fit=fit[ne][0], table=peak_tables[ne], ax=ax, n_peaks=N_peaks[ne]))
    """
    # editables_tables[m] = EditableTable(peak_tables[m])
    selectors.append(selector_class(ax,
                                    partial(select_callback, plots=curr_plot, fig=spectra_to_be_fit_figures[ne], ax=ax,
                                            fit=fit[ne][0]
                                            , n_peaks=N_peaks[ne]), useblit=True, button=[1, 3],
                                    spancoords='pixels', interactive=True))

    if cal_index_a != -1 and cal_index_b != -1:
        spectra_to_be_fit_figures_cal[ne], ax_cal = plt.subplots(nrows=1)
        xdata_cal = (binEdgesToFCal[:-1] + binEdgesToFCal[1:]) / 2
        ydata_cal = ToFhistCal
        curr_plot_cal = ax_cal.plot(xdata_cal / 1e-9, ydata_cal, linestyle='None')
        ax_cal.stairs(ToFhistCal, binEdgesToFCal / 1e-9,
                  label='DT' + str(board_list[index_a]) + 'A' + str(asic_list[index_a]) + 'C' + str(
                      Ch_list[index_a]) + ' vs DT' + str(board_list[index_b]) + 'A' + str(asic_list[index_b]) + 'C' + str(
                      Ch_list[index_b]) + ' , Calibrated')
        # final_plots[ne] = final_ax.plot(xdata, ydata, label=stringList[m])
        fit_cal[ne] = ax_cal.plot([], [], label='Gauss fit', color='r', linestyle='--', marker='None')
        selector_class_cal = RectangleSelector
        ax_cal.set_xlabel("delay [ns]")
        ax_cal.set_ylabel("counts")
        ax_cal.legend()
        """
        ax_reset_button_cal = spectra_to_be_fit_figures_cal[ne].add_axes([0.7, 0.05, 0.1, 0.075])
        reset_buttons_cal[ne] = Button(ax_reset_button_cal, 'Reset')
        reset_buttons_cal[ne].on_clicked(
            partial(reset_spectra, fit=fit_cal[ne][0], table=peak_tables_cal[ne], ax=ax_cal, n_peaks=N_peaks_cal[ne]))
        """
        # editables_tables[m] = EditableTable(peak_tables[m])
        selectors_cal.append(selector_class_cal(ax_cal,
                                        partial(select_callback, plots=curr_plot_cal, fig=spectra_to_be_fit_figures_cal[ne],
                                                ax=ax_cal,
                                                fit=fit_cal[ne][0]
                                                , n_peaks=N_peaks_cal[ne]), useblit=True, button=[1, 3],
                                        spancoords='pixels', interactive=True))

        """
        ax.stairs(ToFhistCalAlt, binEdgesToFCalAlt / 1e-9,
                  label='DT' + str(board_list[index_a]) + 'A' + str(asic_list[index_a]) + 'C' + str(
                      Ch_list[index_a]) + ' vs DT' + str(board_list[index_b]) + 'A' + str(
                      asic_list[index_b]) + 'C' + str(
                      Ch_list[index_b]) + ' , CalibratedV2, mean = ' + str(ToFhistCalAlt.mean() / 1e-9) + ' ns')
      """







plt.show()