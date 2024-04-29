import os
import csv
import numpy as np
import pandas as pd
from distinctipy import distinctipy
import matplotlib
import scipy
import re
import random

from matplotlib.ticker import FuncFormatter
def format_binary(value, _):
    #binary_representation = format(int(value), '0:010b')
    return '{0:10b}'.format(value)


matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

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


fileList = []
stringList = []
cwd = os.getcwd()
os.chdir(cwd)
startAsic = []
startChannels = []
stopAsic = []
stopChannels = []
globalDelay = []
board_list = []
tdc_data_list = []
for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith(".csv"):
            if file == 'ParamScanSet.csv':
                param_data = pd.read_csv(root+'\\'+file, sep=',')
            elif file.startswith('DataSet'):
                board = int(file.split('_')[-1][2])
                board_list.append(board)
                tdc_data_list.append(pd.read_csv(root+'\\'+file, sep=','))

N_boards = len(board_list)
total_curves = 0
TDC_curves_mean_list = []
CT_curves_mean_list = []
TDC_curves_std_list = []
CT_curves_std_list = []
TDC_test_data_list = []
CT_test_data_list = []

asic_channels_board_list = []
possible_channels_board_list = []
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
    possible_channels = np.asarray(possible_channels)
    ft_values = np.asarray(ft_values)
    ct_values = np.asarray(ct_values)
    valid_values = ft_values != 4


    possible_asics = petiroc_id[0:len(np.unique(petiroc_id))]
    N_time_points = int(len(petiroc_id)/len(possible_asics))
    possible_channels = np.unique(possible_channels)
    aligned_rt_values = np.zeros((len(possible_asics),N_time_points))
    aligned_ft_values = np.zeros((len(possible_asics), len(possible_channels),N_time_points))
    aligned_ct_values = np.zeros((len(possible_asics), len(possible_channels),N_time_points))
    aligned_set_values = np.zeros((len(possible_asics),N_time_points))


    asic_ch_list = []

    valid_channel_per_asic_mask = np.zeros((len(possible_asics),len(possible_channels))).astype(bool)

    for an,a in enumerate(np.sort(possible_asics),start=0):
        asic_ch_list.append((petiroc_id[an],possible_channels[valid_values[:,petiroc_id == a].prod(1).astype(bool)]))
        valid_channel_per_asic_mask[an,:] = valid_values[:,petiroc_id == a].prod(1).astype(bool)
        aligned_rt_values[an,:] = (rt_values[petiroc_id == a])
        aligned_ft_values[an, :,:] = (ft_values[:, petiroc_id == a])
        aligned_ct_values[an, :,:] = (ct_values[:, petiroc_id == a])
        aligned_set_values[an,:] = (set_number[petiroc_id == a])



    n_points = len(set_delay)
    TDC_curves_mean = np.zeros((len(possible_asics),len(possible_channels),n_points))
    CT_curves_mean = np.zeros((len(possible_asics),len(possible_channels),n_points))
    TDC_curves_std = np.zeros((len(possible_asics),len(possible_channels),n_points))
    CT_curves_std = np.zeros((len(possible_asics),len(possible_channels),n_points))

    _,measures_per_set = np.unique(aligned_set_values[0,:],return_counts=True)
    TDC_test_data = np.zeros((len(possible_asics),len(possible_channels),n_points,measures_per_set[0]))
    CT_test_data = np.zeros((len(possible_asics),len(possible_channels),n_points,measures_per_set[0]))


    for s in range(n_points):
        ft_masked = aligned_ft_values[:,:,s*measures_per_set[0]:(s+1)*measures_per_set[0]]
        ct_masked = aligned_ct_values[:,:, s*measures_per_set[0]:(s+1)*measures_per_set[0]]
        TDC_curves_mean[:,:,s] = ft_masked.mean(2)
        TDC_curves_std[:,:,s] = ft_masked.std(2)
        CT_curves_mean[:,:, s] = ct_masked.mean(2)
        CT_curves_std[:,:, s] = ct_masked.std(2)
        TDC_test_data[:,:, s,:] = ft_masked
        CT_test_data[:,:, s,:] = ct_masked

    #plt.close('all')



    colors = distinctipy.get_colors(len(possible_asics)*len(possible_channels),rng=1)

    FTtimeConst = 37e-12
    CTtimeConst = 25e-9
    RTtimeConst = 12.775e-6
    STtimeConst = 60.000010325

    n = 0
    for a in range(len(possible_asics)):
        for cin,c in enumerate(asic_ch_list[a][1],start=0):
            fig, axs = plt.subplots(2,sharex=True)
            fig.suptitle('Asic '+ str(asic_ch_list[a][0]) +' Channel '+ str(c))
            ch_ind = np.where(possible_channels == c)
            axs[0].plot(set_delay[:,1],CT_curves_mean[a,ch_ind,:].squeeze(),label='TDC counter', color = colors[n])
            axs[0].fill_between(set_delay[:,1], CT_curves_mean[a,ch_ind,:].squeeze()-CT_curves_std[a,ch_ind,:].squeeze(),  CT_curves_mean[a,ch_ind,:].squeeze()+CT_curves_std[a,ch_ind,:].squeeze(), color = colors[n],alpha=0.5)
            axs[1].plot(set_delay[:,1],TDC_curves_mean[a,ch_ind,:].squeeze(),label='TDC ramp', color = colors[n])
            axs[1].fill_between(set_delay[:,1], TDC_curves_mean[a,ch_ind,:].squeeze()-TDC_curves_std[a,ch_ind,:].squeeze(),  TDC_curves_mean[a,ch_ind,:].squeeze()+TDC_curves_std[a,ch_ind,:].squeeze(), color = colors[n],alpha=0.5)
            axs[1].set_xlabel('Delay [ns]')
            axs[1].set_ylabel('TDC units')
            axs[0].set_ylabel('CT units')
            n += 1

    n = 0
    fig, axs = plt.subplots(2,sharex=True)
    fig.suptitle('TDC offset comparison')
    for a in range(len(possible_asics)):
        for cin, c in enumerate(asic_ch_list[a][1], start=0):
            ch_ind = np.where(possible_channels == c)
            axs[0].plot(set_delay[:, 1], CT_curves_mean[a,ch_ind, :].squeeze(), label='TDC counter A'+str(asic_ch_list[a][0]) + ' Ch'+str(c), color=colors[n])
            axs[0].fill_between(set_delay[:, 1], CT_curves_mean[a,ch_ind, :].squeeze() - CT_curves_std[a,ch_ind, :].squeeze(),
                                CT_curves_mean[a,ch_ind, :].squeeze() + CT_curves_std[a,ch_ind, :].squeeze(), color=colors[a], alpha=0.5)
            axs[1].plot(set_delay[:, 1], TDC_curves_mean[a,ch_ind, :].squeeze(), label='TDC ramp A'+str(asic_ch_list[a][0]) + ' Ch'+str(c), color=colors[n])
            axs[1].fill_between(set_delay[:, 1], TDC_curves_mean[a,ch_ind, :].squeeze() - TDC_curves_std[a,ch_ind, :].squeeze(),
                                TDC_curves_mean[a,ch_ind, :].squeeze() + TDC_curves_std[a,ch_ind, :].squeeze(), color=colors[n], alpha=0.5)
            n += 1
    axs[1].set_xlabel('Delay [ns]')
    axs[1].set_ylabel('TDC units')
    axs[0].set_ylabel('CT units')
    axs[0].legend()
    axs[1].legend()
    axs[0].grid(True)
    axs[1].grid(True)

    TDC_curves_mean_list.append(TDC_curves_mean)
    CT_curves_mean_list.append(CT_curves_mean)
    TDC_curves_std_list.append(TDC_curves_std)
    CT_curves_std_list.append(CT_curves_std)
    TDC_test_data_list.append(TDC_test_data)
    CT_test_data_list.append(CT_test_data)

    asic_channels_board_list.append(asic_ch_list)
    total_curves += (n)
    possible_channels_board_list.append(possible_channels)

    # build the ramp extension


colors = distinctipy.get_colors(total_curves,rng=1)
n = 0
fig, axs = plt.subplots(2,sharex=True)
fig.suptitle('TDC offset comparison')
for e in range(N_boards):
    for a in range(np.size(TDC_curves_mean_list[e],axis=0)):
        for cin, c in enumerate(asic_channels_board_list[e][a][1], start=0):
            ch_ind = np.where(possible_channels_board_list[e] == c)
            axs[0].plot(set_delay[:, 1], CT_curves_mean_list[e][a,ch_ind, :].squeeze(), label='TDC counter DT'+ str(board_list[e]) +' A'+str(asic_channels_board_list[e][a][0]) + ' Ch'+str(c), color=colors[n])
            axs[0].fill_between(set_delay[:, 1], CT_curves_mean_list[e][a,ch_ind, :].squeeze() - CT_curves_std_list[e][a,ch_ind, :].squeeze(),
                                CT_curves_mean_list[e][a,ch_ind, :].squeeze() + CT_curves_std_list[e][a,ch_ind, :].squeeze(), color=colors[a], alpha=0.5)
            axs[1].plot(set_delay[:, 1], TDC_curves_mean_list[e][a,ch_ind, :].squeeze(), label='TDC ramp DT'+ str(board_list[e]) +' A'+str(asic_channels_board_list[e][a][0]) + ' Ch'+str(c), color=colors[n])
            axs[1].fill_between(set_delay[:, 1], TDC_curves_mean_list[e][a,ch_ind, :].squeeze() - TDC_curves_std_list[e][a,ch_ind, :].squeeze(),
                                TDC_curves_mean_list[e][a,ch_ind, :].squeeze() + TDC_curves_std_list[e][a,ch_ind, :].squeeze(), color=colors[n], alpha=0.5)
            n += 1
axs[1].set_xlabel('Delay [ns]')
axs[1].set_ylabel('TDC units')
axs[0].set_ylabel('CT units')
axs[0].legend()
axs[1].legend()
axs[0].grid(True)
axs[1].grid(True)
"""
# generating calibration data, first attempt
for e in range(N_boards):
    for a in range(np.size(TDC_curves_mean_list[e],axis=0)):
        for cin, c in enumerate(asic_channels_board_list[e][a][1], start=0):
            ch_ind = np.where(possible_channels_board_list[e] == c)
            board_ID = board_list[e]
            asic_ID = asic_channels_board_list[e][a][0]
            channel_ID = c
            time_data = set_delay[:, 1]
            CT_data = CT_curves_mean_list[e][a,ch_ind, :].squeeze()
            TDC_data = TDC_curves_mean_list[e][a,ch_ind, :].squeeze()

            TDC_test_data = TDC_test_data_list[e][a,ch_ind, :].squeeze().astype(int)
            CT_test_data = CT_test_data_list[e][a,ch_ind, :].squeeze().astype(int)

            # try to find the transition period by inspecting changes in the coarse time
            CT_data_norm = CT_data - np.min(CT_data)
            period_transitions = linear_extrap(np.array([0.5,1.5]),np.maximum.accumulate(CT_data_norm),time_data)
            estimated_period = period_transitions[1] - period_transitions[0]


            # find where we have ramp transitions
            ramp_transition_mask = CT_data == (CT_data.astype(int))

            # find what values of CT we start with. The middle is usually the one with the full ramp
            CT_values = np.unique(CT_data[ramp_transition_mask])

            # separate the three ramp pieces
            TDC_ramp_1 = TDC_data[CT_data == CT_values[0]]
            time_ramp_1 = time_data[CT_data == CT_values[0]]
            TDC_ramp_2 = TDC_data[CT_data == CT_values[1]]
            time_ramp_2 = time_data[CT_data == CT_values[1]]
            TDC_ramp_3 = TDC_data[CT_data == CT_values[2]]
            time_ramp_3 = time_data[CT_data == CT_values[2]]

            better_TDC_test_data_a = TDC_test_data[CT_data == CT_values[2], :]
            better_CT_test_data_a = CT_test_data[CT_data == CT_values[2], :]
            better_TDC_test_data_b = TDC_test_data[CT_data == CT_values[0],:]
            better_CT_test_data_b = CT_test_data[CT_data == CT_values[0],:]




            # find the offset of time such that, when offsetting the ramp, the error produced by overlapping is minimized
            # as data, we keep only the data points that are not during ramp transition
            x_data = time_data[ramp_transition_mask]
            y_data = TDC_data[ramp_transition_mask]

            def error(params):
                x_trans = x_data + params # we apply an offset to the timing data
                x_mask = (x_data >= np.min(x_trans) )*(x_data <= np.max(x_trans))# we should only compare points that are contained within the original data set
                y_trans = linear_extrap(x_data[x_mask],x_trans,y_data) # here we interpolate the function on the original data points in order to take the difference
                error = np.mean((y_trans - y_data[x_mask])**2) #mse of the points
                return error
            # optimize
            optimal_params = scipy.optimize.minimize_scalar(error, bounds=(23,27))
            # offset
            offset = optimal_params.x
            #offset = estimated_period

            # test if this is a good value
            # plt.figure()
            # plt.scatter(np.mod(time_data + time_data[1:]*np.arange(512).reshape(1,-1), offset), all_tdc_data, label='original data extended (wrapped)',
            #             marker='.', s=1)


            plt.figure()
            plt.plot(time_data[ramp_transition_mask],TDC_data[ramp_transition_mask],label='Base data',linestyle='None',marker='.',markersize=1)
            plt.plot(time_data[ramp_transition_mask] + estimated_period, TDC_data[ramp_transition_mask], label='Base data + optimal offset forward',
                     linestyle='None', marker='.',markersize=1)
            plt.plot(time_data[ramp_transition_mask] - estimated_period, TDC_data[ramp_transition_mask],
                     label='Base data + optimal offset backward',
                     linestyle='None', marker='.',markersize=1)
            plt.legend()

            # we need to build 512 ramps
            all_tdc_data = TDC_ramp_2.copy()
            all_time_data = time_ramp_2.copy()
            for ct in range(1,513):
                next_tdc_data = TDC_data[ramp_transition_mask]
                next_time_data = time_data[ramp_transition_mask] + ct*offset
                # refine the next data only for the points that lies outside the already present range. We want to extend it, not to clog it
                next_mask = next_time_data > all_time_data[-1]
                all_tdc_data = np.append(all_tdc_data,next_tdc_data[next_mask],axis=0)
                all_time_data = np.append(all_time_data,next_time_data[next_mask],axis=0)


            # plt.legend()
            # detect jumps, those will determine the value of coarse time
            differential_tdc_data = np.diff(all_tdc_data)
            tdc_range = np.max(all_tdc_data) - np.min(tdc_data)
            jump_locations = np.argwhere(differential_tdc_data > 0.5*tdc_range).squeeze()

            # now we split the TDC ramps into a list sequence with the index given by the coarse time
            TDC_data_list = [[] for i in range(512)]
            time_data_list = [[] for i in range(512)]
            TDC_data_list[0] = all_tdc_data[0:jump_locations[0]+1]
            time_data_list[0] = all_time_data[0:jump_locations[0]+1]
            for i in range(1,512):
                TDC_data_list[i] = all_tdc_data[jump_locations[i]+1:jump_locations[i+1]+1]
                time_data_list[i] = all_time_data[jump_locations[i]+1:jump_locations[i+1]+1]

            # now we start to build a matrix where we place our ramps
            TDC_calibration_matrix = np.zeros((512,1024))
            min_tdc_point = np.min(all_tdc_data)
            max_tdc_point = np.max(all_tdc_data)
            TDC_prompt = np.arange(np.round(min_tdc_point), np.round(max_tdc_point)).astype(int)
            for ct in range(512):
                TDC_inverse_data = linear_extrap(np.arange(1024),np.flip(np.minimum.accumulate(TDC_data_list[ct])),np.flip(time_data_list[ct]))
                TDC_calibration_matrix[ct,:] = TDC_inverse_data
            # test if we found a good parameter
            # if it is the case, by calculating the difference using calibrated data we should get a distribution around 0
            arrival_Time_test = TDC_calibration_matrix[CT_test_data,TDC_test_data]
            # delta_T_hist, bin_edges = np.histogram(deltaT[1:,:].flatten(), bins=1000)
            #

            plt.figure()
            for i in range(len(arrival_Time_test[0,:])):
                plt.plot(time_data,arrival_Time_test[:,i] - np.min(arrival_Time_test), linestyle='None', marker='.', markersize=2, color='b')
            plt.xlabel('True arrival time')
            plt.ylabel('Reconstructed arrival time')

            plt.figure()
            for i in range(len(arrival_Time_test[0,:])):
                plt.plot(time_data,time_data-(arrival_Time_test[:,i] - np.min(arrival_Time_test)), linestyle='None', marker='.', markersize=2, color='b')
            plt.xlabel('True arrival time')
            plt.ylabel('True - Reconstructed arrival time')



            # saving calibration data into a csv file with the name equal to the combination : BOARD/ASIC/CHANNEL
            np.savetxt("DT"+str(board_ID) +"_A"+str(asic_ID) +"_C"+str(channel_ID)+"_TDC_calibration_data.csv",TDC_calibration_matrix,delimiter=',')
"""
"""
# generating calibration data, another attempt
for e in range(N_boards):
    for a in range(np.size(TDC_curves_mean_list[e],axis=0)):
        for cin, c in enumerate(asic_channels_board_list[e][a][1], start=0):
            ch_ind = np.where(possible_channels_board_list[e] == c)
            board_ID = board_list[e]
            asic_ID = asic_channels_board_list[e][a][0]
            channel_ID = c
            time_data = set_delay[:, 1]
            CT_data = CT_curves_mean_list[e][a,ch_ind, :].squeeze()
            TDC_data = TDC_curves_mean_list[e][a,ch_ind, :].squeeze()

            TDC_test_data = TDC_test_data_list[e][a,ch_ind, :].squeeze().astype(int)
            CT_test_data = CT_test_data_list[e][a,ch_ind, :].squeeze().astype(int)

            # find where we have ramp transitions
            ramp_transition_mask = CT_data == (CT_data.astype(int))

            # find what values of CT we start with. The middle is usually the one with the full ramp
            CT_values = np.unique(CT_data[ramp_transition_mask])

            # separate the three ramp pieces
            TDC_ramp_1 = TDC_data[CT_data == CT_values[0]]
            time_ramp_1 = time_data[CT_data == CT_values[0]]
            TDC_ramp_2 = TDC_data[CT_data == CT_values[1]]
            time_ramp_2 = time_data[CT_data == CT_values[1]]
            TDC_ramp_3 = TDC_data[CT_data == CT_values[2]]
            time_ramp_3 = time_data[CT_data == CT_values[2]]

            # we know the complete ramp is the second, therefore we are going to invert it
            TDC_inverse_data = linear_extrap(np.arange(1024), np.flip(np.minimum.accumulate(TDC_ramp_2)),np.flip(time_ramp_2))
            def error(param):
                TDC_calibration_matrix_temp = np.expand_dims(TDC_inverse_data,axis=1)  + np.expand_dims(param*np.arange(0,512),axis=1).T
                deltaT_temp = TDC_calibration_matrix_temp[TDC_test_data, CT_test_data]
                # subtract the true arrival time
                deltaT_temp_relative = (deltaT_temp - np.min(deltaT_temp[0,:]))
                time_error = time_data - deltaT_temp_relative.T
                return np.abs(time_error).sum()

            # optimize
            optimal_params = scipy.optimize.minimize_scalar(error, bounds=(23,27))
            # offset
            offset = optimal_params.x
            # test if this is a good value

            plt.figure()
            plt.plot(time_data[ramp_transition_mask],TDC_data[ramp_transition_mask],label='Base data',linestyle='None',marker='.')
            plt.plot(time_data[ramp_transition_mask] + offset, TDC_data[ramp_transition_mask], label='Base data + optimal offset forward',
                     linestyle='None', marker='.')
            plt.plot(time_data[ramp_transition_mask] - offset, TDC_data[ramp_transition_mask],
                     label='Base data + optimal offset backward',
                     linestyle='None', marker='.')
            plt.legend()

            TDC_calibration_matrix = (np.expand_dims(TDC_inverse_data, axis=1) + np.expand_dims(offset * np.arange(0, 512), axis=1).T).T

            # test if we found a good parameter
            # if it is the case, by calculating the difference using calibrated data we should get a distribution around 0
            arrival_Time_test = TDC_calibration_matrix[CT_test_data,TDC_test_data]
            # delta_T_hist, bin_edges = np.histogram(deltaT[1:,:].flatten(), bins=1000)
            #
            plt.figure()
            for i in range(len(arrival_Time_test[0,:])):
                plt.plot(time_data,arrival_Time_test[:,i] - np.min(arrival_Time_test), linestyle='None', marker='.', markersize=2, color='b')
            plt.xlabel('True arrival time')
            plt.ylabel('Reconstructed arrival time')

            plt.figure()
            for i in range(len(arrival_Time_test[0,:])):
                plt.plot(time_data,time_data-(arrival_Time_test[:,i] - np.min(arrival_Time_test)), linestyle='None', marker='.', markersize=2, color='b')
            plt.xlabel('True arrival time')
            plt.ylabel('True - Reconstructed arrival time')

            # plt.stairs(delta_T_hist,bin_edges*1e-9)

            # saving calibration data into a csv file with the name equal to the combination : BOARD/ASIC/CHANNEL
            np.savetxt("DT"+str(board_ID) +"_A"+str(asic_ID) +"_C"+str(channel_ID)+"_TDC_calibration_data.csv",TDC_calibration_matrix,delimiter=',')

"""


"""
#better attempt to calibrate the TDC
for e in range(N_boards):
    for a in range(np.size(TDC_curves_mean_list[e],axis=0)):
        for cin, c in enumerate(asic_channels_board_list[e][a][1], start=0):
            ch_ind = np.where(possible_channels_board_list[e] == c)
            board_ID = board_list[e]
            asic_ID = asic_channels_board_list[e][a][0]
            channel_ID = c
            time_data = set_delay[:, 1]
            CT_data = CT_curves_mean_list[e][a,ch_ind, :].squeeze()
            TDC_data = TDC_curves_mean_list[e][a,ch_ind, :].squeeze()

            TDC_test_data = TDC_test_data_list[e][a,ch_ind, :].squeeze().astype(int)
            CT_test_data = CT_test_data_list[e][a,ch_ind, :].squeeze().astype(int)

            # find where we have ramp transitions
            ramp_transition_mask = CT_data == (CT_data.astype(int))

            # find what values of CT we start with. The middle is usually the one with the full ramp
            CT_values = np.unique(CT_data[ramp_transition_mask])

            # separate the three ramp pieces
            TDC_ramp_1 = TDC_data[CT_data == CT_values[0]]
            time_ramp_1 = time_data[CT_data == CT_values[0]]
            TDC_ramp_2 = TDC_data[CT_data == CT_values[1]]
            time_ramp_2 = time_data[CT_data == CT_values[1]]
            TDC_ramp_3 = TDC_data[CT_data == CT_values[2]]
            time_ramp_3 = time_data[CT_data == CT_values[2]]

            # find the offset of time such that, when offsetting the ramp, the error produced by overlapping is minimized
            # as data, we keep only the data points that are not during ramp transition
            x_data = time_data[ramp_transition_mask]
            y_data = TDC_data[ramp_transition_mask]

            def startingParameter(params):
                x_trans = x_data + params # we apply an offset to the timing data
                x_mask = (x_data >= np.min(x_trans) )*(x_data <= np.max(x_trans))# we should only compare points that are contained within the original data set
                y_trans = linear_extrap(x_data[x_mask],x_trans,y_data) # here we interpolate the function on the original data points in order to take the difference
                error = np.mean((y_trans - y_data[x_mask])**2) #mse of the points
                return error
            # optimize
            starting_offset = scipy.optimize.minimize_scalar(startingParameter, bounds=(23,27))

            # test if this is a good value
            def minimizeVariance(params):
                # we need to build 512 ramps
                all_tdc_data = TDC_ramp_2.copy()
                all_time_data = time_ramp_2.copy()
                for ct in range(1, 513):
                    next_tdc_data = TDC_data[ramp_transition_mask]
                    next_time_data = time_data[ramp_transition_mask] + ct * offset
                    # refine the next data only for the points that lies outside the already present range. We want to extend it, not to clog it
                    next_mask = next_time_data > all_time_data[-1]
                    all_tdc_data = np.append(all_tdc_data, next_tdc_data[next_mask], axis=0)
                    all_time_data = np.append(all_time_data, next_time_data[next_mask], axis=0)

                differential_tdc_data = np.diff(all_tdc_data)
                tdc_range = np.max(all_tdc_data) - np.min(tdc_data)
                jump_locations = np.argwhere(differential_tdc_data > 0.5*tdc_range).squeeze()

                # now we split the TDC ramps into a list sequence with the index given by the coarse time
                TDC_data_list = [[] for i in range(512)]
                time_data_list = [[] for i in range(512)]
                TDC_data_list[0] = all_tdc_data[0:jump_locations[0]+1]
                time_data_list[0] = all_time_data[0:jump_locations[0]+1]
                for i in range(1,512):
                    TDC_data_list[i] = all_tdc_data[jump_locations[i]+1:jump_locations[i+1]+1]
                    time_data_list[i] = all_time_data[jump_locations[i]+1:jump_locations[i+1]+1]

                # now we start to build a matrix where we place our ramps
                TDC_calibration_matrix = np.zeros((512,1024))
                min_tdc_point = np.min(all_tdc_data)
                max_tdc_point = np.max(all_tdc_data)
                TDC_prompt = np.arange(np.round(min_tdc_point), np.round(max_tdc_point)).astype(int)
                for ct in range(512):
                    TDC_inverse_data = linear_extrap(np.arange(1024),np.flip(np.minimum.accumulate(TDC_data_list[ct])),np.flip(time_data_list[ct]))
                    TDC_calibration_matrix[ct,:] = TDC_inverse_data
                # test if we found a good parameter
                # if it is the case, by calculating the difference using calibrated data we should get a distribution around 0
                deltaT = TDC_calibration_matrix[CT_test_data,TDC_test_data]
                return deltaT[1:,:].flatten().var()

            # optimize
            optimal_params = scipy.optimize.minimize_scalar(minimizeVariance,starting_offset.x,bounds=(23, 27))
            best_offset =optimal_params.x

            all_tdc_data = TDC_ramp_2.copy()
            all_time_data = time_ramp_2.copy()
            for ct in range(1, 513):
                next_tdc_data = TDC_data[ramp_transition_mask]
                next_time_data = time_data[ramp_transition_mask] + ct * best_offset
                # refine the next data only for the points that lies outside the already present range. We want to extend it, not to clog it
                next_mask = next_time_data > all_time_data[-1]
                all_tdc_data = np.append(all_tdc_data, next_tdc_data[next_mask], axis=0)
                all_time_data = np.append(all_time_data, next_time_data[next_mask], axis=0)

            # plt.figure()
            # plt.plot(all_time_data,all_tdc_data,label='reconstructed ramp sequence',linestyle='None',marker='.')
            # plt.legend()
            # detect jumps, those will determine the value of coarse time
            differential_tdc_data = np.diff(all_tdc_data)
            tdc_range = np.max(all_tdc_data) - np.min(tdc_data)
            jump_locations = np.argwhere(differential_tdc_data > 0.5 * tdc_range).squeeze()

            # now we split the TDC ramps into a list sequence with the index given by the coarse time
            TDC_data_list = [[] for i in range(512)]
            time_data_list = [[] for i in range(512)]
            TDC_data_list[0] = all_tdc_data[0:jump_locations[0] + 1]
            time_data_list[0] = all_time_data[0:jump_locations[0] + 1]
            for i in range(1, 512):
                TDC_data_list[i] = all_tdc_data[jump_locations[i] + 1:jump_locations[i + 1] + 1]
                time_data_list[i] = all_time_data[jump_locations[i] + 1:jump_locations[i + 1] + 1]

            # now we start to build a matrix where we place our ramps
            TDC_calibration_matrix = np.zeros((512, 1024))
            min_tdc_point = np.min(all_tdc_data)
            max_tdc_point = np.max(all_tdc_data)
            TDC_prompt = np.arange(np.round(min_tdc_point), np.round(max_tdc_point)).astype(int)
            for ct in range(512):
                TDC_inverse_data = linear_extrap(np.arange(1024), np.flip(np.minimum.accumulate(TDC_data_list[ct])),
                                                 np.flip(time_data_list[ct]))
                TDC_calibration_matrix[ct, :] = TDC_inverse_data
            # test if we found a good parameter
            # if it is the case, by calculating the difference using calibrated data we should get a distribution around 0
            deltaT = TDC_calibration_matrix[CT_test_data, TDC_test_data]
            delta_T_hist, bin_edges = np.histogram(deltaT[1:, :].flatten(), bins=1000)
            deltaT.var(1)
            plt.figure()
            plt.plot(deltaT,linestyle='None',marker='.',markersize=5)

            # saving calibration data into a csv file with the name equal to the combination : BOARD/ASIC/CHANNEL
            np.savetxt(
                "DT" + str(board_ID) + "_A" + str(asic_ID) + "_C" + str(channel_ID) + "_TDC_calibration_data.csv",
                TDC_calibration_matrix, delimiter=',')




            # saving calibration data into a csv file with the name equal to the combination : BOARD/ASIC/CHANNEL
            np.savetxt("DT"+str(board_ID) +"_A"+str(asic_ID) +"_C"+str(channel_ID)+"_TDC_calibration_data.csv",TDC_calibration_matrix,delimiter=',')
"""
# generating calibration data, last attempt
for e in range(N_boards):
    for a in range(np.size(TDC_curves_mean_list[e],axis=0)):
        for cin, c in enumerate(asic_channels_board_list[e][a][1], start=0):
            ch_ind = np.where(possible_channels_board_list[e] == c)
            board_ID = board_list[e]
            asic_ID = asic_channels_board_list[e][a][0]
            channel_ID = c
            time_data = set_delay[:, 1]
            CT_data = CT_curves_mean_list[e][a,ch_ind, :].squeeze()
            TDC_data = TDC_curves_mean_list[e][a,ch_ind, :].squeeze()

            TDC_test_data = TDC_test_data_list[e][a,ch_ind, :].squeeze().astype(int)
            CT_test_data = CT_test_data_list[e][a,ch_ind, :].squeeze().astype(int)

            # try to find the transition period by inspecting changes in the coarse time
            CT_data_norm = CT_data - np.min(CT_data)
            period_transitions = linear_extrap(np.array([0.5,1.5]),np.maximum.accumulate(CT_data_norm),time_data)
            estimated_period = period_transitions[1] - period_transitions[0]


            # find where we have ramp transitions
            ramp_transition_mask = CT_data == (CT_data.astype(int))

            # find what values of CT we start with. The middle is usually the one with the full ramp
            CT_values = np.unique(CT_data[ramp_transition_mask])

            # separate the three ramp pieces
            TDC_ramp_1 = TDC_data[CT_data == CT_values[0]]
            time_ramp_1 = time_data[CT_data == CT_values[0]]
            TDC_ramp_2 = TDC_data[CT_data == CT_values[1]]
            time_ramp_2 = time_data[CT_data == CT_values[1]]
            TDC_ramp_3 = TDC_data[CT_data == CT_values[2]]
            time_ramp_3 = time_data[CT_data == CT_values[2]]

            better_TDC_test_data_a = TDC_test_data[CT_data == CT_values[2], :]
            better_CT_test_data_a = CT_test_data[CT_data == CT_values[2], :]
            better_TDC_test_data_b = TDC_test_data[CT_data == CT_values[0],:]
            better_CT_test_data_b = CT_test_data[CT_data == CT_values[0],:]

            # invert the TDC values by inverting the second ramp
            TDC_inverse_data = linear_extrap(np.arange(1024), np.flip(np.minimum.accumulate(TDC_ramp_2)), np.flip(time_ramp_2))


            def error(current_period):
                # build the calibration matrix
                TDC_calibration_matrix = TDC_inverse_data.reshape(-1,1)  + np.ones((1024,512))*np.arange(512)*current_period

                arrival_test_a = TDC_calibration_matrix[better_TDC_test_data_a,better_CT_test_data_a]
                arrival_test_b = TDC_calibration_matrix[better_TDC_test_data_b, better_CT_test_data_b]
                true_delta_arrival_test = time_ramp_3.reshape(-1,1) - time_ramp_1
                err = 0
                for i in range(np.size(arrival_test_a,axis=1)):
                    a = arrival_test_a[:,i]
                    b = arrival_test_b[:,i]
                    d = a.reshape(-1, 1) - b
                    err += np.mean(np.abs(true_delta_arrival_test - d))
                return err
            # optimize
            optimal_params = scipy.optimize.minimize_scalar(error, bounds=(23,27))




            # offset
            offset = optimal_params.x
            #offset = estimated_period

            TDC_calibration_matrix = (TDC_inverse_data.reshape(-1,1)  + np.ones((1024,512))*np.arange(512)*offset).T

            plt.figure()
            plt.plot(time_data[ramp_transition_mask],TDC_data[ramp_transition_mask],label='Base data',linestyle='None',marker='.',markersize=1)
            plt.plot(time_data[ramp_transition_mask] + offset, TDC_data[ramp_transition_mask], label='Base data + optimal offset forward',
                     linestyle='None', marker='.',markersize=1)
            plt.plot(time_data[ramp_transition_mask] - offset, TDC_data[ramp_transition_mask],
                     label='Base data + optimal offset backward',
                     linestyle='None', marker='.',markersize=1)
            plt.legend()




            # saving calibration data into a csv file with the name equal to the combination : BOARD/ASIC/CHANNEL
            np.savetxt("DT"+str(board_ID) +"_A"+str(asic_ID) +"_C"+str(channel_ID)+"_TDC_calibration_data.csv",TDC_calibration_matrix,delimiter=',')


plt.show()