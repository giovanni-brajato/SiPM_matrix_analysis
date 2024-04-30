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
import statsmodels.api as sm
from statsmodels.graphics import tsaplots

import scipy.signal
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from numpy import matlib
import matplotlib
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


os.system('SETLOCAL EnableDelayedExpansion')

fileList = []
chList = []
FDList = []
QTList = []
AsicList = []
configFile = ""
oldMethodFile = "";
cwd = os.getcwd()
os.chdir(cwd)

clkPeriod = 0
winLength = 0
numberOfRep = 0



for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith(".csv"):
            if file == "old_method.csv":
                oldMethodFile = root + "\\" + file
            else:
                fileList.append(os.path.join(root, file))
                fileInfo = file.split('_')
                chList.append(int(fileInfo[1][fileInfo[2].find('CH')-1:]))
                FDList.append(fileInfo[3][0])
                QTList.append(fileInfo[2])
                AsicList.append(int(root[root.find("ASIC_")+5]))
        elif file.endswith(".config"):
            configFile = root + "\\" + file
plt.close('all')

# extract info from the configuration file
scurveConfig = open(configFile, "r")
for line in scurveConfig:
  params = line.split('=')
  if params[0] == "ScurveClockPeriod":
      clkPeriod = float(params[1])
  elif params[0] == "ScurveWindowLength":
      winLength = int(params[1])
  elif params[0] == "ScurveRepetitionNumber":
      numberOfRep = int(params[1])
m = -1


for file in fileList:
    m +=1

    with open(file, 'r') as currentFile:
        data = pd.read_csv(currentFile,sep=',')
        threshold_DAC_units = data.values[:,0]
        if FDList[m]=='D':
            raw_d_counts_new = data.values[:, 1:]
        else:
            raw_f_counts_new = data.values[:, 1:]



data = pd.read_csv(oldMethodFile,sep=';')
raw_d_counts_old = data.values[0,1:-1].astype(int)


plt.figure()
plt.plot(threshold_DAC_units,raw_d_counts_new.mean(1),label='New method')
plt.plot(threshold_DAC_units,raw_d_counts_old,label='Old method')
plt.xlabel("Vth DAC units")
plt.ylabel("Raw count")
plt.legend()

plt.figure()
plt.plot(threshold_DAC_units,raw_d_counts_new.mean(1)/max(raw_d_counts_new.mean(1)),label='New method, duration count')
plt.plot(threshold_DAC_units,np.gradient(raw_d_counts_new.mean(1))/max(np.gradient(raw_d_counts_new.mean(1))),label='New method, derivative of duration count')
plt.plot(threshold_DAC_units,raw_f_counts_new.mean(1)/max(raw_f_counts_new.mean(1)),label='New method, frequency count')
plt.plot(threshold_DAC_units,np.cumsum(raw_f_counts_new.mean(1))/max(np.cumsum(raw_f_counts_new.mean(1))),label='New method, integral of frequency count')
plt.xlabel("Vth DAC units")
plt.ylabel("Normalized raw count")
plt.legend()

plt.figure()
plt.plot(threshold_DAC_units,raw_f_counts_new.mean(1)/(winLength*clkPeriod),label='New method, pulse count')
plt.xlabel("Vth DAC units")
plt.ylabel("Frequency [Hz]")
plt.legend()


plt.show()

