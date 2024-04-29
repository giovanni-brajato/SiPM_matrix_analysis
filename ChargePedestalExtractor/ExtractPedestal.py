

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
import sys
from matplotlib.table import CustomCell
from matplotlib.widgets import TextBox

from lmfit.models import PowerLawModel, ExponentialModel, GaussianModel, LinearModel

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


os.system('SETLOCAL EnableDelayedExpansion')

fileList = []
stringList = []
cwd = os.getcwd()
os.chdir(cwd)

pedestal_matrix = np.zeros((4,32))

for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith(".csv"):
            pedestal_raw_data = pd.read_csv((os.path.join(root, file)))
            asic_number = int(file.split('_')[0][-1])
            pedestal = pedestal_raw_data.values.mean(0)
            pedestal_matrix[asic_number%4,:] = pedestal
print('Pedestal data to copy and paste in your script:')
print(repr(pedestal_matrix))