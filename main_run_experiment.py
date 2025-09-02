# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 21:17:09 2025

@author: a.stratigakos
"""

import yaml
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from utils.helpers import create_grid_dict
from time import time
from matpowercaseframes import CaseFrames

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

from utils.helpers import create_grid_dict

# IEEE format plot parameters    
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

# import cvxpy as cp
# from cvxpylayers.torch import CvxpyLayer
import torch
from torch import nn

#%%
with open("config.yaml") as f:
    config = yaml.safe_load(f)

grid = create_grid_dict(f'{cd}\\data\\RTS_GMLC.m')
scenarios_path = config['paths']['rts_scenario_data_dir']
pglib_path = config['paths']['pglib_data_dir']
matgrid = CaseFrames(f'{cd}\\data\\RTS_GMLC.m')
v2_matgrid = CaseFrames(f'{pglib_path}\\pglib_opf_case73_ieee_rts.m')


#%% Load input/target data

## Feature data: Nodal net load values (actuals, point forecasts, MC scenarios)
nl_actual_df = pd.read_csv(scenarios_path  + '\\net_load_actual_no_hydro_rts_gmlc.csv', index_col = 0, parse_dates=True)
nl_forecast_df = pd.read_csv(scenarios_path  + '\\net_load_point_forecast_no_hydro_rts_gmlc.csv', index_col = 0, parse_dates=True)
nl_actual_df.columns = nl_actual_df.columns.values.astype(int)
nl_forecast_df.columns = nl_actual_df.columns.values.astype(int)

# net load prob forecasts (500 scenarios)
with open(scenarios_path + '\\net_load_prob_forecasts_dynamic_hydro_rts_gmlc.pkl', 'rb') as f:
    nl_scenarios_df = pickle.load(f)
    
nl_scenarios_df.columns = nl_scenarios_df.columns.set_names(['node', 'scenario'])

# reshape scenarios in np array
nl_scenarios_tensor = torch.FloatTensor(nl_scenarios_df.to_numpy().reshape(nl_scenarios_df.shape[0], grid['n_nodes'], 500))
nl_scenarios_array = (nl_scenarios_df.to_numpy().reshape(nl_scenarios_df.shape[0], grid['n_nodes'], 500))
print(nl_scenarios_tensor.shape)  # should be (time periods, nodes, scenarios)

## Target/ response data: histograms/ outputs of optimization solver
# !!!!! Add to config file
opt_output_path = config['paths']['optim_output_data_dir']

# !!!!! Read this from config file, based on experiment name
target_file = 'RTS_no_reserves_single_period.m.pickle'
with open(f'{opt_output_path}\\{target_file}', 'rb') as f:
    target_output = pickle.load(f)

#%%

# !!!!! Read from config (for the moment, only 1-d variables)
variables_interest = ['gen_cost']
# !!!!! For loop
# !!!!! Fix Y// does not cover the full length for some reason
array_Y = target_output[variables_interest[0]][:,0,:]
Y = pd.DataFrame(data = array_Y, index = nl_actual_df.index[:len(array_Y)])

train_start = config['splits']['train_start']
test_start = config['splits']['test_start']

n_valid_obs = int(config['shared_data']['validation_perc']*len(Y))

#%%
from data_utils.loaders import create_data_loader

train_Y = Y[train_start:test_start].dropna()[:-n_valid_obs]
valid_Y = Y[train_start:test_start].dropna()[-n_valid_obs:]
test_Y = Y[test_start:].dropna()

# Create Predictors

# !!!!!! Predictors that represent distributional uncertainty
Predictors = nl_forecast_df.copy()
Predictors['Hour'] = Predictors.index.hour
Predictors['Weekday'] = Predictors.index.weekday

train_Pred = Predictors[train_Y.index[0]:train_Y.index[-1]]
valid_Pred = Predictors[valid_Y.index[0]:valid_Y.index[-1]]
test_Pred = Predictors[test_Y.index[0]:test_Y.index[-1]]

# Numpy arrays to tensor
tensor_train_Pred = torch.FloatTensor(train_Pred.values)
tensor_valid_Pred = torch.FloatTensor(valid_Pred.values)
tensor_test_Pred = torch.FloatTensor(test_Pred.values)

tensor_train_Y = torch.FloatTensor(train_Y.values)
tensor_valid_Y = torch.FloatTensor(valid_Y.values)
tensor_test_Y = torch.FloatTensor(test_Y.values)

train_loader = create_data_loader([tensor_train_Pred, tensor_train_Y], batch_size = 256, shuffle = False)
valid_loader = create_data_loader([tensor_valid_Pred, tensor_valid_Y], batch_size = 256, shuffle = False)
test_loader = create_data_loader([tensor_test_Pred, tensor_test_Y], batch_size = 256, shuffle = False)

#%% Train model
from models.distribution_proxy import DistributionProxy, DistributionProxyWrapper, train_model

input_dim = tensor_train_Pred.shape[1]
# !!!!!! Config inputs
hidden_sizes = 3*[100]
n_quantiles = 9
dict_size = 25

q_grid = torch.linspace(0, 1, n_quantiles)

nn_model = DistributionProxy(input_dim, hidden_sizes, dict_size, n_quantiles)
optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
nn_wrapper = DistributionProxyWrapper(nn_model, q_grid, loss_kind = 'l2')

###### MLP Regressor
nn_model, logs = train_model(nn_wrapper, train_loader, valid_loader, optimizer, num_epochs = 500, patience=25, 
                              device="cpu", verbose=True)









