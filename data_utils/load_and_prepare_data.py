# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 18:29:56 2025

@author: astratig
"""
from data_utils.preprocessing import add_lead_lagged_features
from data_utils.loaders import create_data_loader
import pandas as pd
import torch

# data_utils/load_and_prepare_data.py

def load_and_prepare_dataset(config):
    """
    Full data loading and preprocessing pipeline.
    Returns train/calib/test tensors and dataloaders.
    """

    # freq = config['shared_data']['frequency']
    horizon = config['shared_data']['horizon']
    lead_horizon = config['shared_data']['lead_horizon']
    # num_plants_selected = config['shared_data']['num_plants']
    # num_uncertainties = num_plants_selected
    lag_measurements = config['shared_data']['lag_measurements']
    target_zone = config['shared_data']['target_zone']
    validation_percentage = config['shared_data']['validation_perc']
    # results_dir = config['paths']['results_dir']

    data_dir = config["paths"]["data_dir"]
    nyiso_data_dir = config["paths"]["nyiso_data_dir"]

    # 1. Load data
    wind_metadata_df = pd.read_csv(f'{nyiso_data_dir}\\MetaData\\wind_meta.csv', index_col = 0)
    wind_actual_df = pd.read_csv(f'{nyiso_data_dir}\\Actuals\\2018\\Wind\\2018_wind_site.csv', 
                                 index_col = 0, parse_dates=True).resample(config['shared_data']['frequency']).mean()

    # Load weather-based forecasts, extrapolate to the chosen frequency
    ID_forecasts_df = pd.read_csv(f'{data_dir}\\wind_ID_forecasts.csv', parse_dates=True, index_col = 0)
    ID_forecasts_df = ID_forecasts_df.resample(config['shared_data']['frequency']).interpolate()

    zone_names = wind_metadata_df['load_zone'].unique()

    # 2. Process features

    # Create aggregate production per zone
    zonal_wind_df = pd.DataFrame(data = [], columns = zone_names)
    for z in zone_names:
        plant_zone = wind_metadata_df.index[(wind_metadata_df['load_zone']==z).values]    
        zonal_wind_df[z] = wind_actual_df[plant_zone].sum(1)/(wind_metadata_df.loc[plant_zone].capacity).sum()

    # Normalize plant production by nominal capacity
    for p in wind_metadata_df.index.values:
        wind_actual_df[p] = wind_actual_df[p]/wind_metadata_df.loc[p].capacity

    zone_all_plants = wind_metadata_df.query(f'load_zone in {target_zone}').index.values
    if target_zone == ['illustrative']:
        selected_plants = ['Marble_River', 'Noble_Clinton']
    else:
        selected_plants = zone_all_plants
        
    num_selected_plants = len(selected_plants)
    power_df = wind_actual_df[selected_plants].copy()    

    # 3. Add lags/leads
    processed_power_df, pred_cols, target_cols = add_lead_lagged_features(power_df.copy(), selected_plants, 
                                                                          min_lag = horizon, max_lag = lag_measurements + horizon, 
                                                                          lead_interval = lead_horizon)

    # 4. Merge with ID forecasts and split sets
    Predictors = processed_power_df[pred_cols]
    Y = processed_power_df[target_cols]

    train_start = config['splits']['train_start']
    calib_start = config['splits']['calib_start']
    test_start = config['splits']['test_start']

    n_valid_obs = int(validation_percentage*len(Y[train_start:calib_start]))
    
    # Merge predictors with weather
    Predictors = pd.merge(Predictors, ID_forecasts_df[ [f'{p}_ID_for' for p in selected_plants] ], how='inner', left_index=True, right_index=True).dropna()

    # Predictors with weather
    train_Pred = Predictors[train_start:calib_start].dropna()[:-n_valid_obs]
    valid_Pred = Predictors[train_start:calib_start].dropna()[-n_valid_obs:]
    calib_Pred = Predictors[calib_start:test_start].dropna()
    test_Pred = Predictors[test_start:].dropna()

    train_Y = Y[train_Pred.index[0]:train_Pred.index[-1]]
    valid_Y = Y[valid_Pred.index[0]:valid_Pred.index[-1]]
    calib_Y = Y[calib_Pred.index[0]:calib_Pred.index[-1]]
    test_Y = Y[test_Pred.index[0]:test_Pred.index[-1]]
    # Target = Y[test_Pred.index[0]:test_Pred.index[-1]]

    tensor_train_Pred = torch.FloatTensor(train_Pred.values)
    tensor_valid_Pred = torch.FloatTensor(valid_Pred.values)
    tensor_calib_Pred = torch.FloatTensor(calib_Pred.values)
    tensor_test_Pred = torch.FloatTensor(test_Pred.values)

    tensor_train_Y = torch.FloatTensor(train_Y.values)
    tensor_valid_Y = torch.FloatTensor(valid_Y.values)
    tensor_calib_Y = torch.FloatTensor(calib_Y.values)
    tensor_test_Y = torch.FloatTensor(test_Y.values)

    train_loader = create_data_loader([tensor_train_Pred, tensor_train_Y], batch_size = 512, shuffle = False)
    valid_loader = create_data_loader([tensor_valid_Pred, tensor_valid_Y], batch_size = 512, shuffle = False)
    calib_loader = create_data_loader([tensor_calib_Pred, tensor_calib_Y], batch_size = 512, shuffle = False)
    test_loader = create_data_loader([tensor_test_Pred, tensor_test_Y], batch_size = 250, shuffle = False)

    return {
        # DataLoaders
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "calib_loader": calib_loader,
        "test_loader": test_loader,

        # Raw data (pandas)
        "train_Pred": train_Pred,
        "train_Y": train_Y,
        "valid_Pred": valid_Pred,
        "valid_Y": valid_Y,
        "calib_Pred": calib_Pred,
        "calib_Y": calib_Y,
        "test_Pred": test_Pred,
        "test_Y": test_Y,

        # Torch tensors (optional, if useful)
        "tensor_train_Pred": tensor_train_Pred,
        "tensor_train_Y": tensor_train_Y,
        "tensor_valid_Pred": tensor_valid_Pred,
        "tensor_valid_Y": tensor_valid_Y,
        "tensor_calib_Pred": tensor_calib_Pred,
        "tensor_calib_Y": tensor_calib_Y,
        "tensor_test_Pred": tensor_test_Pred,
        "tensor_test_Y": tensor_test_Y,

        # Metadata
        "n_features": tensor_train_Pred.shape[1],
        "output_dim": tensor_train_Y.shape[1], 
        "num_selected_plants": num_selected_plants}

