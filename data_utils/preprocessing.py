# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:58:16 2025

@author: astratig
"""

def create_IDsupervised(target_parks, df, min_lag, max_lag):
    ''' Supervised learning set for ID forecasting with lags'''
    #min_lag = 1
    #max_lag = min_lag + 4 # 4 steps back
    # lead_time_name = '-' + target_col + '_t'+str(min_lag)
    p_df = df.copy()

    # Create supervised set
    pred_col = []
    for park in p_df.columns:
        for lag in range(min_lag, max_lag):
            p_df[park+'_'+str(lag)] = p_df[park].shift(lag)
            pred_col.append(park+'_'+str(lag))
    
    Predictors = p_df[pred_col]
    if len(target_parks) == 1:
        Y = p_df[target_parks].to_frame()
    else:
        Y = p_df[target_parks]
    return Y, Predictors, pred_col


def add_lead_lagged_features(df, columns, min_lag = 1, max_lag = 4, lead_interval  = 0, drop_na=True):
    """
    Adds lagged versions of specified columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        columns (list): List of column names to shift.
        min_lag (int): The min lag value (number of time steps to shift).
        max_lag (int): The max lag value (number of time steps to shift).
        max_lead (int): The maximum lead value (how many steps forward to shift).

    Returns:
        pd.DataFrame: The dataframe with new lagged columns.
        lists with column names for regression
    """
    df_copy = df.copy()
    
    predictor_col_names = []
    for col in columns:
        for lag in range(min_lag, max_lag + 1):
            new_col_name = f"{col}_lag_{lag}"
            df_copy[new_col_name] = df_copy[col].shift(lag)
            predictor_col_names.append(new_col_name)
            
    target_col_names = []            
    for col in columns:
        for lead in range(0, lead_interval + 1):
            new_col_name = f"{col}_lead_{lead}"
            df_copy[new_col_name] = df_copy[col].shift(-lead)
            target_col_names.append(new_col_name)
            
    if drop_na:
        df_copy.dropna(inplace=True)
        
    return df_copy, predictor_col_names, target_col_names


def v2_create_IDsupervised(target_parks, df, min_lag = 1, max_lag = 4, max_lead = 0):
    ''' Supervised learning set for ID forecasting with lags'''
    #min_lag = 1
    #max_lag = min_lag + 4 # 4 steps back
    # lead_time_name = '-' + target_col + '_t'+str(min_lag)
    p_df = df.copy()
    # original_columns = p_df.columns
    # Create supervised set
    
    # lagged predictors
    pred_col = []
    
    for park in target_parks:
        for lag in range(min_lag, max_lag):
            new_lag_col_name = park+'_lag_'+str(lag)

            p_df[new_lag_col_name] = p_df[park].shift(lag)
            pred_col.append(new_lag_col_name)
    
    target_col = target_parks

    if (max_lead > 0):
        for park in target_parks:
            
            for lead_ind in range(1, max_lead+1):
                new_col_name = park+'_lead_'+str(lead_ind)

                p_df[new_col_name] = p_df[park].shift(-lead_ind)
                target_col.append(new_col_name)
            
    Predictors = p_df[pred_col]

    if len(target_col) == 1:
        Y = p_df[target_col].to_frame()
    else:
        Y = p_df[target_col]
    return Y, Predictors, pred_col, target_col