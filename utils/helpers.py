# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 21:31:47 2025

@author: a.stratigakos
"""

import os 
from matpowercaseframes import CaseFrames
import numpy as np
import pandas as pd

def create_grid_dict(path, save = False):
    ''' reads .m file with matpowercaseframes, returns dictionary with problem matrices'''

    matgrid = CaseFrames(path)
        
    num_nodes = len(matgrid.bus)
    num_lines = len(matgrid.branch)
    
    #num_gen = len(matgrid.gen[gen_mask]) 
    #num_load = len(matgrid.bus)  # assume demand at each node

    # Construct incidence matrix
    A = np.zeros((num_lines, num_nodes))
    
    for l in range(num_lines):
        temp_line = matgrid.branch.iloc[l]
        #A[l, temp_line['F_BUS'].astype(int)-1] = 1
        #A[l, temp_line['T_BUS'].astype(int)-1] = -1
        A[l, np.where(matgrid.bus.BUS_I == temp_line['F_BUS'])[0]] = 1
        A[l, np.where(matgrid.bus.BUS_I == temp_line['T_BUS'])[0]] = -1
        
    # Construct diagonal reactance matrix
    react = 1/matgrid.branch['BR_X'].values
    b_diag = np.diag(react)
    
    # Bus susceptance matrix
    B_susc = A.T@b_diag@A
    
    B_line = b_diag@A
    B_inv = np.zeros(B_susc.shape)
    B_inv[1:,1:] = np.linalg.inv(B_susc[1:,1:])
    PTDF = B_line@B_inv
    
    #node_G = np.zeros((num_nodes, num_gen))
    #print(matgrid.gen)
    #for i in range(len(matgrid.gen[gen_mask])):
    #    node_G[np.where(matgrid.bus.BUS_I == matgrid.gen[gen_mask].GEN_BUS.iloc[i])[0], i] = 1
        
    node_L = np.diag(np.ones(num_nodes))
    
    #node_demand = matgrid.bus.PD.values
    Line_cap = matgrid.branch.RATE_A.values

    # Construct the incidence matrices for wind, solar, and conventional generators
    gen_names = pd.DataFrame(matgrid.gen_name)
    gen_name_split = gen_names['gen_name'].str.split("'\\t'", expand=True)
    gen_name_split.columns = ['name', 'type', 'fuel']
    gen_name_split = gen_name_split.apply(lambda x: x.str.strip("'"))

    # Get the names of wind and solar plants in the right order
    names_solar_all = gen_name_split[gen_name_split['fuel'] == 'Solar']['name']
    names_csp = gen_name_split[gen_name_split['type'] == 'CSP']['name']
    names_solar = names_solar_all[names_solar_all != '212_CSP_1'] # i want to skip CSP now
    names_wind = gen_name_split[gen_name_split['fuel'] == 'Wind']['name']
    names_neg_load = gen_name_split[gen_name_split['fuel'] == 'Hydro']['name']

    # Get row indices based on fuel type
    # solar, but do not consider the CSP (assume it is out of service)
    idx_solar = gen_name_split[gen_name_split['fuel'] == 'Solar'].index.tolist()
    idx_csp = gen_name_split[gen_name_split['type'] == 'CSP'].index.tolist()
    idx_solar = [x for x in idx_solar if x not in idx_csp]
    # other resources
    idx_wind = gen_name_split[gen_name_split['fuel'] == 'Wind'].index.tolist()
    idx_coal = gen_name_split[gen_name_split['fuel'] == 'Coal'].index.tolist()
    idx_oil = gen_name_split[gen_name_split['fuel'] == 'Oil'].index.tolist()
    idx_ng = gen_name_split[gen_name_split['fuel'] == 'NG'].index.tolist()
    idx_nuclear = gen_name_split[gen_name_split['fuel'] == 'Nuclear'].index.tolist()
    idx_hydro = gen_name_split[gen_name_split['fuel'] == 'Hydro'].index.tolist()
    idx_storage = gen_name_split[gen_name_split['fuel'] == 'Storage'].index.tolist()
    idx_sync_cond = gen_name_split[gen_name_split['fuel'] == 'Sync_Cond'].index.tolist()
    
    # Create combined fuel type groups
    idx_gen = sorted(list(set(idx_coal + idx_oil + idx_ng + idx_nuclear)))
    idx_neg_load = sorted(list(set(idx_hydro)))
    
    # Get bus indices for each fuel type
    #bus_idx_solar = matgrid.gen.iloc[idx_solar]['GEN_BUS'].tolist() if idx_solar else []
    #bus_idx_wind = matgrid.gen.iloc[idx_wind]['GEN_BUS'].tolist() if idx_wind else []
    #bus_idx_coal = matgrid.gen.iloc[idx_coal]['GEN_BUS'].tolist() if idx_coal else []
    #bus_idx_oil = matgrid.gen.iloc[idx_oil]['GEN_BUS'].tolist() if idx_oil else []
    #bus_idx_ng = matgrid.gen.iloc[idx_ng]['GEN_BUS'].tolist() if idx_ng else []
    #bus_idx_nuclear = matgrid.gen.iloc[idx_nuclear]['GEN_BUS'].tolist() if idx_nuclear else []
    #bus_idx_hydro = matgrid.gen.iloc[idx_hydro]['GEN_BUS'].tolist() if idx_hydro else []
    #bus_idx_storage = matgrid.gen.iloc[idx_storage]['GEN_BUS'].tolist() if idx_storage else []
    #bus_idx_sync_cond = matgrid.gen.iloc[idx_sync_cond]['GEN_BUS'].tolist() if idx_sync_cond else []
    
    # Bus indices for combined groups
    #bus_idx_gen = matgrid.gen.iloc[idx_gen]['GEN_BUS'].tolist() if idx_gen else []
    #bus_idx_neg_load = matgrid.gen.iloc[idx_neg_load]['GEN_BUS'].tolist() if idx_neg_load else []
    
    # Additional filtered data for each fuel type
    # Get generator capacities by fuel type
    #if idx_solar:
    #    solar_capacities = matgrid.gen.iloc[idx_solar]['PMAX'].tolist()
    
    #if idx_wind:
    #    wind_capacities = matgrid.gen.iloc[idx_wind]['PMAX'].tolist()

    #if idx_gen:
    #    gen_p_max = np.array(matgrid.gen.iloc[idx_gen]['PMAX'].tolist())
    #    gen_p_min = np.array(matgrid.gen.iloc[idx_gen]['PMIN'].tolist())
    #    gen_cost_1 = np.array(matgrid.gencost.iloc[idx_gen]['COST_1'].tolist())
    #    gen_cost_2 = np.array(matgrid.gencost.iloc[idx_gen]['COST_2'].tolist())

    # Solar incidence matrix
    node_Solar = np.zeros((num_nodes, len(idx_solar)))
    for i, gen_idx in enumerate(idx_solar):
        # Use original generator dataframe, not filtered one
        gen_bus = matgrid.gen.iloc[gen_idx]['GEN_BUS']
        node_idx = np.where(matgrid.bus.BUS_I == gen_bus)[0][0]
        node_Solar[node_idx, i] = 1
    
    # Wind incidence matrix
    node_Wind = np.zeros((num_nodes, len(idx_wind)))
    for i, gen_idx in enumerate(idx_wind):
        # Use original generator dataframe, not filtered one
        gen_bus = matgrid.gen.iloc[gen_idx]['GEN_BUS']
        node_idx = np.where(matgrid.bus.BUS_I == gen_bus)[0][0]
        node_Wind[node_idx, i] = 1

    # Conventional generator incidence matrix
    node_Generator = np.zeros((num_nodes, len(idx_gen)))
    for i, gen_idx in enumerate(idx_gen):
        # Use original generator dataframe, not filtered one
        gen_bus = matgrid.gen.iloc[gen_idx]['GEN_BUS']
        node_idx = np.where(matgrid.bus.BUS_I == gen_bus)[0][0]
        node_Generator[node_idx, i] = 1

    # Negative load incidence matrix
    node_Negative_Load = np.zeros((num_nodes, len(idx_neg_load)))
    for i, gen_idx in enumerate(idx_neg_load):
        # Use original generator dataframe, not filtered one
        gen_bus = matgrid.gen.iloc[gen_idx]['GEN_BUS']
        node_idx = np.where(matgrid.bus.BUS_I == gen_bus)[0][0]
        node_Negative_Load[node_idx, i] = 1

    # Create grid dictionary
    grid = {}
    grid['Bus_ID'] = matgrid.bus['BUS_I'].values
    grid['Area'] = matgrid.bus['BUS_AREA'].values
    grid['Pd'] = matgrid.bus['PD'].values
    #grid['Pmax'] = matgrid.gen['PMAX'].values[gen_mask]
    grid['Pmax'] = np.array(matgrid.gen.iloc[idx_gen]['PMAX'].tolist())
    #grid['Pmin'] = matgrid.gen['PMIN'].values[gen_mask]
    grid['Pmin'] = np.array(matgrid.gen.iloc[idx_gen]['PMIN'].tolist())
    #grid['Cost'] = matgrid.gencost['COST_1'].values[gen_mask]
    grid['Pneg_load'] = np.array(matgrid.gen.iloc[idx_neg_load]['PMAX'].tolist()) # we simply take the PMAX of these generators as negative load
    grid['Solar_cap'] = np.array(matgrid.gen.iloc[idx_solar]['PMAX'].tolist())
    grid['Wind_cap'] = np.array(matgrid.gen.iloc[idx_wind]['PMAX'].tolist())
    
    # Cost vector: Use the slope of the first segment from the piecewise linear formulation
    # See MATPOWER Manual for details on the data format
    
    if (matgrid.gencost['MODEL']==1).all():    
        # Model == 1: Estimate slope
        n_pwl_segments = matgrid.gencost['NCOST'].values[0].astype(int)
        
        x0_ind = 2*n_pwl_segments - 1
        x1_ind = 2*n_pwl_segments - 3
        f0_ind = 2*n_pwl_segments - 2
        f1_ind = 2*n_pwl_segments - 4
        
        # Store coefficient for all generators
        matgrid.gencost['c1_slope'] = (matgrid.gencost[f'COST_{f1_ind}'] - matgrid.gencost[f'COST_{f0_ind}'])/\
            (matgrid.gencost[f'COST_{x1_ind}'] - matgrid.gencost[f'COST_{x0_ind}'])
            
        grid['Cost'] = matgrid.gencost['c1_slope'].values[idx_gen].round()
    
    elif (matgrid.gencost['MODEL']==2).all():
        # Model == 2: Retrieve coefficient directly        
        grid['Cost'] = matgrid.gencost['COST_1'].values[idx_gen]
    else:
        raise ValueError('Model type of cost function is not the same across all generators.')
        
    grid['Quadratic_Cost'] = np.zeros((len(idx_gen))) # simply make a np.zeros array now
    grid['baseMVA'] = matgrid.baseMVA

    grid['names_solar'] = names_solar
    grid['names_wind'] = names_wind
    grid['names_neg_load'] = names_neg_load
    
    grid['Line_Capacity'] = Line_cap
    #grid['node_G'] = node_G
    grid['node_G'] = node_Generator
    grid['node_W'] = node_Wind
    grid['node_S'] = node_Solar
    grid['node_Neg_Load'] = node_Negative_Load
    grid['node_L'] = node_L
    grid['B_susc'] = B_susc
    grid['A'] = A
    grid['b_diag'] = b_diag
    grid['B_line'] = B_line
    grid['PTDF'] = PTDF
    
    # Cardinality of sets
    grid['n_nodes'] = num_nodes
    grid['n_lines'] = num_lines
    #grid['n_unit'] = num_gen
    grid['n_unit'] = len(idx_gen)
    #grid['n_loads'] = num_load
    grid['n_wind'] = len(idx_wind)
    grid['n_solar'] =len(idx_solar)
    grid['n_neg_load'] = len(idx_neg_load)
    
    #Other parameters set by user
    #grid['VOLL'] = 500   #Value of Lost Load
    #grid['VOWS'] = 35   #Value of wind spillage
    #grid['gshed'] = 200   #Value of wind spillage
    
    grid['B_line'] = grid['b_diag']@grid['A']
    B_inv = np.zeros(grid['B_susc'].shape)
    B_inv[1:,1:] = np.linalg.inv(grid['B_susc'][1:,1:])
    grid['PTDF'] = grid['B_line']@B_inv

    #if save:  
    #    pickle.dump(grid, open(cd+'\\data\\'+network.split('.')[0]+'.sav', 'wb'))
    return grid