# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 17:18:27 2025

@author: a.stratigakos
"""

import os, sys

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

# import pandapower
# from matpowercaseframes import CaseFrames
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matpowercaseframes import CaseFrames
# os.environ["KMP_DUPLICATE_LIB_OK"]="True"

# IEEE format plot parameters    
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

#%%
# Utility function

def grid_dict(path, save = False):
    ''' reads .m file with matpowercaseframes, returns dictionary with problem matrices'''

    matgrid = CaseFrames(path)
    # set cardinalities
    #gen_mask = matgrid.gen.PMAX > 0
        
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
    grid['Cost'] = np.array(matgrid.gencost.iloc[idx_gen,5].tolist()) # i take the starting cost of the first segment in the piecewise linear cost function
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

#%%

# Set problem parameters
data_folder = cd + f'\\data\\'    
results_folder = cd + f'\\results\\'
scenarios_path = 'C:/Users/akyla/OneDrive - University College London/RTS_GMLC_data'

## Construct net load profiles
## Load the rts system
grid = grid_dict(data_folder + 'RTS_GMLC.m')
## Potentially, compute the nodal load data from the zonal data and distribution factors
compute_nodal_load = 'no' # 'yes'

#%%
## [new] rts system with carmona data
# actuals net load
nl_actual_df = pd.read_csv(scenarios_path  + '\\net_load_actual_dynamic_hydro_rts_gmlc.csv', index_col = 0, parse_dates=True)

# point forecast net load
nl_forecast_df = pd.read_csv(scenarios_path  + '\\net_load_point_forecast_dynamic_hydro_rts_gmlc.csv', index_col = 0, parse_dates=True)

nl_actual_df.columns = nl_actual_df.columns.values.astype(int)
nl_forecast_df.columns = nl_actual_df.columns.values.astype(int)

# net load prob forecasts (500 scenarios)
with open(scenarios_path + '\\net_load_prob_forecasts_dynamic_hydro_rts_gmlc.pkl', 'rb') as f:
    nl_scenarios_df = pickle.load(f)
nl_scenarios_df.columns = nl_scenarios_df.columns.set_names(['node', 'scenario'])
    
#%% Check forecast accuracy

rmse = np.sqrt(np.square(nl_actual_df.values - nl_forecast_df.values).mean())
print(f'Aggregate RMSE:{rmse}')

nl_cols = nl_actual_df.columns
start_day = '2020-01-01'
end_day = '2020-01-03'

c_ind = 4

fig, ax = plt.subplots()

nl_actual_df[start_day:end_day][nl_cols[c_ind]].plot(ax = ax)
nl_forecast_df[start_day:end_day][nl_cols[c_ind]].plot(ax = ax)
nl_scenarios_df.xs(nl_cols[c_ind], level='node',axis=1)[start_day:end_day].iloc[:,:10].plot(ax=ax, 
                                                                                            linewidth = 0.5, color = 'black', 
                                                                                            alpha = 0.5, 
                                                                                            legend = False)

plt.show()

#%%
# net load quantiles of 500 scenarios
with open(scenarios_path + 'net_load_quantiles_dynamic_hydro_rts_gmlc.pkl', 'rb') as f:
    nl_quantiles_df = pickle.load(f)
nl_quantiles_df.columns = nl_quantiles_df.columns.set_names(['node', 'quantile'])

#%%
# point forecast load
l_forecast_df = pd.read_csv(scenarios_path  + '\\load_nodal_point_forecast_rts_gmlc.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# point forecast wind
w_forecast_df = pd.read_csv(scenarios_path  + '\\wind_point_forecast_ordered_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# point forecast solar
s_forecast_df = pd.read_csv(scenarios_path  + '\\solar_point_forecast_ordered_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# actuals load
l_actual_df = pd.read_csv(scenarios_path  + '\\load_nodal_actual_rts_gmlc.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# actuals wind
w_actual_df = pd.read_csv(scenarios_path  + '\\wind_actual_ordered_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# actuals solar
s_actual_df = pd.read_csv(scenarios_path  + '\\solar_actual_ordered_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')


#%%
if compute_nodal_load == 'yes':
    print('')
    # RTS-GMLC system: compute the nodal values of load, starting from the zonal values, and use the distribution factors following the demand reported in the RTS system
    #if test_system in config['rts_gmlc_case']:
        # Compute the sum of the P_d in each zone
        #load_tot_zone_1 = 0
        #load_tot_zone_2 = 0
        #load_tot_zone_3 = 0
        #for n in range(len(grid['Area'])):
        #    if grid['Area'][n] == 1:
        #        load_tot_zone_1 += grid['Pd'][n]
        #    elif grid['Area'][n] == 2:
        #        load_tot_zone_2 += grid['Pd'][n]
        #    elif grid['Area'][n] == 3:
        #        load_tot_zone_3 += grid['Pd'][n]
        # Compute the nodal fractions
        #grid['Nodal_fraction_load'] = np.zeros(len(grid['Pd']))
        #for n in range(len(grid['Area'])):
        #    grid['Nodal_fraction_load'][n] = grid['Pd'][n] / (load_tot_zone_1 if grid['Area'][n] == 1 else load_tot_zone_2 if grid['Area'][n] == 2 else load_tot_zone_3)

    # RTS-GMLC system: construct the timeseries of point forecast, actuals, and prob forecasts for load
    #if test_system in config['rts_gmlc_case']:
        
        #print("Loading zonal load data...")
        
        # Load zonal load data
        #load_zones_actual = pd.read_csv(scenarios_path + 'load_zones_actual_rts_gmlc_zones.csv', index_col=0)
        #load_zones_point_forecast = pd.read_csv(scenarios_path + 'load_zones_point_forecast_rts_gmlc_zones.csv', index_col=0)
        
        #print(f"Loaded zonal data shapes - Actual: {load_zones_actual.shape}, Point forecast: {load_zones_point_forecast.shape}")
        #print(f"Actual columns: {list(load_zones_actual.columns)}")
        #print(f"Point forecast columns: {list(load_zones_point_forecast.columns)}")
        
        # Get the number of nodes and check grid structure
        #n_nodes = len(grid['Area'])
        #print(f"Number of nodes in grid: {n_nodes}")
        #print(f"Grid areas: {grid['Area']}")
        #print(f"First few distribution factors: {grid['Nodal_fraction_load'][:10]}")
        
        # Define bus names for RTS-GMLC system (73 buses)
        # Use grid indices to map to bus names correctly
        #bus_names = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
        #            119, 120, 121, 122, 123, 124, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
        #            213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 301, 302, 303, 304, 305, 306,
        #            307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324,
        #            325]
        
        # Ensure we have the right number of bus names
        #if len(bus_names) != n_nodes:
        #    print(f"Warning: Number of bus names ({len(bus_names)}) doesn't match number of nodes ({n_nodes})")
            # Create sequential bus names if mismatch
        #    bus_names = list(range(1, n_nodes + 1))
        
        # Get number of time steps from the data and preserve datetime index
        #n_timesteps = len(load_zones_actual)
        #print(f"Number of timesteps: {n_timesteps}")
        #print(f"Datetime index sample: {load_zones_actual.index[:5]}")
        
        # Initialize nodal dataframes with same datetime index as input files
        #nodal_actual = pd.DataFrame(0.0, index=load_zones_actual.index, columns=bus_names)
        #nodal_point_forecast = pd.DataFrame(0.0, index=load_zones_point_forecast.index, columns=bus_names)
        
        #print("Distributing zonal loads to nodal loads using distribution factors...")
        
        # Distribute zonal values to nodal values for each zone
        #for zone in [1, 2, 3]:
        #    print(f"\nProcessing zone {zone}...")
            
            # Find nodes belonging to this zone
        #    zone_nodes = [n for n in range(n_nodes) if grid['Area'][n] == zone]
        #    zone_bus_names = [bus_names[n] for n in zone_nodes]
            
        #    print(f"Zone {zone} has {len(zone_nodes)} nodes")
        #    print(f"Zone {zone} bus names: {zone_bus_names[:5]}..." if len(zone_bus_names) > 5 else f"Zone {zone} bus names: {zone_bus_names}")
            
            # Find the correct column name for this zone
        #    zone_col = None
        #    possible_names = [f'Zone_{zone}', f'zone_{zone}', f'Zone{zone}', f'area_{zone}', f'Area_{zone}', str(zone)]
            
        #    for name in possible_names:
        #        if name in load_zones_actual.columns:
        #            zone_col = name
        #            break
            
            # If no match found, use column by index
        #    if zone_col is None:
        #        if zone-1 < len(load_zones_actual.columns):
        #            zone_col = load_zones_actual.columns[zone-1]
        #            print(f"Using column by index: '{zone_col}' for zone {zone}")
        #        else:
        #            print(f"Error: Could not find column for zone {zone}")
        #            continue
        #    else:
        #        print(f"Using column '{zone_col}' for zone {zone}")
            
            # Get the zonal data for this zone
        #    zone_actual_data = load_zones_actual[zone_col].values
        #    zone_forecast_data = load_zones_point_forecast[zone_col].values
            
        #    print(f"Zone {zone} sample actual data: {zone_actual_data[:5]}")
        #    print(f"Zone {zone} sample forecast data: {zone_forecast_data[:5]}")
            
            # Distribute to each node in this zone
        #    for n in zone_nodes:
        #        bus_name = bus_names[n]
        #        distribution_factor = grid['Nodal_fraction_load'][n]
                
        #        print(f"Node {n} (Bus {bus_name}): distribution factor = {distribution_factor}")
                
                # Distribute actual values
        #        nodal_actual[bus_name] = zone_actual_data * distribution_factor
                
                # Distribute point forecast values
        #        nodal_point_forecast[bus_name] = zone_forecast_data * distribution_factor
        
        #print("\nSaving nodal load data...")
        
        # Check if data was actually distributed
        #print(f"Sample nodal actual data (first bus, first 5 timesteps): {nodal_actual.iloc[:5, 0].values}")
        #print(f"Sample nodal forecast data (first bus, first 5 timesteps): {nodal_point_forecast.iloc[:5, 0].values}")
        
        # Save nodal actual loads (include datetime index)
        #nodal_actual.to_csv(scenarios_path + 'load_nodal_actual_rts_gmlc.csv', index=True)
        #print(f"Saved nodal actual loads: {nodal_actual.shape}")
        
        # Save nodal point forecasts (include datetime index)
        #nodal_point_forecast.to_csv(scenarios_path + 'load_nodal_point_forecast_rts_gmlc.csv', index=True)
        #print(f"Saved nodal point forecasts: {nodal_point_forecast.shape}")
        
        #print("Nodal load distribution completed successfully!")
        
        # Verification: check that zonal totals match
        #print("\nVerification - checking zonal totals...")
        #for zone in [1, 2, 3]:
        #    zone_nodes = [n for n in range(n_nodes) if grid['Area'][n] == zone]
        #    zone_bus_names = [bus_names[n] for n in zone_nodes]
            
            # Sum nodal actuals for this zone
        #    nodal_zone_total = nodal_actual[zone_bus_names].sum(axis=1)
            
            # Get original zonal value
        #    zone_col = None
        #    possible_names = [f'Zone_{zone}', f'zone_{zone}', f'Zone{zone}', f'area_{zone}', f'Area_{zone}', str(zone)]
        #    for name in possible_names:
        #        if name in load_zones_actual.columns:
        #            zone_col = name
        #            break
        #    if zone_col is None and zone-1 < len(load_zones_actual.columns):
        #        zone_col = load_zones_actual.columns[zone-1]
            
        #    if zone_col and zone_col in load_zones_actual.columns:
        #        original_zone_total = load_zones_actual[zone_col]
        #        max_diff = abs(nodal_zone_total - original_zone_total).max()
        #        print(f"Zone {zone}: Maximum difference between original and sum of nodal values: {max_diff:.6f}")
        #        print(f"Zone {zone}: Sample comparison - Original: {original_zone_total.iloc[0]:.2f}, Nodal sum: {nodal_zone_total.iloc[0]:.2f}")

        # Load zonal probabilistic forecasts (500 scenarios)
        #print("\nLoading zonal probabilistic forecasts...")
        #with open(scenarios_path + 'load_prob_forecasts_rts_gmlc_zones.pkl', 'rb') as f:
        #    load_zones_prob_forecast = pickle.load(f)
        
        #print(f"Loaded probabilistic forecasts shape: {load_zones_prob_forecast.shape}")
        #print(f"Probabilistic forecasts columns: {list(load_zones_prob_forecast.columns)}")
        
        # Initialize nodal probabilistic forecasts with multi-level columns (Bus, Scenario)
        #scenario_columns = []
        #n_scenarios = load_zones_prob_forecast.shape[1] // 3  # Assuming equal scenarios per zone
        #for bus_name in bus_names:
        #    for scenario in range(n_scenarios):
        #        scenario_columns.append((bus_name, scenario))
        
        #nodal_prob_forecast = pd.DataFrame(0.0, index=load_zones_prob_forecast.index, 
        #                                columns=pd.MultiIndex.from_tuples(scenario_columns, names=['Bus', 'Scenario']))
        
        #print(f"Distributing {n_scenarios} scenarios per zone to nodal loads...")
        
        # Distribute probabilistic forecasts for each zone
        #for zone in [1, 2, 3]:
        #    zone_nodes = [n for n in range(n_nodes) if grid['Area'][n] == zone]
            
            # Get zone columns (scenarios for this zone)
        #    zone_scenario_cols = [col for col in load_zones_prob_forecast.columns if f'zone_{zone}' in str(col).lower() or f'Zone_{zone}' in str(col)]
        #    if not zone_scenario_cols:
                # Try alternative column naming
        #        start_idx = (zone-1) * n_scenarios
        #        end_idx = zone * n_scenarios
        #        zone_scenario_cols = load_zones_prob_forecast.columns[start_idx:end_idx]
            
        #    print(f"Zone {zone}: Using {len(zone_scenario_cols)} scenario columns")
            
            # Distribute each scenario to nodes in this zone
        #    for scenario_idx, scenario_col in enumerate(zone_scenario_cols):
        #        zone_scenario_data = load_zones_prob_forecast[scenario_col].values
                
        #        for n in zone_nodes:
        #            bus_name = bus_names[n]
        #            distribution_factor = grid['Nodal_fraction_load'][n]
        #            nodal_prob_forecast[(bus_name, scenario_idx)] = zone_scenario_data * distribution_factor
        
        # Save nodal probabilistic forecasts
        #print("\nSaving nodal probabilistic forecasts...")
        #with open(scenarios_path + 'load_nodal_prob_forecasts_rts_gmlc_500_scenarios.pkl', 'wb') as f:
        #    pickle.dump(nodal_prob_forecast, f)
        #print(f"Saved nodal probabilistic forecasts: {nodal_prob_forecast.shape}")

else:
    print('')
    #break
    # Simply load the nodal load data which was previously computed
    with open(scenarios_path + 'load_nodal_prob_forecasts_rts_gmlc.pkl', 'rb') as f:
        load_nodal_prob_forecasts = pickle.load(f)
    load_nodal_prob_forecasts.columns = load_nodal_prob_forecasts.columns.set_names(['node', 'scenario'])

    load_nodal_actual_df = pd.read_csv(scenarios_path + 'load_nodal_actual_rts_gmlc.csv', index_col = 0, parse_dates=True, 
                         date_format = '%d/%m/%Y %H:%M')

    load_nodal_point_forecast_df = pd.read_csv(scenarios_path + 'load_nodal_point_forecast_rts_gmlc.csv', index_col = 0, parse_dates=True, 
                         date_format = '%d/%m/%Y %H:%M')
    
    # Same for solar data
    with open(scenarios_path + 'solar_prob_forecast_rts_gmlc_plants.pkl', 'rb') as f:
        solar_prob_forecasts = pickle.load(f)
    solar_prob_forecasts.columns = solar_prob_forecasts.columns.set_names(['plant', 'scenario'])

    solar_actual_df = pd.read_csv(scenarios_path + 'solar_actual_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
                         date_format = '%d/%m/%Y %H:%M')

    solar_point_forecast_df = pd.read_csv(scenarios_path + 'solar_point_forecast_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
                         date_format = '%d/%m/%Y %H:%M')

    # Same for wind data
    with open(scenarios_path + 'wind_prob_forecast_rts_gmlc_plants.pkl', 'rb') as f:
        wind_prob_forecasts = pickle.load(f)
    wind_prob_forecasts.columns = wind_prob_forecasts.columns.set_names(['plant', 'scenario'])

    wind_actual_df = pd.read_csv(scenarios_path + 'wind_actual_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
                         date_format = '%d/%m/%Y %H:%M')

    wind_point_forecast_df = pd.read_csv(scenarios_path + 'wind_point_forecast_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
                         date_format = '%d/%m/%Y %H:%M')

# Now, let's construct the nodal net load profiles, if needed
construct_net_load_profiles = 'yes'
if (construct_net_load_profiles == 'yes'): #and (test_system in config['rts_gmlc_case']):

    ## Let's first load the Hydro schedules (available from the RTS-GMLC github repo)
    # these schedules are for DA and RT (we disregard uncertainty in them)
    neg_load_actual_df = pd.read_csv(scenarios_path + 'DAY_AHEAD_hydro.csv', index_col = 0, parse_dates=True, 
                         date_format = '%d/%m/%Y %H:%M')
    
    neg_load_point_forecast_df = pd.read_csv(scenarios_path + 'DAY_AHEAD_hydro.csv', index_col = 0, parse_dates=True, 
                         date_format = '%d/%m/%Y %H:%M')

    # Go to nodal values
    neg_load_nodal_actual = neg_load_actual_df.values @ grid['node_Neg_Load'].T
    neg_load_nodal_point_forecast = neg_load_point_forecast_df.values @ grid['node_Neg_Load'].T
    
    # Create DataFrame with proper indexing using actual Bus IDs
    neg_load_nodal_actual_df = pd.DataFrame(
        neg_load_nodal_actual,
        index=wind_point_forecast_df.index,  # Keep the time index
        columns=[f'{int(bus_id)}' for bus_id in grid['Bus_ID']])

    neg_load_nodal_point_forecast_df = pd.DataFrame(
        neg_load_nodal_point_forecast,
        index=wind_point_forecast_df.index,  # Keep the time index
        columns=[f'{int(bus_id)}' for bus_id in grid['Bus_ID']])

    # Create multilevel DataFrame for neg_load_nodal_point_forecast with 500 scenarios
    # Each scenario contains the same data for all nodes (node-specific duplication)
    n_nodes = neg_load_nodal_point_forecast.shape[1]  # 73 nodes
    n_scenarios = len(wind_prob_forecasts.columns.get_level_values('scenario').unique())
    nodes = [bus_id for bus_id in grid['Bus_ID']]
    scenarios = wind_prob_forecasts.columns.get_level_values('scenario').unique()
    columns = pd.MultiIndex.from_product([nodes, scenarios], names=['node', 'scenario'])
    
    repeated_data = np.zeros((neg_load_nodal_point_forecast.shape[0], n_nodes * n_scenarios))

    for node_idx in range(n_nodes):
        start_col = node_idx * n_scenarios
        end_col = (node_idx + 1) * n_scenarios
        # Repeat the node column data across all scenarios
        node_data = neg_load_nodal_point_forecast[:, node_idx].reshape(-1, 1)
        repeated_data[:, start_col:end_col] = np.tile(node_data, (1, n_scenarios))
    
    # Create the DataFrame with multilevel columns
    neg_load_nodal_prob_forecast_df = pd.DataFrame(
        repeated_data, 
        columns=columns,
        index=wind_actual_df.index  # Use the same index as other time series
    )

    # Let's now create the nodal net load timeseries via the wind and solar incidence matrix
    # Let's first create the right order of the columns, based on how they appear in the matpower.m file of the RTS system
    wind_columns_order = grid['names_wind']
    solar_columns_order = grid['names_solar']

    ## -- ACTUAL --
    wind_actual_df = wind_actual_df[wind_columns_order]
    solar_actual_df = solar_actual_df[solar_columns_order]
    wind_nodal_actual = wind_actual_df.values @ grid['node_W'].T
    solar_nodal_actual = solar_actual_df.values @ grid['node_S'].T
    #neg_load_nodal = grid['node_Neg_Load'] @ grid['Pneg_load']  # Shape: (n_nodes,)
    #neg_load_nodal_matrix = np.tile(neg_load_nodal, (len(wind_actual_df), 1))  # Shape: (n_timesteps, n_nodes)

    # Create DataFrame with proper indexing using actual Bus IDs
    wind_nodal_actual_df = pd.DataFrame(
        wind_nodal_actual,
        index=wind_actual_df.index,  # Keep the time index
        columns=[f'{int(bus_id)}' for bus_id in grid['Bus_ID']])

    solar_nodal_actual_df = pd.DataFrame(
        solar_nodal_actual,
        index=solar_actual_df.index,  # Keep the time index
        columns=[f'{int(bus_id)}' for bus_id in grid['Bus_ID']])

    #neg_load_nodal_actual_df = pd.DataFrame(
    #    neg_load_nodal_matrix,
    #    index=wind_actual_df.index,  # Use the same time index as wind/solar
    #    columns=[f'{int(bus_id)}' for bus_id in grid['Bus_ID']])
    
    # Now, let's combine everything to get the net load = load - solar - wind - neg_load
    net_load_nodal_actual_df = load_nodal_actual_df - solar_nodal_actual_df - wind_nodal_actual_df - neg_load_nodal_actual_df

    ## -- POINT FORECAST --
    wind_point_forecast_df = wind_point_forecast_df[wind_columns_order]
    solar_point_forecast_df = solar_point_forecast_df[solar_columns_order]
    wind_nodal_point_forecast = wind_point_forecast_df.values @ grid['node_W'].T
    solar_nodal_point_forecast = solar_point_forecast_df.values @ grid['node_S'].T

    # Create DataFrame with proper indexing using actual Bus IDs
    wind_nodal_point_forecast_df = pd.DataFrame(
        wind_nodal_point_forecast,
        index=wind_point_forecast_df.index,  # Keep the time index
        columns=[f'{int(bus_id)}' for bus_id in grid['Bus_ID']])
    
    solar_nodal_point_forecast_df = pd.DataFrame(
        solar_nodal_point_forecast,
        index=solar_point_forecast_df.index,  # Keep the time index
        columns=[f'{int(bus_id)}' for bus_id in grid['Bus_ID']])

    #neg_load_nodal_point_forecast_df = pd.DataFrame(
    #    neg_load_nodal_matrix,
    #    index=wind_point_forecast_df.index,  # Use the same time index as wind/solar
    #    columns=[f'{int(bus_id)}' for bus_id in grid['Bus_ID']])

    # Now, let's combine everything to get the net load = load - solar - wind - neg_load
    net_load_nodal_point_forecast_df = load_nodal_point_forecast_df - solar_nodal_point_forecast_df - wind_nodal_point_forecast_df - neg_load_nodal_point_forecast_df

    ## -- PROBABILISTIC FORECASTS --
    # Transform wind probabilistic forecasts from plants to nodes
    # Use the SAME plant order as the point forecasts that work correctly
    scenarios = wind_prob_forecasts.columns.get_level_values('scenario').unique()
    plants_ordered = wind_point_forecast_df.columns  # Use exact same order as point forecasts!
    nodes = [bus_id for bus_id in grid['Bus_ID']]

    # Initialize result array
    wind_nodal_data = np.zeros((len(wind_prob_forecasts), len(nodes), len(scenarios)))
    
    # Process each scenario
    for i, scenario in enumerate(scenarios):
        # Extract data for this scenario in the SAME order as point forecasts
        scenario_data = np.zeros((len(wind_prob_forecasts), len(plants_ordered)))
        
        for j, plant in enumerate(plants_ordered):
            scenario_data[:, j] = wind_prob_forecasts[(plant, scenario)].values
        
        # Apply incidence matrix (same as point forecasts)
        nodal_data = scenario_data @ grid['node_W'].T
        wind_nodal_data[:, :, i] = nodal_data
    
    # Create final DataFrame
    columns = pd.MultiIndex.from_product([nodes, scenarios], names=['node', 'scenario'])
    wind_nodal_prob_forecast_df = pd.DataFrame(
        wind_nodal_data.reshape(len(wind_prob_forecasts), -1),
        index=wind_prob_forecasts.index,
        columns=columns
    )

    # Transform solar probabilistic forecasts from plants to nodes
    # Use the SAME plant order as the point forecasts that work correctly
    solar_scenarios = solar_prob_forecasts.columns.get_level_values('scenario').unique()
    solar_plants_ordered = solar_point_forecast_df.columns  # Use exact same order as point forecasts!
    
    # Initialize result array
    solar_nodal_data = np.zeros((len(solar_prob_forecasts), len(nodes), len(solar_scenarios)))
    
    # Process each scenario
    for i, scenario in enumerate(solar_scenarios):
        # Extract data for this scenario in the SAME order as point forecasts
        scenario_data = np.zeros((len(solar_prob_forecasts), len(solar_plants_ordered)))
        
        for j, plant in enumerate(solar_plants_ordered):
            scenario_data[:, j] = solar_prob_forecasts[(plant, scenario)].values
        
        # Apply incidence matrix (same as point forecasts)
        nodal_data = scenario_data @ grid['node_S'].T
        solar_nodal_data[:, :, i] = nodal_data
    
    # Create final DataFrame
    solar_columns = pd.MultiIndex.from_product([nodes, solar_scenarios], names=['node', 'scenario'])
    solar_nodal_prob_forecast_df = pd.DataFrame(
        solar_nodal_data.reshape(len(solar_prob_forecasts), -1),
        index=solar_prob_forecasts.index,
        columns=solar_columns
    )

    # Create negative load probabilistic forecasts (constant across scenarios)
    # Negative load has no uncertainty, so same value for all scenarios
    #n_scenarios = len(scenarios)  # Use same number of scenarios as wind
    #n_timesteps = len(solar_prob_forecasts)  # Same time index
    
    # Calculate nodal negative load (constant for all timesteps and scenarios)
    #neg_load_nodal = grid['node_Neg_Load'] @ grid['Pneg_load']  # Shape: (n_nodes,)
    
    # Create 3D array: (timesteps, nodes, scenarios) - same value across all scenarios
    #neg_load_nodal_data = np.zeros((n_timesteps, len(nodes), n_scenarios))
    #for i in range(n_scenarios):
        # Tile the nodal negative load across all timesteps for this scenario
    #    neg_load_nodal_data[:, :, i] = np.tile(neg_load_nodal, (n_timesteps, 1))
    
    # Create final DataFrame with same structure as wind/solar
    #neg_load_columns = pd.MultiIndex.from_product([nodes, scenarios], names=['node', 'scenario'])
    #neg_load_nodal_prob_forecast_df = pd.DataFrame(
    #    neg_load_nodal_data.reshape(n_timesteps, -1),
    #    index=solar_prob_forecasts.index,  # Use same time index
    #    columns=neg_load_columns
    #)

    # Now, let's combine everything to get the net load = load - solar - wind - neg_load
    # let's first ensure the index and columns are aligned, so the substraction works fine
    load_nodal_prob_forecasts = load_nodal_prob_forecasts.reindex(index=solar_prob_forecasts.index, columns=solar_nodal_prob_forecast_df.columns, fill_value=0)
    solar_nodal_prob_forecast_df = solar_nodal_prob_forecast_df.reindex(index=load_nodal_prob_forecasts.index, columns=load_nodal_prob_forecasts.columns, fill_value=0)
    wind_nodal_prob_forecast_df = wind_nodal_prob_forecast_df.reindex(index=load_nodal_prob_forecasts.index, columns=load_nodal_prob_forecasts.columns, fill_value=0)
    neg_load_nodal_prob_forecast_df = neg_load_nodal_prob_forecast_df.reindex(index=load_nodal_prob_forecasts.index, columns=load_nodal_prob_forecasts.columns, fill_value=0)

    net_load_prob_forecast_df = load_nodal_prob_forecasts - solar_nodal_prob_forecast_df - wind_nodal_prob_forecast_df - neg_load_nodal_prob_forecast_df

    # save the results as .csv or pickle file
    net_load_nodal_actual_df.to_csv(scenarios_path + 'net_load_actual_dynamic_hydro_rts_gmlc.csv')
    net_load_nodal_point_forecast_df.to_csv(scenarios_path + 'net_load_point_forecast_dynamic_hydro_rts_gmlc.csv')
    with open(scenarios_path + 'net_load_prob_forecasts_dynamic_hydro_rts_gmlc.pkl', 'wb') as f:
        pickle.dump(net_load_prob_forecast_df, f)

#else:
    #print('')
    #break
    # Simply load the nodal load data which was previously computed
    #with open(scenarios_path + 'net_load_prob_forecasts_rts_gmlc.pkl', 'rb') as f:
    #    net_load_prob_forecasts_df = pickle.load(f)
    #    net_load_prob_forecasts_df.columns = net_load_prob_forecasts_df.columns.set_names(['node', 'scenario'])

    #net_load_actual_df = pd.read_csv(scenarios_path + 'net_load_actual_rts_gmlc.csv', index_col = 0, parse_dates=True, date_format = '%d/%m/%Y %H:%M')

    #net_load_point_forecast_df = pd.read_csv(scenarios_path + 'net_load_point_forecast_rts_gmlc.csv', index_col = 0, parse_dates=True, date_format = '%d/%m/%Y %H:%M')

# Construct the dataframe with quantiles from the probabilistic forecasts
construct_quantiles_net_load_profiles = 'yes'
if (construct_quantiles_net_load_profiles == 'yes'): #and (test_system in config['rts_gmlc_case']):
    # define list of quantiles
    list_quantiles = [0, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.98, 0.99, 0.995, 1]

    # make the multilevel dataframe of quantiles of the 500 scenarios
    # Get the nodes from the net load DataFrame
    nodes = net_load_prob_forecast_df.columns.get_level_values('node').unique()
    
    # Initialize array to store quantile results
    n_timesteps = len(net_load_prob_forecast_df)
    n_nodes = len(nodes)
    n_quantiles = len(list_quantiles)
    
    quantiles_data = np.zeros((n_timesteps, n_nodes, n_quantiles))
    
    # Compute quantiles for each node across the 500 scenarios
    for i, node in enumerate(nodes):
        # Extract all scenarios for this node
        node_scenarios = net_load_prob_forecast_df[node].values  # Shape: (8760, 500)
        
        # Compute quantiles across scenarios (axis=1)
        for j, quantile in enumerate(list_quantiles):
            quantiles_data[:, i, j] = np.quantile(node_scenarios, quantile, axis=1)
    
    # Create MultiIndex columns with nodes and quantiles
    quantiles_columns = pd.MultiIndex.from_product([nodes, list_quantiles], names=['node', 'quantile'])
    
    # Create the final quantiles DataFrame
    net_load_quantiles_df = pd.DataFrame(
        quantiles_data.reshape(n_timesteps, -1),
        index=net_load_prob_forecast_df.index,
        columns=quantiles_columns
    )

    print('')
    print(net_load_quantiles_df)
    print('')
    
    # Save the quantiles dataframe
    with open(scenarios_path + 'net_load_quantiles_dynamic_hydro_rts_gmlc.pkl', 'wb') as f:
        pickle.dump(net_load_quantiles_df, f)

#else:
    #print('')
    #break
    # Simply load the quantiles dataframe if it exists
    #with open(scenarios_path + 'net_load_quantiles_rts_gmlc.pkl', 'rb') as f:
    #    net_load_quantiles_df = pickle.load(f)

#%%
## Some code to adapt the wind and solar dataframes, so that the column order is in the same order as the mpc.gen_name of the RTS github
## Remark that now I do it only for the actuals and point forecast, as the prob forecast I only use for calculating the SW req's, for which I don't care about the exact order
## Load the RTS dictionary
#rts_sys = grid_dict(cd + f'\\data\\' + 'RTS_GMLC.m')

#wind_columns_order = rts_sys['names_wind']
#solar_columns_order = rts_sys['names_solar']

## Adapt wind df's
#wind_actual_df = pd.read_csv(scenarios_path + 'wind_actual_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
#                     date_format = '%d/%m/%Y %H:%M')
#wind_point_forecast_df = pd.read_csv(scenarios_path + 'wind_point_forecast_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
#                     date_format = '%d/%m/%Y %H:%M')

#wind_actual_ordered_df = wind_actual_df[wind_columns_order]
#wind_point_forecast_ordered_df = wind_point_forecast_df[wind_columns_order]

#wind_actual_ordered_df.to_csv(scenarios_path + 'wind_actual_ordered_rts_gmlc_plants.csv')
#wind_point_forecast_ordered_df.to_csv(scenarios_path + 'wind_point_forecast_ordered_rts_gmlc_plants.csv')

## Adapt solar df's
#solar_actual_df = pd.read_csv(scenarios_path + 'solar_actual_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
#                     date_format = '%d/%m/%Y %H:%M')
#solar_point_forecast_df = pd.read_csv(scenarios_path + 'solar_point_forecast_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
#                     date_format = '%d/%m/%Y %H:%M')

#solar_actual_ordered_df = solar_actual_df[solar_columns_order]
#solar_point_forecast_ordered_df = solar_point_forecast_df[solar_columns_order]

#solar_actual_ordered_df.to_csv(scenarios_path + 'solar_actual_ordered_rts_gmlc_plants.csv')
#solar_point_forecast_ordered_df.to_csv(scenarios_path + 'solar_point_forecast_ordered_rts_gmlc_plants.csv')

#%%
## Load data
## [old] 5bus data with ercot data
# actual wind production
#wind_actual_all_df = pd.read_csv(f'{data_folder}\\wind_actual_norm.csv', index_col = 0, parse_dates=True, 
#                             date_format = '%d/%m/%Y %H:%M')
# forecasted wind production
#wind_forecast_all_df = pd.read_csv(f'{data_folder}\\wind_forecast_det_norm.csv', index_col = 0, parse_dates=True, 
#                               date_format = '%d/%m/%Y %H:%M')
# Carmona 5000 scenarios wind production
#with open(scenarios_path + 'wind_full_scenarios_norm.pkl', 'rb') as f:
#    scenarios_all = pickle.load(f)
#scenarios_all.columns = scenarios_all.columns.set_names(['plant', 'scenario'])
# Carmona 5000 scenarios wind production quantiles
#with open(scenarios_path + 'wind_full_quantiles_norm.pkl', 'rb') as f:
#    quantiles_all = pickle.load(f)
#quantiles_all.columns = quantiles_all.columns.set_names(['plant', 'scenario'])

## [new] rts system with carmona data
# actuals net load
nl_actual_df = pd.read_csv(scenarios_path  + '\\net_load_actual_dynamic_hydro_rts_gmlc.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# point forecast net load
nl_forecast_df = pd.read_csv(scenarios_path  + '\\net_load_point_forecast_dynamic_hydro_rts_gmlc.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# point forecast load
l_forecast_df = pd.read_csv(scenarios_path  + '\\load_nodal_point_forecast_rts_gmlc.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# point forecast wind
w_forecast_df = pd.read_csv(scenarios_path  + '\\wind_point_forecast_ordered_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# point forecast solar
s_forecast_df = pd.read_csv(scenarios_path  + '\\solar_point_forecast_ordered_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# actuals load
l_actual_df = pd.read_csv(scenarios_path  + '\\load_nodal_actual_rts_gmlc.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# actuals wind
w_actual_df = pd.read_csv(scenarios_path  + '\\wind_actual_ordered_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# actuals solar
s_actual_df = pd.read_csv(scenarios_path  + '\\solar_actual_ordered_rts_gmlc_plants.csv', index_col = 0, parse_dates=True, 
                             date_format = '%d/%m/%Y %H:%M')
# net load prob forecasts (500 scenarios)
with open(scenarios_path + 'net_load_prob_forecasts_dynamic_hydro_rts_gmlc.pkl', 'rb') as f:
    nl_scenarios_df = pickle.load(f)
nl_scenarios_df.columns = nl_scenarios_df.columns.set_names(['node', 'scenario'])
# net load quantiles of 500 scenarios
with open(scenarios_path + 'net_load_quantiles_dynamic_hydro_rts_gmlc.pkl', 'rb') as f:
    nl_quantiles_df = pickle.load(f)
nl_quantiles_df.columns = nl_quantiles_df.columns.set_names(['node', 'quantile'])