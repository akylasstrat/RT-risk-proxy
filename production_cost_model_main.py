# -*- coding: utf-8 -*-
"""
Script to run production cost model and generate data

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

import cvxpy as cp
import torch
from torch import nn
from cvxpylayers.torch import CvxpyLayer
# from torch.utils.data import Dataset, DataLoader

# Construct Economic Dipsatch Problem// cvxpy layers for solving in batches
class ED_single_period_layer(nn.Module):        
    
    # !!!!! ToDo: Add arg for level of reserves    
    "Single-period Economic Dispatch problem"
    
    def __init__(self, grid, c_viol = 1e3):
        super(ED_single_period_layer, self).__init__()
                
        node_G = grid['node_G']
        node_L = grid['node_L']
        # node_W = grid['node_W']            
        PTDF = grid['PTDF']

        self.Pmax = torch.FloatTensor(grid['Pmax'])
        self.Pmin = torch.FloatTensor(np.zeros(grid['n_unit']))
        self.Cost = torch.FloatTensor(grid['Cost'])
        # self.C_r_up = torch.FloatTensor(grid['C_r_up'])
        # self.C_r_down = torch.FloatTensor(grid['C_r_down'])
        
        self.c_viol = c_viol
                
        self.grid = grid
        
        # Parameters: nominal net load (could be negative)
        net_load_nominal = cp.Parameter((grid['n_nodes']))
        
        ###### DA variables and linear decision rules        
        ### variables    
        # DA Variables                        
        p_G = cp.Variable((grid['n_unit']), nonneg = True)
        pos_slack = cp.Variable((grid['n_nodes']), nonneg = True)
        neg_slack = cp.Variable((grid['n_nodes']), nonneg = True)
        gen_cost = cp.Variable((1))
        
        # r_up_G = cp.Variable((grid['n_unit']), nonneg = True)
        # r_down_G = cp.Variable((grid['n_unit']), nonneg = True)

        f_margin_up = cp.Variable((grid['n_lines']), nonneg = True)
        f_margin_down = cp.Variable((grid['n_lines']), nonneg = True)
        
        ### DA constraints
        Constraints = [p_G.sum() + pos_slack.sum() - neg_slack.sum() == net_load_nominal.sum(), 
                          p_G <= grid['Pmax'].reshape(-1),
                          grid['Line_Capacity'].reshape(-1) - f_margin_up == PTDF@( node_G@p_G - node_L@net_load_nominal + node_L@(pos_slack - neg_slack)),
                          grid['Line_Capacity'].reshape(-1) - f_margin_down == -PTDF@( node_G@p_G - node_L@net_load_nominal + node_L@(pos_slack - neg_slack)), 
                          gen_cost == self.Cost@p_G + self.c_viol*(pos_slack.sum()) + 0*neg_slack.sum()]
        
        # Res_constraints = [p_G + r_up_G<= grid['Pmax'].reshape(-1), p_G - r_down_G >= 0]
        # Res_cap_cost = self.C_r_up@r_up_G + self.C_r_down@r_down_G
        
        objective_funct = cp.Minimize( gen_cost ) 
                
        ed_prob = cp.Problem(objective_funct, Constraints)
         
        self.ed_layer = CvxpyLayer(ed_prob, parameters=[net_load_nominal],
                                           variables = [p_G, pos_slack, neg_slack, f_margin_up, f_margin_down, gen_cost])
        
    def forward(self, net_load_hat):
        'Forward pass of optimization solver'
        # batch_size = net_load_hat.shape[0]
        
        net_load_hat_tensor = torch.FloatTensor(net_load_hat)
        
        # solve ED market
        ed_output = self.ed_layer(net_load_hat_tensor, solver_args={'max_iters':50_000, "solve_method": "ECOS"})                
        
        # Store outputs
        ed_solutions = {}
        
        ed_solutions['p_G'] = ed_output[0].detach()
        ed_solutions['pos_slack'] = ed_output[1].detach()
        ed_solutions['neg_slack'] = ed_output[2].detach()
        ed_solutions['f_margin_up'] = ed_output[3].detach()
        ed_solutions['f_margin_down'] = ed_output[4].detach()
        ed_solutions['gen_cost'] = ed_output[5].detach()
                
        return ed_solutions

def ED_single_period(grid, c_viol = 1e3):
    node_G = grid['node_G']
    node_L = grid['node_L']
    # node_W = grid['node_W']            
    # node_RE = grid['node_Neg_Load']
    PTDF = grid['PTDF']
    
    # CVXPY problem
    # Parameters: nominal net load (could be negative)
    net_load_hat = cp.Parameter((grid['n_nodes']), name='net_load_hat')
    
    ###### DA variables and linear decision rules        
    ### variables    
    # DA Variables                        
    p_G = cp.Variable((grid['n_unit']), nonneg = True, name = 'p_G')
    pos_slack = cp.Variable((grid['n_nodes']), nonneg = True, name = 'pos_slack')   # load shedding
    neg_slack = cp.Variable((grid['n_nodes']), nonneg = True, name = 'neg_slack')   # renewable curtailment
    gen_cost = cp.Variable((1), name = 'gen_cost')
    
    f_margin_up = cp.Variable((grid['n_lines']), nonneg = True, name = 'f_margin_up')
    f_margin_down = cp.Variable((grid['n_lines']), nonneg = True, name = 'f_margin_down')
    
    ### DA constraints
    Constraints = [p_G.sum() + pos_slack.sum() - neg_slack.sum() == net_load_hat.sum(), 
                      p_G <= grid['Pmax'].reshape(-1),
                      grid['Line_Capacity'].reshape(-1) - f_margin_up == PTDF@( node_G@p_G - node_L@(net_load_hat - pos_slack + neg_slack)),
                      grid['Line_Capacity'].reshape(-1) - f_margin_down == -PTDF@( node_G@p_G - node_L@(net_load_hat - pos_slack + neg_slack)), 
                      gen_cost == grid['Cost']@p_G + c_viol*(pos_slack.sum()) + 0*neg_slack.sum()]
        
    objective_funct = cp.Minimize( gen_cost )         
    ed_prob = cp.Problem(objective_funct, Constraints)
    
    return ed_prob

#%%

with open("config.yaml") as f:
    config = yaml.safe_load(f)

grid = create_grid_dict(f'{cd}\\data\\RTS_GMLC.m')
scenarios_path = config['paths']['rts_scenario_data_dir']
pglib_path = config['paths']['pglib_data_dir']
matgrid = CaseFrames(f'{cd}\\data\\RTS_GMLC.m')
v2_matgrid = CaseFrames(f'{pglib_path}\\pglib_opf_case73_ieee_rts.m')

#%%

# !!!!!! Net load values are wrong
# Load nodal net load values (actuals, point forecasts, MC scenarios)
nl_actual_df = pd.read_csv(scenarios_path  + '\\net_load_actual_no_hydro_rts_gmlc.csv', index_col = 0, parse_dates=True)
nl_forecast_df = pd.read_csv(scenarios_path  + '\\net_load_point_forecast_no_hydro_rts_gmlc.csv', index_col = 0, parse_dates=True)
nl_actual_df.columns = nl_actual_df.columns.values.astype(int)
nl_forecast_df.columns = nl_actual_df.columns.values.astype(int)

# net load prob forecasts (500 scenarios)
with open(scenarios_path + '\\net_load_prob_forecasts_dynamic_hydro_rts_gmlc.pkl', 'rb') as f:
    nl_scenarios_df = pickle.load(f)
    
# with open(scenarios_path + '\\net_load_prob_forecasts_no_hydro_rts_gmlc.pkl', 'wb') as f:
#     pickle.dump(nl_scenarios_df.astype('float32'), f, protocol=pickle.HIGHEST_PROTOCOL)
nl_scenarios_df.columns = nl_scenarios_df.columns.set_names(['node', 'scenario'])
    

# reshape scenarios in np array
nl_scenarios_tensor = torch.FloatTensor(nl_scenarios_df.to_numpy().reshape(nl_scenarios_df.shape[0], grid['n_nodes'], 500))
nl_scenarios_array = (nl_scenarios_df.to_numpy().reshape(nl_scenarios_df.shape[0], grid['n_nodes'], 500))
print(nl_scenarios_tensor.shape)  # should be (time periods, nodes, scenarios)

#%% Sanity check, forecast accuracy + some plots
rmse = np.sqrt(np.square(nl_actual_df.values - nl_forecast_df.values).mean())
print(f'Aggregate RMSE:{rmse}')

nl_cols = nl_actual_df.columns
start_day = '2020-01-01'
end_day = '2020-01-10'

c_ind = 64

fig, ax = plt.subplots()

nl_actual_df[start_day:end_day][nl_cols[c_ind]].plot(ax = ax)
nl_forecast_df[start_day:end_day][nl_cols[c_ind]].plot(ax = ax)
nl_scenarios_df.xs(nl_cols[c_ind], level='node',axis=1)[start_day:end_day].iloc[:,:10].plot(ax=ax, 
                                                                                            linewidth = 0.5, color = 'black', 
                                                                                            alpha = 0.5, 
                                                                                            legend = False)

plt.show()

#%% Solve for a single problem instance// Layer

ED_model = ED_single_period_layer(grid, c_viol = 1e3)
start_time = time()
test = ED_model(nl_actual_df.values[:10])
print(time() - start_time)

print(test['gen_cost'])
#%% Solve for a single problem instance// CVXPY problem

ed_prob = ED_single_period(grid)

start_time = time()
for i in range(10):
    
    ed_prob.param_dict['net_load_hat'].value = nl_actual_df.values[i]
    test = ed_prob.solve(solver = 'GUROBI')
    print(test)
    
print(time() - start_time)


#%% Solve for MC scenarios

num_MC_scenarios = 100
batch_size_scen = 100
nl_scenarios_tensor

ED_prob = ED_single_period(grid, c_viol = 1e3)

for i in range(nl_actual_df.shape[0]):
    
    
    if i%250 == 0:
        print(f'Time period:{i}')
            
    for s in range(num_MC_scenarios):
        
        # Solve for MC scenarios
        
        ED_prob.param_dict['net_load_hat'].value = nl_scenarios_array[i, :, s]
        ED_prob.solve(solver = 'GUROBI')
    
        # temp_dict = ED_model(nl_scenarios_tensor[i,:,:num_MC_scenarios].T)
        
    
        if (s==0):
            
            scenario_out_dict = {}            
            
            for key in ED_prob.var_dict.keys():
                
                scenario_out_dict[key] = ED_prob.var_dict[key].value.reshape(-1,1)
        
        else:
            for key in ED_prob.var_dict.keys():
                scenario_out_dict[key] = np.concatenate([scenario_out_dict[key], ED_prob.var_dict[key].value.reshape(-1,1)], 1)
                
    
    # Concatenate over observations
    
    
    # !!!!!! Save this at lower resolution (float32 or even smaller, to save memory)
    if i == 0:
        Output = {}
        
        for key in scenario_out_dict.keys():            
            Output[key] = np.expand_dims(scenario_out_dict[key], axis = 0)    
    else:
        for key in scenario_out_dict.keys():            
            Output[key] = np.concatenate([Output[key], np.expand_dims(scenario_out_dict[key], axis = 0)], axis = 0) 
    
if (config['experiment']['save'] == True) and (i%100==0):
    with open(f'{cd}\\checkpoints\\RTS_no_reserves_single_period.m.pickle', 'wb') as handle:
        pickle.dump(Output, handle, protocol=pickle.HIGHEST_PROTOCOL)

# plt.plot(nl_forecast_df.iloc[0].values)
# plt.plot(nl_forecast_df.iloc[9].values)
# plt.show()
#%%

plt.hist(Output['gen_cost'][0,0,:], bins = 30)
plt.hist(Output['gen_cost'][1,0,:], bins = 30)
plt.hist(Output['gen_cost'][2,0,:], bins = 30)
plt.hist(Output['gen_cost'][3,0,:], bins = 30)
plt.show()




