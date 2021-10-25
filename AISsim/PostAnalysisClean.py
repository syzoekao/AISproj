'''
Simulation results
'''

import numpy as np
import pandas as pd
import copy
import json
import timeit
import os
import AISsim.UtilFunction as util
import AISsim.AdaptMCMC as AdaptMCMC
os.chdir("/Users/zoekao/Documents/FishProject/virsim2")
cwd = os.getcwd()
print(cwd)

import matplotlib as mpl
# print(mpl.rcParams['backend'])
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
# plt.get_backend()
import corner
import seaborn as sns

att = pd.read_csv('data/lake_attribute.csv')
# att['boat'].fillna(att['boat'].mean(), inplace=True)
att['infest'].fillna(0, inplace=True)
att['inspect'].fillna(0, inplace=True)
att['zm_suit'].fillna(0, inplace=True)
att['ss_suit'].fillna(0, inplace=True)
print(list(att.columns.values))

zm_dow = pd.read_csv('data/zm_dow.csv')
ss_dow = pd.read_csv('data/ss_dow.csv')

att = pd.merge(att, zm_dow, how = 'left', left_on = 'dow', right_on = 'dow')
att = pd.merge(att, ss_dow, how = 'left', left_on = 'dow', right_on = 'dow')


# creating data vectors 
lake_id = copy.deepcopy(att['id'].values)


# risk of zm for each lake 
temp = []
for i in range(1, 101): 
	with open('results/scenarios_res_zm_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))

scenarios_res_zm = [val for sublist in temp for val in sublist]

lake_risk_dict = dict()

for s in ['S', 'E', 'P', 'D', 'T']: 
	risk_counts = np.zeros(lake_id.shape[0])
	risk_boat_counts = np.zeros(lake_id.shape[0])
	risk_river_counts = np.zeros(lake_id.shape[0])

	for i in range(len(scenarios_res_zm)): 
		# i = 0
		inf_id = np.array(scenarios_res_zm[i][s]['id']).astype(int)
		tmp_zeros = np.zeros(lake_id.shape[0])
		tmp_zeros[inf_id] = 1
		risk_counts = risk_counts + tmp_zeros

		temp = np.array(scenarios_res_zm[i][s]['boat']).astype(float)
		temp[temp == 0.0] = np.nan
		boat_id = temp*inf_id
		boat_id = boat_id[~np.isnan(boat_id)].astype(int)
		tmp_zeros = np.zeros(lake_id.shape[0])
		tmp_zeros[boat_id] = 1
		risk_boat_counts = risk_boat_counts + tmp_zeros

		temp = np.array(scenarios_res_zm[i][s]['river']).astype(float)
		temp[temp == 0.0] = np.nan
		river_id = temp*inf_id
		river_id = river_id[~np.isnan(river_id)].astype(int)
		tmp_zeros = np.zeros(lake_id.shape[0])
		tmp_zeros[river_id] = 1
		risk_river_counts = risk_river_counts + tmp_zeros

	output = np.column_stack((lake_id, risk_counts/10000, risk_boat_counts/risk_counts, \
		risk_river_counts/risk_counts, risk_boat_counts/10000, risk_river_counts/10000))
	output[np.isnan(output)] = 0
	output = pd.DataFrame(output)
	output.columns = ['id', s+'_p_infest', s+'_p_boat', s+'_p_river', s+'_r_boat', s+'_r_river']
	output['id'].astype(int)

	lake_risk_dict.update({s: output})

lake_risk_zm = pd.merge(att[['id','dow','lake_name', 'county_name', 'utm_x', 'utm_y', 'acre', 'infest_zm', 'zm_infest2018']], 
	lake_risk_dict['S'], how = 'left', left_on = 'id', right_on = 'id')
lake_risk_zm = pd.merge(lake_risk_zm, lake_risk_dict['E'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_zm = pd.merge(lake_risk_zm, lake_risk_dict['P'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_zm = pd.merge(lake_risk_zm, lake_risk_dict['D'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_zm = pd.merge(lake_risk_zm, lake_risk_dict['T'], how = 'left', 
	left_on = 'id', right_on = 'id')

lake_risk_zm.to_csv('results/risk table (zm)(new).csv')


# risk of ss for each lake 
temp = []
for i in range(1, 101): 
	with open('results/scenarios_res_ss_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))

scenarios_res_ss = [val for sublist in temp for val in sublist]

lake_risk_dict = dict()

for s in ['S', 'E', 'P', 'D', 'T']: 
	risk_counts = np.zeros(lake_id.shape[0])
	risk_boat_counts = np.zeros(lake_id.shape[0])
	risk_river_counts = np.zeros(lake_id.shape[0])

	for i in range(len(scenarios_res_ss)): 
		# i = 0
		inf_id = np.array(scenarios_res_ss[i][s]['id']).astype(int)
		tmp_zeros = np.zeros(lake_id.shape[0])
		tmp_zeros[inf_id] = 1
		risk_counts = risk_counts + tmp_zeros

		temp = np.array(scenarios_res_ss[i][s]['boat']).astype(float)
		temp[temp == 0.0] = np.nan
		boat_id = temp*inf_id
		boat_id = boat_id[~np.isnan(boat_id)].astype(int)
		tmp_zeros = np.zeros(lake_id.shape[0])
		tmp_zeros[boat_id] = 1
		risk_boat_counts = risk_boat_counts + tmp_zeros

		temp = np.array(scenarios_res_ss[i][s]['river']).astype(float)
		temp[temp == 0.0] = np.nan
		river_id = temp*inf_id
		river_id = river_id[~np.isnan(river_id)].astype(int)
		tmp_zeros = np.zeros(lake_id.shape[0])
		tmp_zeros[river_id] = 1
		risk_river_counts = risk_river_counts + tmp_zeros

	output = np.column_stack((lake_id, risk_counts/10000, risk_boat_counts/risk_counts, \
		risk_river_counts/risk_counts, risk_boat_counts/10000, risk_river_counts/10000))
	output[np.isnan(output)] = 0
	output = pd.DataFrame(output)
	output.columns = ['id', s+'_p_infest', s+'_p_boat', s+'_p_river', s+'_r_boat', s+'_r_river']
	output['id'].astype(int)

	lake_risk_dict.update({s: output})

lake_risk_ss = pd.merge(att[['id','dow','lake_name', 'county_name', 'utm_x', 'utm_y', 'acre', 'infest_ss', 'ss_infest2018']], 
	lake_risk_dict['S'], how = 'left', left_on = 'id', right_on = 'id')
lake_risk_ss = pd.merge(lake_risk_ss, lake_risk_dict['E'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_ss = pd.merge(lake_risk_ss, lake_risk_dict['P'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_ss = pd.merge(lake_risk_ss, lake_risk_dict['D'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_ss = pd.merge(lake_risk_ss, lake_risk_dict['T'], how = 'left', 
	left_on = 'id', right_on = 'id')

lake_risk_ss.to_csv('results/risk table (ss)(new).csv')

