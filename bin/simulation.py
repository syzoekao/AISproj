
import numpy as np
from scipy.misc import comb
from itertools import product
import pandas as pd
import copy
import json
import timeit
import networkx as nx
import scipy.stats as stats
import os
import UtilFunction as util
import AdaptMCMC
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)

nfile = 1

att = pd.read_csv('data/lake_attribute.csv')
att['boat'].fillna(att['boat'].mean(), inplace=True)
att['infest'].fillna(0, inplace=True)
att['inspect'].fillna(0, inplace=True)
att['zm_suit'].fillna(0, inplace=True)
att['ss_suit'].fillna(0, inplace=True)

zm2018 = pd.read_csv('data/zm_dow.csv')
zm2018.columns = ['dow', 'zm2018']
att = pd.merge(att, zm2018, how='left', left_on = 'dow', right_on = 'dow')
att['zm2018'].fillna(0, inplace=True)

ss2018 = pd.read_csv('data/ss_dow.csv')
ss2018.columns = ['dow', 'ss2018']
att = pd.merge(att, ss2018, how='left', left_on = 'dow', right_on = 'dow')
att['ss2018'].fillna(0, inplace=True)

print(list(att.columns.values))

del zm2018, ss2018

# creating data vectors 
lake_id = copy.deepcopy(att['id'].values)
infest_zm = copy.deepcopy(att['infest.zm'].values)
infest_ss = copy.deepcopy(att['infest.ss'].values)
infest_both = copy.deepcopy(infest_zm)
infest_zm2018 = copy.deepcopy(att['zm2018'].values)
infest_ss2018 = copy.deepcopy(att['ss2018'].values)
infest_both2018 = 1*(infest_zm2018 + infest_ss2018 > 0) 
zm_suit = copy.deepcopy(att['zm_suit'].values)
ss_suit = copy.deepcopy(att['ss_suit'].values)

# read networks
with open('data/boat_dict'+str(nfile%20)+'.txt') as json_file:  
    boat_dict = json.load(json_file)

boat_net = dict()
tmp_key = [k for k, v in boat_dict.items()]
for k in tmp_key: 
	k2, v2 = [[key for key, val in boat_dict[k].items() if key != k], \
	[val for key, val in boat_dict[k].items() if key != k]]
	k2 = [int(x) for x in k2]
	v2 = [int(x) for x in v2]
	if len(k2) > 0: 
		boat_net.update({int(k): dict(zip(k2, v2))})
del boat_dict


with open('data/small_prob'+str(nfile%20)+'.txt') as json_file:  
    small_dict = json.load(json_file)

small_prob = dict()
tmp_key = [k for k, v in small_dict.items()]
for k in tmp_key: 
	k2, v2 = [[key for key, val in small_dict[k].items() if key != k], \
	[val for key, val in small_dict[k].items() if key != k]]
	k2 = [int(x) for x in k2]
	if len(k2) > 0: 
		small_prob.update({int(k): dict(zip(k2, v2))})
del small_dict


# print([k for k,v in boat_net.items()])

river_net = pd.read_csv('data/river_net_sim.csv')
river_o = river_net['origin'].values
river_d = river_net['destination'].values
river_w = river_net['weight'].values

sim_param = np.load("data/param_sample.npy")

scenarios_ann_zm = []
scenarios_ann_ss = []
scenarios_res_zm = []
scenarios_res_ss = []

pre_ann_out = []
pre_res_zm = []
pre_res_ss = []

bb = timeit.default_timer()


for i in range(1, 101): 
	tmp_param = np.array([0.13699835, 0.00780083, 0.02649427, 0.30544969, 0.20874079, 0.08939699, 0.20789656])

	aa = timeit.default_timer()
	ann_out, res_zm0, infest_zm0, tmp_suit_zm0, res_ss0, infest_ss0, tmp_suit_ss0 = \
		util.pre_infest_outcome_func(factor_ss = tmp_param[0], 
			e_violate_zm = tmp_param[1], e_violate_ss = tmp_param[2], 
			river_inf_zm = tmp_param[3], river_inf_ss = tmp_param[4], 
			back_suit_zm = tmp_param[5], back_suit_ss = tmp_param[6], 
			boat_net = boat_net, small_prob = small_prob, river_o=river_o, river_d=river_d, river_w=river_w, 
			lake_id=lake_id, infest_zm=infest_zm, infest_ss=infest_ss, infest_both=infest_both, 
			zm_suit=zm_suit, ss_suit=ss_suit)
	print(timeit.default_timer()-aa)
	pre_ann_out.append(ann_out.tolist())

pre_ann = np.array(pre_ann_out)
print(np.mean(pre_ann, 0))


for i in range(1, 101): 
	print(i)

	tmp_param = sim_param[np.random.randint(1, sim_param.shape[0]+1)]

	aa = timeit.default_timer()
	ann_out, res_zm0, infest_zm0, tmp_suit_zm0, res_ss0, infest_ss0, tmp_suit_ss0 = \
		util.pre_infest_outcome_func(factor_ss = tmp_param[0], 
			e_violate_zm = tmp_param[1], e_violate_ss = tmp_param[2], 
			river_inf_zm = tmp_param[3], river_inf_ss = tmp_param[4], 
			back_suit_zm = tmp_param[5], back_suit_ss = tmp_param[6], 
			boat_net = boat_net, small_prob = small_prob, river_o=river_o, river_d=river_d, river_w=river_w, 
			lake_id=lake_id, infest_zm=infest_zm, infest_ss=infest_ss, infest_both=infest_both, 
			zm_suit=zm_suit, ss_suit=ss_suit)
	# print(timeit.default_timer()-aa)

	infest_both0 = 1*(infest_zm0 + infest_ss0 > 0) 

	pre_ann_out.append(ann_out.tolist())
	pre_res_zm.append(res_zm0)
	pre_res_ss.append(res_ss0)

	# Status Quo
	# aa = timeit.default_timer()
	ann_zm, res_zm, ann_ss, res_ss = util.Scenario(factor_ss = tmp_param[0], 
			e_violate_zm = tmp_param[1], e_violate_ss = tmp_param[2], 
			river_inf_zm = tmp_param[3], river_inf_ss = tmp_param[4], 
			back_suit_zm = tmp_param[5], back_suit_ss = tmp_param[6], 
			boat_net = boat_net, small_prob = small_prob, river_o=river_o, river_d=river_d, river_w=river_w, 
			lake_id=lake_id, infest_zm=infest_zm2018, infest_ss=infest_ss2018, infest_both=infest_both2018, 
			tmp_suit_zm=tmp_suit_zm0, tmp_suit_ss=tmp_suit_ss0, 
			scenario = "StatusQuo", result2dict = True)
	# print(timeit.default_timer()-aa)


	# Education
	# aa = timeit.default_timer()
	ann_zm_e, res_zm_e, ann_ss_e, res_ss_e = util.Scenario(factor_ss = tmp_param[0], 
			e_violate_zm = tmp_param[1], e_violate_ss = tmp_param[2], 
			river_inf_zm = tmp_param[3], river_inf_ss = tmp_param[4], 
			back_suit_zm = tmp_param[5], back_suit_ss = tmp_param[6], 
			boat_net = boat_net, small_prob = small_prob, river_o=river_o, river_d=river_d, river_w=river_w, 
			lake_id=lake_id, infest_zm=infest_zm2018, infest_ss=infest_ss2018, infest_both=infest_both2018, 
			tmp_suit_zm=tmp_suit_zm0, tmp_suit_ss=tmp_suit_ss0, 
			scenario = "Education", result2dict = True)
	# print(timeit.default_timer()-aa)


	# Penalty
	# aa = timeit.default_timer()
	ann_zm_p, res_zm_p, ann_ss_p, res_ss_p = util.Scenario(factor_ss = tmp_param[0], 
			e_violate_zm = tmp_param[1], e_violate_ss = tmp_param[2], 
			river_inf_zm = tmp_param[3], river_inf_ss = tmp_param[4], 
			back_suit_zm = tmp_param[5], back_suit_ss = tmp_param[6], 
			boat_net = boat_net, small_prob = small_prob, river_o=river_o, river_d=river_d, river_w=river_w, 
			lake_id=lake_id, infest_zm=infest_zm2018, infest_ss=infest_ss2018, infest_both=infest_both2018, 
			tmp_suit_zm=tmp_suit_zm0, tmp_suit_ss=tmp_suit_ss0, 
			scenario = "Penalty", result2dict = True)
	# print(timeit.default_timer()-aa)


	# MandDecon
	# aa = timeit.default_timer()
	ann_zm_d, res_zm_d, ann_ss_d, res_ss_d = util.Scenario(factor_ss = tmp_param[0], 
			e_violate_zm = tmp_param[1], e_violate_ss = tmp_param[2], 
			river_inf_zm = tmp_param[3], river_inf_ss = tmp_param[4], 
			back_suit_zm = tmp_param[5], back_suit_ss = tmp_param[6], 
			boat_net = boat_net, small_prob = small_prob, river_o=river_o, river_d=river_d, river_w=river_w, 
			lake_id=lake_id, infest_zm=infest_zm2018, infest_ss=infest_ss2018, infest_both=infest_both2018, 
			tmp_suit_zm=tmp_suit_zm0, tmp_suit_ss=tmp_suit_ss0, 
			scenario = "MandDecon", result2dict = True)
	# print(timeit.default_timer()-aa)


	# ReduceTraffic
	# aa = timeit.default_timer()
	ann_zm_t, res_zm_t, ann_ss_t, res_ss_t = util.Scenario(factor_ss = tmp_param[0], 
			e_violate_zm = tmp_param[1], e_violate_ss = tmp_param[2], 
			river_inf_zm = tmp_param[3], river_inf_ss = tmp_param[4], 
			back_suit_zm = tmp_param[5], back_suit_ss = tmp_param[6], 
			boat_net = boat_net, small_prob = small_prob, river_o=river_o, river_d=river_d, river_w=river_w, 
			lake_id=lake_id, infest_zm=infest_zm2018, infest_ss=infest_ss2018, infest_both=infest_both2018, 
			tmp_suit_zm=tmp_suit_zm0, tmp_suit_ss=tmp_suit_ss0, 
			scenario = "ReduceTraffic", result2dict = True)
	print(timeit.default_timer()-aa)


	scenarios_ann_zm.append({'S': ann_zm.tolist(), 'E': ann_zm_e.tolist(), 'P': ann_zm_p.tolist(), 'D': ann_zm_d.tolist(), \
	'T': ann_zm_t.tolist()})

	scenarios_ann_ss.append({'S': ann_ss.tolist(), 'E': ann_ss_e.tolist(), 'P': ann_ss_p.tolist(), 'D': ann_ss_d.tolist(), \
	'T': ann_ss_t.tolist()})

	scenarios_res_zm.append({'S': res_zm, 'E': res_zm_e, 'P': res_zm_p, 'D': res_zm_d, 'T': res_zm_t})
	scenarios_res_ss.append({'S': res_ss, 'E': res_ss_e, 'P': res_ss_p, 'D': res_ss_d, 'T': res_ss_t})

	del ann_out, res_zm0, infest_zm0, tmp_suit_zm0, res_ss0, infest_ss0, tmp_suit_ss0, \
	ann_zm, ann_zm_e, ann_zm_p, ann_zm_d, ann_zm_t, ann_ss, ann_ss_e, ann_ss_p, ann_ss_d, ann_ss_t, \
	res_zm, res_zm_e, res_zm_p, res_zm_d, res_zm_t, res_ss, res_ss_e, res_ss_p, res_ss_d, res_ss_t

print(timeit.default_timer()-bb)


with open('results/scenarios_ann_zm_'+str(nfile)+'.txt', 'w') as fout:
    json.dump(scenarios_ann_zm, fout)

with open('results/scenarios_ann_ss_'+str(nfile)+'.txt', 'w') as fout:
    json.dump(scenarios_ann_ss, fout)

with open('results/scenarios_res_zm_'+str(nfile)+'.txt', 'w') as fout:
    json.dump(scenarios_res_zm, fout)

with open('results/scenarios_res_ss_'+str(nfile)+'.txt', 'w') as fout:
    json.dump(scenarios_res_ss, fout)

with open('results/pre_ann_out_'+str(nfile)+'.txt', 'w') as fout:
    json.dump(pre_ann_out, fout)

with open('results/pre_res_zm_'+str(nfile)+'.txt', 'w') as fout:
    json.dump(pre_res_zm, fout)

with open('results/pre_res_ss_'+str(nfile)+'.txt', 'w') as fout:
    json.dump(pre_res_ss, fout)


'''
# pre-management scenarios 
# zebra mussels

risk_counts = np.zeros(lake_id.shape[0])
risk_boat_counts = np.zeros(lake_id.shape[0])
risk_river_counts = np.zeros(lake_id.shape[0])

for i in range(len(scenarios_res_zm)): 
	# i = 0
	inf_id = np.array(pre_res_zm[i]['id']).astype(int)
	tmp_zeros = np.zeros(lake_id.shape[0])
	tmp_zeros[inf_id] = 1
	risk_counts = risk_counts + tmp_zeros

	temp = np.array(pre_res_zm[i]['boat']).astype(float)
	temp[temp == 0.0] = np.nan
	boat_id = temp*inf_id
	boat_id = boat_id[~np.isnan(boat_id)].astype(int)
	tmp_zeros = np.zeros(lake_id.shape[0])
	tmp_zeros[boat_id] = 1
	risk_boat_counts = risk_boat_counts + tmp_zeros

	temp = np.array(pre_res_zm[i]['river']).astype(float)
	temp[temp == 0.0] = np.nan
	river_id = temp*inf_id
	river_id = river_id[~np.isnan(river_id)].astype(int)
	tmp_zeros = np.zeros(lake_id.shape[0])
	tmp_zeros[river_id] = 1
	risk_river_counts = risk_river_counts + tmp_zeros

output = np.column_stack((lake_id, risk_counts, risk_boat_counts, risk_river_counts))
output = pd.DataFrame(output)
output.columns = ['id', 'pre_n_infest', 'pre_n_boat', 'pre_n_river']
output['id'].astype(int)

pre_risk_zm = pd.merge(att[['id','dow','lake_name', 'acre', 'infest.zm']], output, how = 'left', 
	left_on = 'id', right_on = 'id')


# starry stonewort

risk_counts = np.zeros(lake_id.shape[0])
risk_boat_counts = np.zeros(lake_id.shape[0])
risk_river_counts = np.zeros(lake_id.shape[0])

for i in range(len(scenarios_res_ss)): 
	# i = 0
	inf_id = np.array(pre_res_ss[i]['id']).astype(int)
	tmp_zeros = np.zeros(lake_id.shape[0])
	tmp_zeros[inf_id] = 1
	risk_counts = risk_counts + tmp_zeros

	temp = np.array(pre_res_ss[i]['boat']).astype(float)
	temp[temp == 0.0] = np.nan
	boat_id = temp*inf_id
	boat_id = boat_id[~np.isnan(boat_id)].astype(int)
	tmp_zeros = np.zeros(lake_id.shape[0])
	tmp_zeros[boat_id] = 1
	risk_boat_counts = risk_boat_counts + tmp_zeros

	temp = np.array(pre_res_ss[i]['river']).astype(float)
	temp[temp == 0.0] = np.nan
	river_id = temp*inf_id
	river_id = river_id[~np.isnan(river_id)].astype(int)
	tmp_zeros = np.zeros(lake_id.shape[0])
	tmp_zeros[river_id] = 1
	risk_river_counts = risk_river_counts + tmp_zeros

output = np.column_stack((lake_id, risk_counts, risk_boat_counts, risk_river_counts))
output = pd.DataFrame(output)
output.columns = ['id', 'pre_n_infest', 'pre_n_boat', 'pre_n_river']
output['id'].astype(int)

pre_risk_ss = pd.merge(att[['id','dow','lake_name', 'acre', 'infest.ss']], output, how = 'left', 
	left_on = 'id', right_on = 'id')




# risk of zm for each lake 

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

	output = np.column_stack((lake_id, risk_counts, risk_boat_counts, risk_river_counts))
	output = pd.DataFrame(output)
	output.columns = ['id', s+'_n_infest', s+'_n_boat', s+'_n_river']
	output['id'].astype(int)

	lake_risk_dict.update({s: output})

lake_risk_zm = pd.merge(att[['id','dow','lake_name', 'acre', 'infest.zm']], lake_risk_dict['S'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_zm = pd.merge(lake_risk_zm, lake_risk_dict['E'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_zm = pd.merge(lake_risk_zm, lake_risk_dict['P'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_zm = pd.merge(lake_risk_zm, lake_risk_dict['D'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_zm = pd.merge(lake_risk_zm, lake_risk_dict['T'], how = 'left', 
	left_on = 'id', right_on = 'id')

lake_risk_zm.to_csv('results/risk table (zm).csv')

# risk of ss for each lake 

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

	output = np.column_stack((lake_id, risk_counts, risk_boat_counts, risk_river_counts))
	output = pd.DataFrame(output)
	output.columns = ['id', s+'_n_infest', s+'_n_boat', s+'_n_river']
	output['id'].astype(int)

	lake_risk_dict.update({s: output})

lake_risk_ss = pd.merge(att[['id','dow','lake_name', 'acre', 'infest.ss']], lake_risk_dict['S'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_ss = pd.merge(lake_risk_ss, lake_risk_dict['E'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_ss = pd.merge(lake_risk_ss, lake_risk_dict['P'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_ss = pd.merge(lake_risk_ss, lake_risk_dict['D'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_ss = pd.merge(lake_risk_ss, lake_risk_dict['T'], how = 'left', 
	left_on = 'id', right_on = 'id')

lake_risk_ss.to_csv('results/risk table (ss).csv')

# management scenario effect
# zebra mussels
scenario_zm = dict()

for s in ['S', 'E', 'P', 'D', 'T']: 
	temp = np.zeros((len(scenarios_ann_zm), 13))
	for i in range(temp.shape[0]): 
		temp[i] = scenarios_ann_zm[i][s]
	scenario_zm.update({s: temp.tolist()})

with open('results/scenario effect (zm).txt', 'w') as fout:
    json.dump(scenario_zm, fout)

# starry stonewort
scenario_ss = dict()

for s in ['S', 'E', 'P', 'D', 'T']: 
	temp = np.zeros((len(scenarios_ann_ss), 9))
	for i in range(temp.shape[0]): 
		temp[i] = scenarios_ann_ss[i][s]
	scenario_ss.update({s: temp.tolist()})

with open('results/scenario effect (ss).txt', 'w') as fout:
    json.dump(scenario_ss, fout)
'''




