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
import AISsim.UtilFunction as util
import AISsim.AdaptMCMC as AdaptMCMC
os.chdir("/home/ennse/kaoxx085/fish/virsim") 
cwd = os.getcwd() 
print(cwd) 
         
nfile = 1
         
att = pd.read_csv('data/lake_attribute.csv')
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
infest_zm = copy.deepcopy(att['infest_zm'].values)
infest_ss = copy.deepcopy(att['infest_ss'].values)
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
	k2, v2 = [[key for key, val in boat_dict[k].items() if key != k], 
	[val for key, val in boat_dict[k].items() if key != k]]
	k2 = [int(x) for x in k2]
	v2 = [x for x in v2]
	if len(k2) > 0: 
		boat_net.update({int(k): dict(zip(k2, v2))})
del boat_dict
     
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
	print(i) 

	tmp_param = sim_param[np.random.randint(1, sim_param.shape[0]+1)] 

	aa = timeit.default_timer() 
	ann_out, res_zm0, infest_zm0, tmp_suit_zm0, res_ss0, infest_ss0, tmp_suit_ss0 = util.pre_infest_outcome_func(factor_ss = tmp_param[0], 
	e_violate_zm = tmp_param[1], e_violate_ss = tmp_param[2], 
	river_inf_zm = tmp_param[3], river_inf_ss = tmp_param[4], 
	back_suit_zm = tmp_param[5], back_suit_ss = tmp_param[6], 
	boat_net = boat_net, river_o=river_o, river_d=river_d, river_w=river_w, 
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
	boat_net = boat_net, river_o=river_o, river_d=river_d, river_w=river_w, 
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
	boat_net = boat_net, river_o=river_o, river_d=river_d, river_w=river_w, 
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
	boat_net = boat_net, river_o=river_o, river_d=river_d, river_w=river_w, 
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
	boat_net = boat_net, river_o=river_o, river_d=river_d, river_w=river_w, 
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
	boat_net = boat_net, river_o=river_o, river_d=river_d, river_w=river_w, 
	lake_id=lake_id, infest_zm=infest_zm2018, infest_ss=infest_ss2018, infest_both=infest_both2018, 
	tmp_suit_zm=tmp_suit_zm0, tmp_suit_ss=tmp_suit_ss0, 
	scenario = "ReduceTraffic", result2dict = True)
	print(timeit.default_timer()-aa) 
         
	scenarios_ann_zm.append({'S': ann_zm.tolist(), 'E': ann_zm_e.tolist(), 'P': ann_zm_p.tolist(), 'D': ann_zm_d.tolist(),  
	'T': ann_zm_t.tolist()}) 

	scenarios_ann_ss.append({'S': ann_ss.tolist(), 'E': ann_ss_e.tolist(), 'P': ann_ss_p.tolist(), 'D': ann_ss_d.tolist(),  
	'T': ann_ss_t.tolist()}) 

	scenarios_res_zm.append({'S': res_zm, 'E': res_zm_e, 'P': res_zm_p, 'D': res_zm_d, 'T': res_zm_t}) 
	scenarios_res_ss.append({'S': res_ss, 'E': res_ss_e, 'P': res_ss_p, 'D': res_ss_d, 'T': res_ss_t}) 

	del ann_out, res_zm0, infest_zm0, tmp_suit_zm0, res_ss0, infest_ss0, tmp_suit_ss0,  
	ann_zm, ann_zm_e, ann_zm_p, ann_zm_d, ann_zm_t, ann_ss, ann_ss_e, ann_ss_p, ann_ss_d, ann_ss_t,  
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
         
