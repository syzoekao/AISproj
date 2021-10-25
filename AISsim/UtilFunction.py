import numpy as np
from scipy.special import comb
from itertools import product
import pandas as pd
import copy
import timeit
# import networkx as nx
import scipy.stats as stats


def infest_outcome_func(factor_ss, e_violate_zm, e_violate_ss, river_inf_zm, river_inf_ss, \
	back_suit_zm, back_suit_ss, \
	boat_net, river_o, river_d, river_w, lake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit): 
	# bbb = timeit.default_timer()

	'''	
	factor_ss=0.03623706
	e_violate_zm=0.02003358
	e_violate_ss=0.01008539
	river_inf_zm=0.07430393
	river_inf_ss=0.03270713
	back_suit_zm=0.13896991
	back_suit_ss=0.36345734

	np.random.seed(100)
	'''
	bbb = timeit.default_timer()

	infest_zm = copy.deepcopy(infest_zm)
	infest_ss = copy.deepcopy(infest_ss)
	infest_both = copy.deepcopy(infest_both)

	week = 26
	year = 2019-2012 
	# starting year is 2013 and the initial condition is based on the end of 2012 (zebra mussels)
	# the initial condition for starry stonewort is based on the infested lakes at the end of 2016
	time_horizon = week*year

	# infestation history in MN
	zm_count = {2012: 48, 2013: 63, 2014: 79, 2015: 94,  \
		2016: 126, 2017: 160, 2018: 194, 2019: 218}
	ss_count = {2016: 8,  2017: 10, 2018: 13, 2019: 14}

	# get parameters
	o_violate_zm = 0.0075
	o_violate_ss = 0.18 * factor_ss
	e_violate_zm = e_violate_zm
	e_violate_ss = e_violate_ss
	river_inf_zm = river_inf_zm
	river_inf_ss = river_inf_ss

	# print('in function')
	# print(np.round(factor_ss, 4), np.round(e_violate_zm, 4), np.round(e_violate_ss, 4), 
	# 	np.round(river_inf_zm, 4), np.round(river_inf_ss, 4))

	# results table: res = {'id', 'time', 'boat', 'river'} 
	res_zm = {'id': [], 'time': [], 'boat': [], 'river': []}
	res_ss = {'id': [], 'time': [], 'boat': [], 'river': []}

	# If the suitability score wasn't 1 (suitable), use the probability of the background suitbility. 
	# Then we sampled the suitability of these less suitable waters according to the probability. 
	# The sampled suitability was determined over the time horizon. 
	zm_suit[zm_suit < 1] = back_suit_zm
	ss_suit[ss_suit < 1] = back_suit_ss
	tmp_suit_zm0 = np.random.binomial(1, zm_suit)
	tmp_suit_ss0 = np.random.binomial(1, ss_suit)

	# Get the keys of the origin lakes that have outgoing boats 
	boat_key_o = np.array([k for k,v in boat_net.items()])

	# bbb = timeit.default_timer()
	for t in range(1, time_horizon+1): 
	# for t in range(1, 108): 
		# print(t)
		# get the index of the infested lakes 
		infest_lakes0 = np.where(infest_both == 1)[0]
		zm_inf = np.where(infest_zm == 1)[0]
		ss_inf = np.where(infest_ss == 1)[0] if t > 104 else np.array([], dtype = np.int)

		# infestation through boater network
		# list of new infested lakes through boater network 
		boat_zm = []
		boat_ss = []
		infest_lakes = copy.deepcopy(infest_lakes0)

		# aaa = timeit.default_timer()
		zm_lake = np.where(infest_zm == 1)[0].tolist()
		for x in zm_lake: 
			ix_d = []
			if x in boat_key_o: # only handle those lakes that are in the boater network
				# k is the index of the destination lake; w is the corresponding weight (traffic)
				k, w = [np.array([k for k, v in boat_net[x].items()]), np.array([v for k, v in boat_net[x].items()])]
				k = k.astype(np.intp)
				# sample the number of successful exiting and entry violation 
				# if the destination lake is suitable (1), the destination lake changes status to infested; however, if the destination lake is not suitable (0), the destination stay uninfested. 
				w = np.random.poisson(w)
				w = np.random.binomial(w, o_violate_zm*e_violate_zm)*tmp_suit_zm0[k]
				ix_d.append(k[w>0].tolist())
				ix_d = [val for sublist in ix_d for val in sublist]
				if len(ix_d) > 0: 
					ix_d = [y for y in ix_d if y != x] # removing the self loop
					boat_zm.append(ix_d)
		# print(timeit.default_timer() - aaa)
		boat_zm = [val for sublist in boat_zm for val in sublist]

		# starry stonewort wasn't found before 2016 (104 cycles in the model)
		if t > 104: 
			ss_lake = np.where(infest_ss == 1)[0].tolist()
			# aaa = timeit.default_timer()
			for x in ss_lake: # only handle those lakes that are in the boater network
				ix_d = []
				if x in boat_key_o: 
					# k is the index of the destination lake; w is the corresponding weight (traffic)
					k, w = [np.array([k for k, v in boat_net[x].items()]), np.array([v for k, v in boat_net[x].items()])]
					k = k.astype(np.intp)
					# sample the number of successful exiting and entry violation 
					# if the destination lake is suitable (1), the destination lake changes status to infested; however, if the destination lake is not suitable (0), the destination stay uninfested. 
					w = np.random.poisson(w)
					w = np.random.binomial(w, o_violate_ss*e_violate_ss)*tmp_suit_ss0[k]
					ix_d.append(k[w>0].tolist())
					ix_d = [val for sublist in ix_d for val in sublist]			
					if len(ix_d) > 0: 
						ix_d = [y for y in ix_d if y != x] # removing the self loop
						boat_ss.append(ix_d)
			# print(timeit.default_timer() - aaa)
			boat_ss = [val for sublist in boat_ss for val in sublist]

		# removing the repeats from the list of new infested lakes and the lakes that were already infested 
		boat_zm = np.unique(boat_zm)
		boat_zm = np.setdiff1d(boat_zm, zm_inf).tolist()
		boat_ss = np.unique(boat_ss)
		boat_ss = np.setdiff1d(boat_ss, ss_inf).tolist()

		# infestation through river network 
		# list of newly infested lakes via river network
		river_zm = []
		river_ss = []
		infest_lakes = copy.deepcopy(infest_lakes0)

		# aaa = timeit.default_timer()
		# the river network is shaped as an adjacency list 
		ix_o = np.where(np.in1d(river_o, infest_lakes) == True)[0] # get the index of the infested lakes that are also the origin lakes in the river network
		temp_weight = river_w[ix_o] # get the corresponding river weight 
		des = river_d[ix_o].astype(np.intp) # get the index of the corresponding destination lakes
		n_expand = stats.itemfreq(river_o[ix_o]) 
		ix_expand = np.repeat(n_expand[:,0], n_expand[:,1].astype(np.intp)).astype(np.intp) # for expand the index of the origin lakes
		zm_lake = infest_zm[ix_expand].astype(float) # get the zm infested status of the corresponding index of the origin lakes (ix_o)
		ss_lake = infest_ss[ix_expand] if t > 104 else np.zeros(zm_lake.shape) # get the ss infested status of the corresponding index of the origin lakes (ix_o)
		ss_lake = ss_lake.astype(float)
		tmp_suit_zm = tmp_suit_zm0[des] # get the zm suitability information for the destination lakes
		tmp_suit_ss = tmp_suit_ss0[des] # get the ss suitability information for the destination lakes

		# sample whether the destination lakes get infested or not
		out_river_zm = zm_lake*np.random.binomial(1, river_inf_zm*temp_weight)*tmp_suit_zm 
		out_river_ss = ss_lake*np.random.binomial(1, river_inf_ss*temp_weight)*tmp_suit_ss

		ix_zm = des[out_river_zm==1].tolist()
		ix_ss = des[out_river_ss==1] if t > 104 else np.array([])
		ix_ss = ix_ss.tolist()

		# add new infested lakes to the corresponding list of species 
		river_zm.append(ix_zm)
		river_ss.append(ix_ss)

		# print(timeit.default_timer() - aaa)
		# removing repeats and already infested lakes 
		river_zm = np.unique(river_zm)
		river_zm = np.setdiff1d(river_zm, zm_inf).tolist()
		river_ss = np.unique(river_ss)
		river_ss = np.setdiff1d(river_ss, ss_inf).tolist()

		# updating the infested status 
		temp_zm = np.unique(boat_zm + river_zm).astype(int)
		temp_ss = np.unique(boat_ss + river_ss).astype(int)

		infest_zm[temp_zm] = 1
		infest_ss[temp_ss] = 1
		infest_both = 1*(infest_zm + infest_ss > 0) 

		# recording new infested lakes
		out_zm = temp_zm.tolist()
		out_time = np.ceil(np.array(len(out_zm)*[t])/26)
		out_time = out_time.astype(int).tolist()
		out_boat = (1*np.in1d(out_zm, boat_zm)).tolist()
		out_river = (1*np.in1d(out_zm, river_zm)).tolist()
		res_zm['id'].extend(out_zm)
		res_zm['time'].extend(out_time)
		res_zm['boat'].extend(out_boat)
		res_zm['river'].extend(out_river)

		if t > 104: 
			out_ss = temp_ss.tolist()
			out_time = np.ceil(np.array(len(out_ss)*[t])/26)
			out_time = out_time.astype(int).tolist()
			out_boat = (1*np.in1d(out_ss, boat_ss)).tolist()
			out_river = (1*np.in1d(out_ss, river_ss)).tolist()
			res_ss['id'].extend(out_ss)
			res_ss['time'].extend(out_time)
			res_ss['boat'].extend(out_boat)
			res_ss['river'].extend(out_river)

		# if the # of infested lakes are crazy high, break the loop 
		if np.sum(infest_zm) >= 700: 
			return "toss away sample"
		if np.sum(infest_ss) >= 500: 
			return "toss away sample"

		# print(res_zm)
		# print(res_ss)
	# print(timeit.default_timer()-bbb)

	# print(np.sum(infest_zm))
	# calculating annual results
	res_zm = pd.DataFrame(res_zm)
	temp = res_zm.groupby('time').count().T
	ix = temp.loc['id'].index.values-1
	ix = ix.astype(np.intp)
	# print(ix)
	if ix.shape[0] > 0: 
		yy = np.array(year*[0])
		temp = temp.loc['id'].values
		yy[ix] = temp
		ann_zm = np.cumsum(yy) + zm_count[2012]
	else: 
		ann_zm = np.array(year*[zm_count[2012]])

	res_ss = pd.DataFrame(res_ss)
	temp = res_ss.groupby('time').count().T
	ix = temp.loc['id'].index.values-res_ss['time'].min()
	ix = ix.astype(np.intp)
	if ix.shape[0] > 0:
		yy = np.array((2019 - 2016) * [0])
		temp = temp.loc['id'].values
		yy[ix] = temp
		ann_ss = np.cumsum(yy) + ss_count[2016]
	else: 
		ann_ss = np.array((2019 - 2016) * [ss_count[2016]])

	ann_out = np.append(ann_zm, ann_ss)
	# print(ann_out)

	'''
	target_zm = np.array([val for key, val in zm_count.items() if key > 2012])
	target_ss = np.array([val for key, val in ss_count.items() if key > 2016])
	target_all = np.append(target_zm, target_ss)

	sd_zm = np.std(target_zm)
	sd_ss = np.std(target_ss)
	sd_all = np.append(np.repeat(sd_zm, year) , np.repeat(sd_ss, (2018-2016)))

	llk = np.sum(stats.norm.logpdf(ann_out, loc=target_all, scale=sd_all)) 
	'''
	return ann_out

#######################################################################


def pert(n, x_min, x_max, x_mode, lam = 4): 
	x_range = x_max-x_min
	if x_range == 0: 
		np.repeat(x_min, n)
	mu = ( x_min + x_max + lam * x_mode ) / ( lam + 2 )
	if mu == x_mode: 
		v = (lam/2) + 1
	else: 
		v = ((mu-x_min)*(2*x_mode-x_min-x_max))/(( x_mode - mu ) * ( x_max - x_min ))
	w = (v*(x_max - mu))/(mu-x_min)
	return stats.beta.rvs(v, w, size = n)*x_range + x_min


#######################################################################

'''
infest_zm0 = copy.deepcopy(infest_zm)
infest_ss0 = copy.deepcopy(infest_ss)
infest_both0 = copy.deepcopy(infest_both)
'''

def pre_infest_outcome_func(factor_ss, e_violate_zm, e_violate_ss, river_inf_zm, river_inf_ss, back_suit_zm, back_suit_ss, \
	boat_net, river_o, river_d, river_w, lake_id, infest_zm, 
	infest_ss, infest_both, zm_suit, ss_suit): 
	bbb = timeit.default_timer()

	'''
	factor_ss = tmp_param[0]
	e_violate_zm = tmp_param[1]
	e_violate_ss = tmp_param[2]
	river_inf_zm = tmp_param[3]
	river_inf_ss = tmp_param[4]
	back_suit_zm = tmp_param[5]
	back_suit_ss = tmp_param[6]
	'''

	infest_zm = copy.deepcopy(infest_zm)
	infest_ss = copy.deepcopy(infest_ss)
	infest_both = copy.deepcopy(infest_both)
	# print(np.sum(infest_zm))
	# print(np.sum(infest_both))

	week = 26
	year = 2019-2012 # starting year is 2013 and the initial condition is based on the end of 2012 (zebra mussels)
	time_horizon = week*year

	# infestation history in MN
	zm_count = {2012: 48, 2013: 63, 2014: 79, 2015: 94,  \
		2016: 126, 2017: 160, 2018: 194, 2019: 218}
	ss_count = {2016: 8,  2017: 10, 2018: 13, 2019: 14}

	o_violate_zm = 0.0075
	o_violate_ss = 0.18*factor_ss
	e_violate_zm = e_violate_zm
	e_violate_ss = e_violate_ss
	river_inf_zm = river_inf_zm
	river_inf_ss = river_inf_ss

	res_zm = {'id': [], 'time': [], 'boat': [], 'river': []}
	res_ss = {'id': [], 'time': [], 'boat': [], 'river': []}

	zm_suit[zm_suit < 1] = back_suit_zm
	ss_suit[ss_suit < 1] = back_suit_ss

	tmp_suit_zm0 = np.random.binomial(1, zm_suit)
	tmp_suit_ss0 = np.random.binomial(1, ss_suit)

	boat_key_o = np.array([k for k,v in boat_net.items()])

	for t in range(1, time_horizon+1): 
		# print(t)
		# print(np.sum(infest_both))
		# t = 1
		infest_lakes0 = np.where(infest_both == 1)[0]
		zm_inf = np.where(infest_zm == 1)[0]
		ss_inf = np.where(infest_ss == 1)[0] if t > 104 else np.array([], dtype = np.int)
		# sample boats to each lake of destination
		# att.loc[att['id'].isin(infest_lakes), ['lake_name']]

		# boater network 
		boat_zm = []
		boat_ss = []
		infest_lakes = copy.deepcopy(infest_lakes0)

		# aaa = timeit.default_timer()
		zm_lake = np.where(infest_zm == 1)[0].tolist()
		for x in zm_lake: 
			ix_d = []
			if x in boat_key_o: 
				k, w = [np.array([k for k, v in boat_net[x].items()]), np.array([v for k, v in boat_net[x].items()])]
				k = k.astype(np.intp)
				# sample the number of successful exiting and entry violation 
				# if the destination lake is suitable (1), the destination lake changes status to infested; however, if the destination lake is not suitable (0), the destination stay uninfested. 
				w = np.random.poisson(w)
				w = np.random.binomial(w, o_violate_zm*e_violate_zm)*tmp_suit_zm0[k]
				k = np.unique(k[w>0]).tolist()
				ix_d.append(k)
				ix_d = [val for sublist in ix_d for val in sublist]
				if len(ix_d) > 0: 
					ix_d = [y for y in ix_d if y != x]
					boat_zm.append(ix_d)
			# print(timeit.default_timer() - aaa)
		boat_zm = [val for sublist in boat_zm for val in sublist]

		if t > 104: 
			ss_lake = np.where(infest_ss == 1)[0].tolist()
			for x in ss_lake: 
				ix_d = []
				if x in boat_key_o: 
					k, w = [np.array([k for k, v in boat_net[x].items()]), np.array([v for k, v in boat_net[x].items()])]
					k = k.astype(np.intp)
					# sample the number of successful exiting and entry violation 
					# if the destination lake is suitable (1), the destination lake changes status to infested; however, if the destination lake is not suitable (0), the destination stay uninfested. 
					w = np.random.poisson(w)
					w = np.random.binomial(w, o_violate_ss*e_violate_ss)*tmp_suit_ss0[k]
					k = np.unique(k[w>0]).tolist()
					ix_d.append(k)
					ix_d = [val for sublist in ix_d for val in sublist]
					if len(ix_d) > 0: 
						ix_d = [y for y in ix_d if y != x]
						boat_ss.append(ix_d)
				# print(timeit.default_timer() - aaa)
			boat_ss = [val for sublist in boat_ss for val in sublist]
		# print(timeit.default_timer() - aaa)

		boat_zm = np.unique(boat_zm)
		boat_zm = np.setdiff1d(boat_zm, zm_inf).tolist()
		boat_ss = np.unique(boat_ss)
		boat_ss = np.setdiff1d(boat_ss, ss_inf).tolist()

		# river network 
		river_zm = []
		river_ss = []
		infest_lakes = copy.deepcopy(infest_lakes0)

		# aaa = timeit.default_timer()
		ix_o = np.where(np.in1d(river_o, infest_lakes) == True)[0]
		temp_weight = river_w[ix_o]
		des = river_d[ix_o]
		des = des.astype(np.intp)
		n_expand = stats.itemfreq(river_o[ix_o])
		ix_expand = np.repeat(n_expand[:,0], n_expand[:,1].astype(np.intp)).astype(np.intp)
		zm_lake = infest_zm[ix_expand].astype(float)
		ss_lake = infest_ss[ix_expand] if t > 104 else np.zeros(zm_lake.shape)
		ss_lake = ss_lake.astype(float)
		tmp_suit_zm = tmp_suit_zm0[des]
		tmp_suit_ss = tmp_suit_ss0[des]

		out_river_zm = zm_lake*np.random.binomial(1, river_inf_zm*temp_weight)*tmp_suit_zm
		out_river_ss = ss_lake*np.random.binomial(1, river_inf_ss*temp_weight)*tmp_suit_ss

		ix_zm = des[out_river_zm==1].tolist()
		ix_ss = des[out_river_ss==1] if t > 104 else np.array([])
		ix_ss = ix_ss.tolist()

		river_zm.append(ix_zm)
		river_ss.append(ix_ss)

		# print(timeit.default_timer() - aaa)

		river_zm = np.unique(river_zm)
		river_zm = np.setdiff1d(river_zm, zm_inf).tolist()
		river_ss = np.unique(river_ss)
		river_ss = np.setdiff1d(river_ss, ss_inf).tolist()

		# updating 
		temp_zm = np.unique(boat_zm + river_zm).astype(int)
		temp_ss = np.unique(boat_ss + river_ss).astype(int)

		infest_zm[temp_zm] = 1
		infest_ss[temp_ss] = 1
		infest_both = 1*(infest_zm + infest_ss > 0) 

		# recording new infested lakes
		out_zm = temp_zm.tolist()
		out_time = np.ceil(np.array(len(out_zm)*[t])/26)
		out_time = out_time.astype(int).tolist()
		out_boat = (1*np.in1d(out_zm, boat_zm)).tolist()
		out_river = (1*np.in1d(out_zm, river_zm)).tolist()
		res_zm['id'].extend(out_zm)
		res_zm['time'].extend(out_time)
		res_zm['boat'].extend(out_boat)
		res_zm['river'].extend(out_river)

		if t > 104: 
			out_ss = temp_ss.tolist()
			out_time = np.ceil(np.array(len(out_ss)*[t])/26)
			out_time = out_time.astype(int).tolist()
			out_boat = (1*np.in1d(out_ss, boat_ss)).tolist()
			out_river = (1*np.in1d(out_ss, river_ss)).tolist()
			res_ss['id'].extend(out_ss)
			res_ss['time'].extend(out_time)
			res_ss['boat'].extend(out_boat)
			res_ss['river'].extend(out_river)

	# calculating annual results
	res_zm = pd.DataFrame(res_zm)
	temp = res_zm.groupby('time').count().T
	ix = temp.loc['id'].index.values-1
	ix = ix.astype(np.intp)
	if ix.shape[0] > 0: 
		yy = np.array(year*[0])
		temp = temp.loc['id'].values
		yy[ix] = temp
		ann_zm = np.cumsum(yy) + zm_count[2012]
	else: 
		ann_zm = np.array(year*[zm_count[2012]])

	res_ss = pd.DataFrame(res_ss)
	temp = res_ss.groupby('time').count().T
	ix = temp.loc['id'].index.values-res_ss['time'].min()
	ix = ix.astype(np.intp)
	if ix.shape[0] > 0:
		yy = np.array((2019-2016)*[0])
		temp = temp.loc['id'].values
		yy[ix] = temp
		ann_ss = np.cumsum(yy) + ss_count[2016]
	else: 
		ann_ss = np.array((2019-2016)*[ss_count[2016]])

	res_zm = res_zm.reset_index().to_dict(orient='list')
	res_ss = res_ss.reset_index().to_dict(orient='list')
	del res_zm['index'], res_ss['index']

	ann_out = np.append(ann_zm, ann_ss)
	return ann_out, res_zm, infest_zm, tmp_suit_zm0, res_ss, infest_ss, tmp_suit_ss0

'''
factor_ss = tmp_param[0]
e_violate_zm = tmp_param[1]
e_violate_ss = tmp_param[2]
river_inf_zm = tmp_param[3]
river_inf_ss = tmp_param[4]
back_suit_zm = tmp_param[5]
back_suit_ss = tmp_param[6]
infest_zm0 = copy.deepcopy(infest_zm)
infest_ss0 = copy.deepcopy(infest_ss)
infest_both0 = copy.deepcopy(infest_both)
'''


def Scenario(factor_ss, e_violate_zm, e_violate_ss, river_inf_zm, river_inf_ss, back_suit_zm, back_suit_ss, \
	boat_net, river_o, river_d, river_w, lake_id, infest_zm, \
	infest_ss, infest_both, tmp_suit_zm, tmp_suit_ss, net_measure, \
	scenario = "StatusQuo", pReduceTraffic = 0.5, topN = 10, result2dict = True): 
	bbb = timeit.default_timer()

	infest_zm = copy.deepcopy(infest_zm)
	infest_ss = copy.deepcopy(infest_ss)
	infest_both = copy.deepcopy(infest_both)

	zm_count = {2012: 48, 2013: 63, 2014: 79, 2015: 94,  \
		2016: 126, 2017: 160, 2018: 194, 2019: 218}
	ss_count = {2016: 8,  2017: 10, 2018: 13, 2019: 14}

	# res_zm = copy.deepcopy(res_zm)
	# res_ss = copy.deepcopy(res_ss)
	res_zm = {'id': [], 'time': [], 'boat': [], 'river': []}
	res_ss = {'id': [], 'time': [], 'boat': [], 'river': []}
	# del res_zm['index'], res_ss['index']

	week = 26
	year = 2025-2019 # starting year is 2013 and the initial condition is based on the end of 2012 (zebra mussels)
	time_horizon = week*year
	
	EffectPerLake = np.zeros(len(lake_id)) # status quo and reduced traffice have no policy effect
	pReduceTraffic0 = 0.0

	if scenario == "Education": 
		x_min = 0.05 
		x_max = 0.3
		x_mean = 0.2
		EffectPerLake = pert(len(lake_id), x_min, x_max, x_mean)

	if scenario == "Penalty": 
		x_min = 0.25 
		x_max = 0.6
		x_mean = 0.4
		EffectPerLake = pert(len(lake_id), x_min, x_max, x_mean)

	if scenario == "MandDecon": 
		x_min = 0.25 
		x_max = 0.7
		x_mean = 0.5
		EffectPerLake = pert(len(lake_id), x_min, x_max, x_mean)

	if scenario == "ReduceTraffic": 
		pReduceTraffic0 = pReduceTraffic
		topID = lake_id

	if scenario == "TopControl":
		pReduceTraffic0 = pReduceTraffic
		topID = net_measure.loc[(net_measure['between'] <= topN) | (net_measure['degree'] <= topN), 'id'].to_numpy()

	boat_key_o = np.array([k for k,v in boat_net.items()])

	o_violate_zm = 0.0075
	o_violate_ss = 0.18*factor_ss
	e_violate_zm = e_violate_zm
	e_violate_ss = e_violate_ss
	river_inf_zm = river_inf_zm
	river_inf_ss = river_inf_ss

	# zm_suit[zm_suit <= back_suit_zm] = back_suit_zm
	# ss_suit[ss_suit <= back_suit_ss] = back_suit_ss

	tmp_suit_zm0 = copy.deepcopy(tmp_suit_zm)
	tmp_suit_ss0 = copy.deepcopy(tmp_suit_ss)

	for t in range(1, time_horizon + 1): 
	# for t in range(1, 10): 
		# print(t)
		# print(np.sum(infest_both))
		# t = 1
		infest_lakes0 = np.where(infest_both == 1)[0]
		zm_inf = np.where(infest_zm == 1)[0]
		ss_inf = np.where(infest_ss == 1)[0] 
		# sample boats to each lake of destination
		# att.loc[att['id'].isin(infest_lakes), ['lake_name']]

		# boater network 
		boat_zm = []
		boat_ss = []
		infest_lakes = copy.deepcopy(infest_lakes0)

		# aaa = timeit.default_timer()
		zm_lake = np.where(infest_zm == 1)[0].tolist()
		if scenario in ["ReduceTraffic", "TopControl"]: 
			for x in zm_lake: 
				# print(x)
				ix_d = []
				if x in boat_key_o: 
					k, w = [np.array([k for k, v in boat_net[x].items()]), np.array([v for k, v in boat_net[x].items()])]
					k = k.astype(np.intp)
					# sample the number of successful exiting and entry violation 
					# if the destination lake is suitable (1), the destination lake changes status to infested; however, if the destination lake is not suitable (0), the destination stay uninfested. 
					w = np.random.poisson(w)
					pReduceTraffic_new = pReduceTraffic0 * (x in topID) + 0 * (x not in topID)
					tot_boat = np.sum(w)*(1-pReduceTraffic_new)
					if tot_boat > 0: 
						prop = w/np.sum(w)
						w = np.random.multinomial(tot_boat, prop)
						w = np.random.binomial(w, o_violate_zm*e_violate_zm)*tmp_suit_zm0[k]
					k = np.unique(k[w>0]).tolist()
					ix_d.append(k)
					ix_d = [val for sublist in ix_d for val in sublist]					
					if len(ix_d) > 0: 
						ix_d = [y for y in ix_d if y != x]
						boat_zm.append(ix_d)
			boat_zm = [val for sublist in boat_zm for val in sublist]		
		else: 
			for x in zm_lake: 
				ix_d = []
				if x in boat_key_o: 
					k, w = [np.array([k for k, v in boat_net[x].items()]), np.array([v for k, v in boat_net[x].items()])]
					k = k.astype(np.intp)
					# sample the number of successful exiting and entry violation 
					# if the destination lake is suitable (1), the destination lake changes status to infested; however, if the destination lake is not suitable (0), the destination stay uninfested. 
					w = np.random.poisson(w)
					w = np.random.binomial(w, o_violate_zm*(1-EffectPerLake[k])*e_violate_zm)*tmp_suit_zm0[k]
					k = np.unique(k[w>0]).tolist()
					ix_d.append(k)		
					ix_d = [val for sublist in ix_d for val in sublist]

					if len(ix_d) > 0: 
						ix_d = [y for y in ix_d if y != x]
						boat_zm.append(ix_d)
			boat_zm = [val for sublist in boat_zm for val in sublist]

		ss_lake = np.where(infest_ss == 1)[0].tolist()
		if scenario == "ReduceTraffic": 
			for x in ss_lake: 
				# print(x)
				ix_d = []
				if x in boat_key_o: 
					k, w = [np.array([k for k, v in boat_net[x].items()]), np.array([v for k, v in boat_net[x].items()])]
					k = k.astype(np.intp)
					# sample the number of successful exiting and entry violation 
					# if the destination lake is suitable (1), the destination lake changes status to infested; however, if the destination lake is not suitable (0), the destination stay uninfested. 
					w = np.random.poisson(w)
					pReduceTraffic_new = pReduceTraffic0 * (x in topID) + 0 * (x not in topID)
					tot_boat = np.sum(w)*(1-pReduceTraffic_new)
					if tot_boat > 0: 
						prop = w/np.sum(w)
						w = np.random.multinomial(tot_boat, prop)
						w = np.random.binomial(w, o_violate_ss*e_violate_ss)*tmp_suit_ss0[k]
					k = np.unique(k[w>0]).tolist()
					ix_d.append(k)
					ix_d = [val for sublist in ix_d for val in sublist]				
					if len(ix_d) > 0: 
						ix_d = [y for y in ix_d if y != x]
						boat_ss.append(ix_d)
			boat_ss = [val for sublist in boat_ss for val in sublist]
		else: 
			for x in ss_lake: 
				ix_d = []
				if x in boat_key_o: 
					k, w = [np.array([k for k, v in boat_net[x].items()]), np.array([v for k, v in boat_net[x].items()])]
					k = k.astype(np.intp)
					# sample the number of successful exiting and entry violation 
					# if the destination lake is suitable (1), the destination lake changes status to infested; however, if the destination lake is not suitable (0), the destination stay uninfested. 
					w = np.random.poisson(w)
					w = np.random.binomial(w, o_violate_ss*(1-EffectPerLake[k])*e_violate_ss)*tmp_suit_ss0[k]
					k = np.unique(k[w>0]).tolist()
					ix_d.append(k)		
					ix_d = [val for sublist in ix_d for val in sublist]

					if len(ix_d) > 0: 
						ix_d = [y for y in ix_d if y != x]
						boat_ss.append(ix_d)
			boat_ss = [val for sublist in boat_ss for val in sublist]

		# print(timeit.default_timer() - aaa)

		boat_zm = np.unique(boat_zm)
		boat_zm = np.setdiff1d(boat_zm, zm_inf).tolist()
		boat_ss = np.unique(boat_ss)
		boat_ss = np.setdiff1d(boat_ss, ss_inf).tolist()

		# river network 
		river_zm = []
		river_ss = []
		infest_lakes = copy.deepcopy(infest_lakes0)

		# aaa = timeit.default_timer()
		ix_o = np.where(np.in1d(river_o, infest_lakes) == True)[0]
		temp_weight = river_w[ix_o]
		des = river_d[ix_o]
		des = des.astype(np.intp)
		n_expand = stats.itemfreq(river_o[ix_o])
		ix_expand = np.repeat(n_expand[:,0], n_expand[:,1].astype(np.intp)).astype(np.intp)
		zm_lake = infest_zm[ix_expand].astype(float)
		ss_lake = infest_ss[ix_expand].astype(float)
		tmp_suit_zm = tmp_suit_zm0[des]
		tmp_suit_ss = tmp_suit_ss0[des]

		out_river_zm = zm_lake*np.random.binomial(1, river_inf_zm*temp_weight)*tmp_suit_zm
		out_river_ss = ss_lake*np.random.binomial(1, river_inf_ss*temp_weight)*tmp_suit_ss

		ix_zm = des[out_river_zm==1].tolist()
		ix_ss = des[out_river_ss==1].tolist()

		river_zm.append(ix_zm)
		river_ss.append(ix_ss)

		# print(timeit.default_timer() - aaa)

		river_zm = np.unique(river_zm)
		river_zm = np.setdiff1d(river_zm, zm_inf).tolist()
		river_ss = np.unique(river_ss)
		river_ss = np.setdiff1d(river_ss, ss_inf).tolist()

		# updating 
		temp_zm = np.unique(boat_zm + river_zm).astype(int)
		temp_ss = np.unique(boat_ss + river_ss).astype(int)

		infest_zm[temp_zm] = 1
		infest_ss[temp_ss] = 1
		infest_both = 1*(infest_zm + infest_ss > 0) 

		# recording new infested lakes
		out_zm = temp_zm.tolist()
		out_time = np.ceil(np.array(len(out_zm)*[t])/26)
		out_time = out_time.astype(int).tolist()
		out_boat = (1*np.in1d(out_zm, boat_zm)).tolist()
		out_river = (1*np.in1d(out_zm, river_zm)).tolist()
		res_zm['id'].extend(out_zm)
		res_zm['time'].extend(out_time)
		res_zm['boat'].extend(out_boat)
		res_zm['river'].extend(out_river)

		out_ss = temp_ss.tolist()
		out_time = np.ceil(np.array(len(out_ss)*[t])/26)
		out_time = out_time.astype(int).tolist()
		out_boat = (1*np.in1d(out_ss, boat_ss)).tolist()
		out_river = (1*np.in1d(out_ss, river_ss)).tolist()
		res_ss['id'].extend(out_ss)
		res_ss['time'].extend(out_time)
		res_ss['boat'].extend(out_boat)
		res_ss['river'].extend(out_river)
	
	# calculating annual results
	res_zm = pd.DataFrame(res_zm)
	temp = res_zm.groupby('time').count().T
	ix = temp.loc['id'].index.values-1
	ix = ix.astype(np.intp)
	# print(ix)
	if ix.shape[0] > 0: 
		yy = np.array((year)*[0])
		temp = temp.loc['id'].values
		yy[ix] = temp
		ann_zm = np.cumsum(yy) + zm_count[2019]
	else: 
		ann_zm = np.array((year)*[zm_count[2019]])

	res_ss = pd.DataFrame(res_ss)
	temp = res_ss.groupby('time').count().T
	ix = temp.loc['id'].index.values-1
	ix = ix.astype(np.intp)
	# print(ix)
	if ix.shape[0] > 0: 
		yy = np.array((year)*[0])
		temp = temp.loc['id'].values
		yy[ix] = temp
		ann_ss = np.cumsum(yy) + ss_count[2019]
	else: 
		ann_ss = np.array((year)*[ss_count[2019]])

	if result2dict == True: 
		res_zm = res_zm.reset_index().to_dict(orient='list')
		res_ss = res_ss.reset_index().to_dict(orient='list')
		del res_zm['index'], res_ss['index']

	return ann_zm, res_zm, ann_ss, res_ss

