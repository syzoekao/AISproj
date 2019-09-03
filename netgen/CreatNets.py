'''
Creating 20 networks and model performance (get true positive and true negative)
'''

import numpy as np
from scipy.misc import comb
import scipy
from itertools import product
import pandas as pd
import json
import timeit
import os
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)

aa = timeit.default_timer()
move = pd.read_csv('data/movement_for_stat.csv')
print(timeit.default_timer() - aa)

move = move.drop(['Unnamed: 0'], axis = 1)
print(move.shape)

move = move[['dow_origin', 'dow_destination', 'normBoatsExtra', 'dumBoat']]

att = pd.read_csv('data/lake_attribute.csv')
inspID = att.loc[att['inspect'] == 1, 'dow'].to_numpy()

predprob = np.load('data/predprob.npy')
predboat = np.load('data/predboat.npy')

mean_boat = np.zeros(20)
max_boat = np.zeros(20)
std_boat = np.zeros(20)
mean_boat_sim = np.zeros(20)
max_boat_sim = np.zeros(20)
std_boat_sim = np.zeros(20)
true_pos_vec = np.zeros(20)
true_neg_vec = np.zeros(20)

n_true_pos = np.zeros((inspID.shape[0], 20))
n_false_neg = np.zeros((inspID.shape[0], 20))
n_true_neg = np.zeros((inspID.shape[0], 20))
n_false_pos = np.zeros((inspID.shape[0], 20))
true_pos_rate = np.zeros((inspID.shape[0], 20))
true_neg_rate = np.zeros((inspID.shape[0], 20))

for run in range(0, 20): 
	aa = timeit.default_timer()
	temp_samp = np.random.binomial(1, p = predprob)
	temp_dum = (predprob >= 0.35) + (predprob < 0.35) * temp_samp
	np.fill_diagonal(temp_dum, 1)
	out_net = predboat * temp_dum

	ix = np.where(out_net > 0)
	boat_o = ix[0]
	boat_d = ix[1]
	boat_w = np.random.poisson(out_net[ix])
	boat_w[boat_w < 1] = 1

	temp_move = {'id_origin': boat_o, 'id_destination': boat_d, 'weight': boat_w}
	temp_move = pd.DataFrame(temp_move)
	temp_move = pd.merge(temp_move, att[['dow', 'id']], how = 'left', left_on = 'id_origin', right_on = 'id')
	temp_move.drop(["id", "id_origin"], axis = 1, inplace = True)
	temp_move.rename({'dow': 'dow_origin'}, axis = 1, inplace = True)
	temp_move = pd.merge(temp_move, att[['dow', 'id']], how = 'left', left_on = 'id_destination', right_on = 'id')
	temp_move.drop(["id", "id_destination"], axis = 1, inplace = True)
	temp_move.rename({'dow': 'dow_destination'}, axis = 1, inplace = True)
	temp_move = temp_move[['dow_origin', 'dow_destination', 'weight']]
	temp_move.to_csv('data/Annual boater net/boats'+str(run + 1)+'.csv', index = False)

	tmp_move = pd.merge(move, temp_move, how = 'left', on = ['dow_origin', 'dow_destination'])
	tmp_move['weight'].fillna(0, inplace=True)
	tmp_move[['sim_conn']] = 1 * (tmp_move[['weight']] > 0)

	tmp = tmp_move.loc[tmp_move['dumBoat'] == 1, 'weight'].describe()
	mean_boat[run] = tmp.ix['mean']
	max_boat[run] = tmp.ix['max']
	std_boat[run] = tmp.ix['std']

	tmp = tmp_move.loc[tmp_move['sim_conn'] == 1, 'weight'].describe()
	mean_boat_sim[run] = tmp.ix['mean']
	max_boat_sim[run] = tmp.ix['max']
	std_boat_sim[run] = tmp.ix['std']

	cross_table = pd.crosstab(index=tmp_move['sim_conn'], 
		columns=tmp_move['dumBoat'], margins=True)

	tmp_table = cross_table/cross_table.ix['All']

	true_pos_vec[run] = tmp_table.iloc[1,1]
	true_neg_vec[run] = tmp_table.iloc[0,0]

	for i in range(inspID.shape[0]): 
		j = inspID[i]
		tempSet = tmp_move.loc[(tmp_move['dow_origin'] == j) | (tmp_move['dow_destination'] == j)]
		n_true_pos0 = tempSet.loc[(tempSet['sim_conn'] == 1) & (tempSet['dumBoat']==1)].shape[0]
		n_false_neg0 = tempSet.loc[(tempSet['sim_conn'] == 0) & (tempSet['dumBoat']==1)].shape[0]
		n_true_neg0 = tempSet.loc[(tempSet['sim_conn'] == 0) & (tempSet['dumBoat']==0)].shape[0]
		n_false_pos0 = tempSet.loc[(tempSet['sim_conn'] == 1) & (tempSet['dumBoat']==0)].shape[0]
		if (n_true_pos0 + n_false_neg0) > 0: 
			true_pos_rate0 = n_true_pos0/(n_true_pos0 + n_false_neg0)
		else: 
			true_pos_rate0 = 0
		if (n_true_neg0 + n_false_pos0) > 0: 
			true_neg_rate0 = n_true_neg0/(n_true_neg0 + n_false_pos0)
		else: 
			true_neg_rate0 = 0

		n_true_pos[i, run] = n_true_pos0
		n_false_neg[i, run] = n_false_neg0
		n_true_neg[i, run] = n_true_neg0
		n_false_pos[i, run] = n_false_pos0
		true_pos_rate[i, run] = true_pos_rate0
		true_neg_rate[i, run] = true_neg_rate0

	print(timeit.default_timer()-aa)


sim_net_data = pd.DataFrame({"mean_boat": mean_boat, 'max_boat': max_boat, 'std_boat': std_boat, 
	"mean_boat_sim": mean_boat_sim, 'max_boat_sim': max_boat_sim, 'std_boat_sim': std_boat_sim, 
	'true_pos': true_pos_vec, 'true_neg': true_neg_vec})
sim_net_data.to_csv('data/Annual boater net/sim_net_summary.csv')

true_pos_rate_mean = np.mean(true_pos_rate, axis = 1)
true_pos_rate_std = np.std(true_pos_rate, axis = 1)
true_neg_rate_mean = np.mean(true_neg_rate, axis = 1)
true_neg_rate_std = np.std(true_neg_rate, axis = 1)

insp_out = np.array([inspID, true_pos_rate_mean, true_pos_rate_std, true_neg_rate_mean, true_neg_rate_std]).T
insp_out = pd.DataFrame(insp_out)
insp_out.columns = ['dow', 'true_pos_rate_mean', 'true_pos_rate_std', 'true_neg_rate_mean', 'true_neg_rate_std']

pred_by_lakes = pd.merge(att[['dow', 'lake_name', 'inspect', 'infest', 'infest_zm', 'infest_ss']], insp_out, how = 'left', on = 'dow')
pred_by_lakes.to_csv('data/Annual boater net/prediction by lakes.csv', index = False)




'''
creating network for calibration (weekly)
'''
import numpy as np
from scipy.misc import comb
import scipy
from itertools import product
import pandas as pd
import json
import timeit
import statsmodels
import os
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)

att = pd.read_csv('data/lake_attribute.csv')
att = att[['dow', 'lake_name', 'acre', 'utm_x', 'utm_y', 'county', 'county.name', 'infest', 'inspect', \
'infest.zm', 'infest.ss', 'zm_suit', 'ss_suit', 'id']]
traffic = pd.read_csv("data/traffics.csv")
att = pd.merge(att, traffic[['DOW', 'boats']], how = 'left', left_on = 'dow', right_on = 'DOW')
att.columns = ['dow', 'lake_name', 'acre', 'utm_x', 'utm_y', 'county', 'county.name', 'infest', 'inspect', 
'infest.zm', 'infest.ss', 'zm_suit', 'ss_suit', 'id', 'DOW', 'boat_ann']

p_self = pd.read_csv('data/p_self_boat.csv')
weight = np.load('data/avg_weight.npy')
np.fill_diagonal(weight, 0)

tot = np.round(att[['boat_ann']])
p_out = 1-p_self

# dummy = (weight > 1)
# weight = dummy*weight
sum_weight = np.sum(weight, axis = 1)
sum_weight = np.tile(sum_weight, (weight.shape[0], 1)).T 
prop_weight = weight/sum_weight
# prop_weight[np.isnan(prop_weight)] = 0
tmp = tot['boat_ann']*p_out['p_self_boat']
tmp = np.tile(tmp, (weight.shape[0], 1)).T

boat_net = prop_weight*tmp

tmp_diag = np.array(tot['boat_ann']*p_self['p_self_boat'])
kk = np.diag_indices(boat_net.shape[0])
boat_net[kk[0], kk[1]] = tmp_diag

boat_net = boat_net * (boat_net >= 1)
boat_net = boat_net/26 # rescale to weekly boats
boat_net = np.ceil(boat_net).astype(int)

np.save("data/boat_net", boat_net)

boat_dict = dict()
# make boater net dictionary
for ix_o in range(boat_net.shape[0]): 
	# ix_o = 0
	weight = boat_net[ix_o]
	ix_d = np.where(weight >= 1)[0].tolist()
	weight = weight[ix_d].tolist()
	ix_d = [str(x) for x in ix_d]
	if len(ix_d) > 0: 
		temp = {str(ix_o): dict(zip(ix_d, weight))}
		boat_dict.update(temp)

with open('data/boat_dict.txt', 'w') as fout:
    json.dump(boat_dict, fout)


'''
Networks for post simulation
'''

import numpy as np
from scipy.misc import comb
import scipy
from itertools import product
import pandas as pd
import json
import timeit
import statsmodels
import os
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)

att = pd.read_csv('data/lake_attribute.csv')
att = att[['dow', 'lake_name', 'acre', 'utm_x', 'utm_y', 'county', 'county.name', 'infest', 'inspect', \
'infest.zm', 'infest.ss', 'zm_suit', 'ss_suit', 'id']]
traffic = pd.read_csv("data/traffics.csv")
att = pd.merge(att, traffic[['DOW', 'boats']], how = 'left', left_on = 'dow', right_on = 'DOW')
att.columns = ['dow', 'lake_name', 'acre', 'utm_x', 'utm_y', 'county', 'county.name', 'infest', 'inspect', 
'infest.zm', 'infest.ss', 'zm_suit', 'ss_suit', 'id', 'DOW', 'boat_ann']
tot = np.round(att['boat_ann'])
logit = np.load('data/logit.npy')
gamma = np.load('data/gamma_pred.npy')
p_self = pd.read_csv('data/p_self_boat.csv')

inspID = att.loc[att['inspect'] == 1, 'id'].values.tolist()

del traffic

for j in range(1, 21): 
	print(j)
	aa = timeit.default_timer()
	temp_dum = np.random.binomial(n = 1, p = logit)
	# print(timeit.default_timer()-aa)
	np.fill_diagonal(temp_dum, 1)

	out_net = gamma*temp_dum
	sum_out = np.sum(out_net, axis = 1)
	sum_out = np.tile(sum_out, (out_net.shape[0], 1)).T 
	self_loop = np.tile(p_self.values.T, (out_net.shape[0], 1)).T 
	p_out = out_net/sum_out * (1-self_loop)
	p_out[np.isnan(p_out)] = 0

	kk = np.diag_indices(p_out.shape[0])
	p_out[kk[0], kk[1]] = p_self.values.T

	tot_mat = np.tile(tot.values.T, (out_net.shape[0], 1)).T

	boat_net = tot_mat*p_out
	boat_net[np.isnan(boat_net)] = 0

	boat_net[np.where((boat_net<1) & (boat_net>0))] = 0
	small_ix = np.where((boat_net<26) & (boat_net>0))

	boat_net = np.round(boat_net/26, 0)

	boat_dict = dict()
	# make boater net dictionary
	for ix_o in range(boat_net.shape[0]): 
		# ix_o = 0
		weight = boat_net[ix_o]
		ix_d = np.where(weight >= 1)[0].tolist()
		weight = weight[ix_d].tolist()
		ix_d = [str(x) for x in ix_d]
		if len(ix_d) > 0: 
			temp = {str(ix_o): dict(zip(ix_d, weight))}
			boat_dict.update(temp)

	with open('data/boat_dict'+str(j)+'.txt', 'w') as fout:
	    json.dump(boat_dict, fout)

	temp_prob = np.zeros(logit.shape)
	temp_prob[small_ix] = logit[small_ix]
	temp_prob[temp_prob < 0.005] = 0
	small_prob = dict()

	for ix_o in range(temp_prob.shape[0]): 
		# ix_o = 0
		weight = temp_prob[ix_o]
		ix_d = np.where(weight > 0)[0].tolist()
		weight = weight[ix_d].tolist()
		ix_d = [str(x) for x in ix_d]
		if len(ix_d) > 0: 
			temp = {str(ix_o): dict(zip(ix_d, weight))}
			small_prob.update(temp)

	with open('data/small_prob'+str(j)+'.txt', 'w') as fout:
	    json.dump(small_prob, fout)

	print(timeit.default_timer() - aa)




'''
organizing river network
'''

# river network
att = pd.read_csv('data/lake_attribute.csv')
print(list(att.columns.values))

n_row = att.shape[0]
river = pd.read_csv('data/river_net.csv')
river = river[['dow.origin', 'dow.destination', 'weight', 'inverse_weight']]
river = pd.merge(river, att[['dow', 'id']], how = 'left', left_on = 'dow.origin', right_on = 'dow')
river = river[['dow.origin', 'dow.destination', 'weight', 'inverse_weight', 'id']]
river.columns = ['dow.origin', 'dow.destination', 'weight', 'inverse_weight', 'fromID']
river = pd.merge(river, att[['dow', 'id']], how = 'left', left_on = 'dow.destination', right_on = 'dow')
river = river[['dow.origin', 'dow.destination', 'weight', 'inverse_weight', 'fromID', 'id']]
river.columns = ['dow.origin', 'dow.destination', 'weight', 'inverse_weight', 'fromID', 'toID']


river_mat = np.empty((n_row, n_row))

for i in river.index:
	# print(i)
	# print(river.iloc[i]['inverse_weight'])
	river_mat[int(river.iloc[i]['fromID'])][int(river.iloc[i]['toID'])] = river.iloc[i]['inverse_weight']

river_dict = dict()
for i in range(river_mat.shape[0]): 
	ix = np.where(river_mat[i] > 0)[0]
	if ix.shape[0] > 0: 
		val = river_mat[i][ix]
		ix = [str(x) for x in ix]
		river_dict.update({str(i): dict(zip(ix, val))})

with open('data/river_dict.txt', 'w') as fout:
    json.dump(river_dict, fout)

import copy
with open('data/river_dict.txt') as json_file:  
    river_dict = json.load(json_file)

kk = {int(k1): {int(k2): v for k2, v in k2.items()} for k1, k2 in river_dict.items()}
river = copy.deepcopy(kk)

river_key = [k for k,v in river.items()]
river_o = np.array([])
river_d = np.array([])
river_w = np.array([])
for k1 in river_key: 
	k2 = [k for k, v in river[k1].items()]
	v2 = [v for k, v in river[k1].items()]
	temp_len = len(k2)
	empty_o = np.repeat(k1, temp_len).astype(int)
	empty_d = np.array(k2).astype(int)
	empty_w = np.array(v2)
	river_o = np.append(river_o, empty_o)
	river_d = np.append(river_d, empty_d)
	river_w = np.append(river_w, empty_w)
river_o = river_o.astype(int)
river_d = river_d.astype(int)

river_net = np.column_stack((river_o, river_d, river_w))
river_net = pd.DataFrame(river_net)
river_net.columns = ['origin', 'destination', 'weight']
river_net['origin'] = river_net['origin'].astype(int)
river_net['destination'] = river_net['destination'].astype(int)
river_net = pd.merge(river_net, att[['dow', 'id']], how = 'left', left_on = ['origin'], right_on = ['id'])
river_net.columns = ['origin', 'destination', 'weight', 'dow.origin', 'id.origin']
river_net = pd.merge(river_net, att[['dow', 'id']], how = 'left', left_on = ['destination'], right_on = ['id'])
river_net.columns = ['origin', 'destination', 'weight', 'dow.origin', 'id.origin', 'dow.destination', "id.destination"]
river_net.to_csv('data/river_net_sim.csv')


'''
Generate county movements
'''
import numpy as np
import pandas as pd
import json
import timeit
import os
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)


def extract_sim_boat(att, county, path = 'data/Annual boater net'): 
	# county = 'ramsey'
	# path = 'data/Annual boater net'
	if county == 'ramsey': 
		att.loc[att['dow'] == 82016700, ['county', 'county.name']] = [62, county]
	if county == 'crow wing': 
		att.loc[att['dow'] == 11030500, ['county', 'county.name']] = [18, county]
		att.loc[att['dow'] == 48000200, ['county', 'county.name']] = [18, county]
	if county == 'stearns': 
		att.loc[att['dow'] == 86025200, ['county', 'county.name']] = [73, county]

	nLake = att.loc[att['county.name']==county, 'dow'].shape[0] + 4
	lake_set = np.append(att.loc[att['county.name']==county, 'dow'], np.array([0, 1, 2, 3])) 
	# 0: zm infested lakes in other counties; 1: zm uninfested lakes in other counties;
	# 2: ss infested lakes in other counties; 3: ss uninfested lakes in other counties

	boatdf = pd.DataFrame({'dow_origin': np.repeat(lake_set, nLake), 
		'dow_destination': np.tile(lake_set, nLake)})

	# Get attributes of the origin lakes
	attribute_set = ['dow', 'lake_name', 'county.name', 'zm2018', 'ss2018']
	boatdf = pd.merge(boatdf, att[attribute_set], how = 'left', left_on=['dow_origin'], right_on=['dow'])
	boatdf.drop(columns=['dow'], inplace = True)
	boatdf.columns = ['dow_origin', 'dow_destination', 'lake_origin', 'county_origin', 'zm_origin', 'ss_origin']
	boatdf['zm_origin'].fillna(0, inplace = True)
	boatdf.loc[boatdf['dow_origin'] == 0, 'zm_origin'] = 1
	boatdf['ss_origin'].fillna(0, inplace = True)
	boatdf.loc[boatdf['dow_origin'] == 2, 'ss_origin'] = 1

	# Get attributes of the destination lakes
	boatdf = pd.merge(boatdf, att[attribute_set], how = 'left', left_on=['dow_destination'], right_on=['dow'])
	boatdf.drop(columns=['dow'], inplace = True)
	boatdf.columns = ['dow_origin', 'dow_destination', 'lake_origin', 'county_origin', 'zm_origin', 
	'ss_origin', 'lake_destination', 'county_destination', 'zm_destination', 'ss_destination']
	boatdf['zm_destination'].fillna(0, inplace = True)
	boatdf.loc[boatdf['dow_destination'] == 0, 'zm_destination'] = 1
	boatdf['ss_destination'].fillna(0, inplace = True)
	boatdf.loc[boatdf['dow_destination'] == 2, 'ss_destination'] = 1

	boatdf.loc[boatdf['dow_origin'].isin([0, 1, 2, 3]), 'county_origin'] = 'not '+county
	boatdf.loc[boatdf['dow_destination'].isin([0, 1, 2, 3]), 'county_destination'] = 'not '+county

	for i in range(1, 21): 
		# print(i)
		tmp_boat = pd.read_csv(path + '/boats'+str(i) +'.csv')
		tmp_boat = tmp_boat[(tmp_boat['dow_origin'].isin(lake_set)) | (tmp_boat['dow_destination'].isin(lake_set))]
		tmp_boat.columns = ['dow_origin', 'dow_destination', 'weight']
		tmp_boat = pd.merge(tmp_boat, att[['dow', 'county.name', 'zm2018', 'ss2018']], how = 'left', left_on=['dow_origin'], right_on=['dow'])
		tmp_boat.drop(columns=['dow'], inplace = True)
		tmp_boat.columns = ['dow_origin', 'dow_destination', 'weight', 'county_origin', 'zm_origin', 'ss_origin']
		tmp_boat = pd.merge(tmp_boat, att[['dow', 'county.name', 'zm2018', 'ss2018']], how = 'left', left_on=['dow_destination'], right_on=['dow'])
		tmp_boat.drop(columns=['dow'], inplace = True)
		tmp_boat.columns = ['dow_origin', 'dow_destination', 'weight', 'county_origin', 'zm_origin',
		       'ss_origin', 'county_destination', 'zm_destination', 'ss_destination']

		tmp_boat.loc[tmp_boat['county_origin'] != county, 'county_origin'] = 'not ' + county
		tmp_boat.loc[tmp_boat['county_destination'] != county, 'county_destination'] = 'not ' + county

		tmp_boat.loc[(tmp_boat['county_origin'] == 'not ' + county) & (tmp_boat['zm_origin'] == 1), 'dow_origin'] = 0
		tmp_boat.loc[(tmp_boat['county_origin'] == 'not ' + county) & (tmp_boat['zm_origin'] == 0), 'dow_origin'] = 1
		tmp_boat.loc[(tmp_boat['county_origin'] == 'not ' + county) & (tmp_boat['ss_origin'] == 1), 'dow_origin'] = 2
		tmp_boat.loc[(tmp_boat['county_origin'] == 'not ' + county) & (tmp_boat['ss_origin'] == 0), 'dow_origin'] = 3

		tmp_boat.loc[(tmp_boat['county_destination'] == 'not ' + county) & (tmp_boat['zm_destination'] == 1), 'dow_destination'] = 0
		tmp_boat.loc[(tmp_boat['county_destination'] == 'not ' + county) & (tmp_boat['zm_destination'] == 0), 'dow_destination'] = 1
		tmp_boat.loc[(tmp_boat['county_destination'] == 'not ' + county) & (tmp_boat['ss_destination'] == 1), 'dow_destination'] = 2
		tmp_boat.loc[(tmp_boat['county_destination'] == 'not ' + county) & (tmp_boat['ss_destination'] == 0), 'dow_destination'] = 3

		tmp_sum = tmp_boat.groupby(['dow_origin','dow_destination'])['weight'].agg('sum').reset_index()

		boatdf = pd.merge(boatdf, tmp_sum, how = 'left', left_on = ['dow_origin', 'dow_destination'], right_on = ['dow_origin', 'dow_destination'])

		boatdf=boatdf.rename(columns = {'weight': 'weight'+str(i)})
		boatdf['weight'+str(i)].fillna(0, inplace = True)

	boatdf.dow_origin = boatdf.dow_origin.astype(str)
	boatdf.dow_destination = boatdf.dow_destination.astype(str)
	boatdf = boatdf[(~boatdf['dow_origin'].isin(['0', '1', '2', '3']))|(~boatdf['dow_destination'].isin(['0', '1', '2', '3']))]

	boatdf.loc[boatdf['dow_origin']=='0', 'dow_origin'] = 'zm infested other county'
	boatdf.loc[boatdf['dow_origin']=='1', 'dow_origin'] = 'not zm infested other county'
	boatdf.loc[boatdf['dow_origin']=='2', 'dow_origin'] = 'ss infested other county'
	boatdf.loc[boatdf['dow_origin']=='3', 'dow_origin'] = 'not ss infested other county'

	boatdf.loc[boatdf['dow_destination']=='0', 'dow_destination'] = 'zm infested other county'
	boatdf.loc[boatdf['dow_destination']=='1', 'dow_destination'] = 'not zm infested other county'
	boatdf.loc[boatdf['dow_destination']=='2', 'dow_destination'] = 'ss infested other county'
	boatdf.loc[boatdf['dow_destination']=='3', 'dow_destination'] = 'not ss infested other county'

	return boatdf


att = pd.read_csv('data/lake_attribute.csv')
att = att[['dow', 'lake_name', 'acre', 'utm_x', 'utm_y', 'county', 'county.name', 'infest', 'inspect', \
'infest.zm', 'infest.ss', 'zm_suit', 'ss_suit', 'id']]

zm2018 = pd.read_csv('data/zm_dow.csv')
zm2018.columns = ['dow', 'zm2018']
att = pd.merge(att, zm2018, how='left', left_on = 'dow', right_on = 'dow')
att['zm2018'].fillna(0, inplace=True)

ss2018 = pd.read_csv('data/ss_dow.csv')
ss2018.columns = ['dow', 'ss2018']
att = pd.merge(att, ss2018, how='left', left_on = 'dow', right_on = 'dow')
att['ss2018'].fillna(0, inplace=True)

for ct in ['ramsey', 'crow wing', 'stearns']:
	boatdf = extract_sim_boat(att, county = ct)
	boatdf.to_csv('data/Annual boater net/'+ct+'.csv', index = False)









