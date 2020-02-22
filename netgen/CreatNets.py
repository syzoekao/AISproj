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

move = move[['dow_origin', 'dow_destination', 'id_origin', 'id_destination', 'normBoats', 'dumBoat', 'inBoat']]

att = pd.read_csv('data/lake_attribute.csv')
inspID = att.loc[att['inspect'] == 1, 'dow'].to_numpy()

tmpMove = move.loc[move['inBoat'] == 1]
moveMat = np.zeros((att.shape[0], att.shape[0]))
for ix in range(tmpMove.shape[0]): 
	ix_o = tmpMove.iloc[ix]['id_origin'].astype(int)
	ix_d = tmpMove.iloc[ix]['id_destination'].astype(int)
	ix_w = tmpMove.iloc[ix]['normBoats']
	moveMat[ix_o, ix_d] = ix_w

MoveDum = 1 * (moveMat > 0)

del tmpMove


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
sum_boat = np.zeros(20)

n_true_pos = np.zeros((inspID.shape[0], 20))
n_false_neg = np.zeros((inspID.shape[0], 20))
n_true_neg = np.zeros((inspID.shape[0], 20))
n_false_pos = np.zeros((inspID.shape[0], 20))
true_pos_rate = np.zeros((inspID.shape[0], 20))
true_neg_rate = np.zeros((inspID.shape[0], 20))
mean_boat_lake = np.zeros((inspID.shape[0], 20))


for run in range(0, 20): 
	aa = timeit.default_timer()
	temp_samp = np.random.binomial(1, p = predprob)
	temp_dum = (predprob >= 0.5) + (predprob < 0.5) * temp_samp
	np.fill_diagonal(temp_dum, 1)
	mat_w = np.random.poisson(predboat)
	mat_w[mat_w == 0] = 1
	out_net = mat_w * temp_dum
	sum_w_all = np.sum(out_net, axis = 0)
	# sum_w_sub = np.sum(out_net * MoveDum, axis = 0)
	# adj_weight =  (sum_w_all / sum_w_sub)
	# adj_weight[np.isinf(adj_weight)] = 1
	# adj_weight[adj_weight > 10] = 10
	adj_weight = 1

	ann_boat = att[['annualTraffic2']].to_numpy().T[0]
	ann_boat = ann_boat * adj_weight

	# boat_mat = np.round(mat_w / sum_w_all * ann_boat)
	boat_mat = np.ceil((out_net / sum_w_all) * ann_boat)

	ix = np.where(boat_mat >= 1)
	boat_o = ix[0]
	boat_d = ix[1]
	boat_w = boat_mat[ix]
	sum_boat[run] = np.sum(boat_w)

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

	# sum_boat[run] = temp_move.loc[temp_move['dow_origin'] != temp_move['dow_destination']]['weight'].sum()

	tmp_move = pd.merge(move, temp_move, how = 'left', on = ['dow_origin', 'dow_destination'])
	tmp_move['weight'].fillna(0, inplace=True)
	tmp_move[['sim_conn']] = 1 * (tmp_move[['weight']] > 0)

	tmp = tmp_move.loc[(tmp_move['dumBoat'] == 1), 'weight'].describe()
	mean_boat[run] = tmp.ix['mean']
	max_boat[run] = tmp.ix['max']
	std_boat[run] = tmp.ix['std']

	tmp = tmp_move.loc[(tmp_move['sim_conn'] == 1), 'weight'].describe()
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

		tempSet = temp_move.loc[(temp_move['dow_origin'] == j) | (temp_move['dow_destination'] == j)] 
		mean_boat_lake[i, run] = tempSet[['weight']].mean()

	print(timeit.default_timer()-aa)


sim_net_data = pd.DataFrame({"mean_boat": mean_boat, 'max_boat': max_boat, 'std_boat': std_boat, 
	"mean_boat_sim": mean_boat_sim, 'max_boat_sim': max_boat_sim, 'std_boat_sim': std_boat_sim, 
	'true_pos': true_pos_vec, 'true_neg': true_neg_vec, 'sum_boat': sum_boat})
sim_net_data.to_csv('data/Annual boater net/sim_net_summary.csv')

true_pos_rate_mean = np.mean(true_pos_rate, axis = 1)
true_pos_rate_std = np.std(true_pos_rate, axis = 1)
true_neg_rate_mean = np.mean(true_neg_rate, axis = 1)
true_neg_rate_std = np.std(true_neg_rate, axis = 1)
mean_boat_lake_mean = np.mean(mean_boat_lake, axis = 1)

insp_out = np.array([inspID, true_pos_rate_mean, true_pos_rate_std, true_neg_rate_mean, true_neg_rate_std, 
	mean_boat_lake_mean]).T
insp_out = pd.DataFrame(insp_out)
insp_out.columns = ['dow', 'true_pos_rate_mean', 'true_pos_rate_std', 'true_neg_rate_mean', 
'true_neg_rate_std', 'mean_boat']

pred_by_lakes = pd.merge(att[['dow', 'lake_name', 'inspect', 'infest', 'infest_zm', 'infest_ss']], insp_out, how = 'left', on = 'dow')
pred_by_lakes.to_csv('data/Annual boater net/prediction by lakes.csv', index = False)


# import matplotlib as mpl
# import matplotlib.pyplot as plt

# plt.clf()
# fig = plt.figure(figsize=(12,6))
# plt.hist(adj_weight, color = 'cornflowerblue', bins = 40)
# fig.tight_layout()
# plt.savefig('data/Annual boater net/weight_histogram.eps', format='eps', dpi=1000)



'''
creating average network for calibration (weekly)
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

att = pd.read_csv('data/lake_attribute.csv')
boat_csv = pd.read_csv('data/Annual boater net/boats1.csv')
boat_csv[['weight1']] = boat_csv[['weight']]
boat_csv.drop(["weight"], axis = 1, inplace = True)

for j in range(2, 21): 
	tmp_csv = pd.read_csv('data/Annual boater net/boats' + str(j) + '.csv')
	tmp_csv[['weight' + str(j)]] = tmp_csv[['weight']]
	tmp_csv.drop(["weight"], axis = 1, inplace = True)
	boat_csv = pd.merge(boat_csv, tmp_csv, how = 'outer', on = ['dow_origin', 'dow_destination'])

boat_csv.fillna(0, inplace = True)
boat_csv['weight'] = boat_csv[['weight' + str(x) for x in range(1, 21)]].mean(axis=1)
boat_csv = boat_csv[boat_csv.weight >= 0.4]
boat_csv['weight'] = boat_csv['weight'] / 26

boat_csv = boat_csv[['dow_origin', 'dow_destination', 'weight']]

boat_csv = pd.merge(boat_csv, att[['dow', 'id']], how = 'left', left_on = 'dow_origin', right_on = 'dow')
boat_csv.drop(["dow", "dow_origin"], axis = 1, inplace = True)
boat_csv.rename({'id': 'id_origin'}, axis = 1, inplace = True)
boat_csv = pd.merge(boat_csv, att[['dow', 'id']], how = 'left', left_on = 'dow_destination', right_on = 'dow')
boat_csv.drop(["dow", "dow_destination"], axis = 1, inplace = True)
boat_csv.rename({'id': 'id_destination'}, axis = 1, inplace = True)

boat_dict = dict()
# make boater net dictionary
for ix_o in range(att.shape[0]): 
	# ix_o = 0
	ix_d = boat_csv.ix[boat_csv['id_origin'] == ix_o][['id_destination']].to_numpy().T[0]
	weight = boat_csv.ix[boat_csv['id_origin'] == ix_o][['weight']].to_numpy().T[0]
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

for j in range(1, 21): 
	print(j)
	aa = timeit.default_timer()
	boat_csv = pd.read_csv('data/Annual boater net/boats' + str(j) + '.csv')
	boat_csv = pd.merge(boat_csv, att[['dow', 'id']], how = 'left', left_on = 'dow_origin', right_on = 'dow')
	boat_csv.drop(["dow", "dow_origin"], axis = 1, inplace = True)
	boat_csv.rename({'id': 'id_origin'}, axis = 1, inplace = True)
	boat_csv = pd.merge(boat_csv, att[['dow', 'id']], how = 'left', left_on = 'dow_destination', right_on = 'dow')
	boat_csv.drop(["dow", "dow_destination"], axis = 1, inplace = True)
	boat_csv.rename({'id': 'id_destination'}, axis = 1, inplace = True)
	boat_csv[['weight']] = boat_csv[['weight']] / 26

	boat_dict = dict()
	# make boater net dictionary
	for ix_o in range(att.shape[0]): 
		# ix_o = 0
		ix_d = boat_csv.ix[boat_csv['id_origin'] == ix_o][['id_destination']].to_numpy().T[0]
		weight = boat_csv.ix[boat_csv['id_origin'] == ix_o][['weight']].to_numpy().T[0]
		ix_d = [str(x) for x in ix_d]
		if len(ix_d) > 0: 
			temp = {str(ix_o): dict(zip(ix_d, weight))}
			boat_dict.update(temp)

	with open('data/boat_dict'+str(j)+'.txt', 'w') as fout:
	    json.dump(boat_dict, fout)

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

dows_before_rm = np.unique(np.concatenate((river[['dow.origin']].to_numpy().T[0], 
	river[['dow.destination']].to_numpy().T[0])))

river.dropna(how = 'any', inplace = True)

dows_after_rm = np.unique(np.concatenate((river[['dow.origin']].to_numpy().T[0], 
	river[['dow.destination']].to_numpy().T[0])))

print(dows_before_rm.shape[0])
print(dows_after_rm.shape[0])

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
import ast
import timeit
import os
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)


def extract_sim_boat(att, county, path = 'data/Annual boater net'): 
	# county = 'stearns'
	# path = 'data/Annual boater net2'
	if county == 'ramsey': 
		att.loc[att['dow'] == 82016700, ['county', 'county_name']] = [62, county]
	if county == 'crow wing': 
		att.loc[att['dow'] == 11030500, ['county', 'county_name']] = [18, county]
		att.loc[att['dow'] == 48000200, ['county', 'county_name']] = [18, county]
	if county == 'stearns': 
		att.loc[att['dow'] == 86025200, ['county', 'county_name']] = [73, county]
		att.loc[att['dow'] == 73020000, ['county', 'county_name']] = [47, "meeker"]
	if county == 'meeker': 
		att.loc[att['dow'] == 73020000, ['county', 'county_name']] = [47, county]

	nLake = att.loc[att['county_name'] == county, 'dow'].shape[0] + 8
	added_set = pd.Series(['none', 'zm', 'ss', 'ew', 'zm_ss', 'zm_ew', 'ss_ew', 'zm_ss_ew'])
	lake_set = pd.concat([att.loc[att['county_name']==county, 'dow'], added_set]) 
	in_county_lakes = att.loc[att['county_name']==county, 'dow']

	boatdf = pd.DataFrame({'dow_origin': np.repeat(lake_set, nLake), 
		'dow_destination': np.tile(lake_set, nLake)})

	# Get attributes of the origin lakes
	attribute_set = ['dow', 'lake_name', 'county_name', 'infest_cate']
	boatdf = pd.merge(boatdf, att[attribute_set], how = 'left', left_on=['dow_origin'], right_on=['dow'])
	boatdf.drop(columns=['dow'], inplace = True)
	boatdf.columns = ['dow_origin', 'dow_destination', 'lake_origin', 'county_origin', 'infest_origin']

	# Get attributes of the destination lakes
	boatdf = pd.merge(boatdf, att[attribute_set], how = 'left', left_on=['dow_destination'], right_on=['dow'])
	boatdf.drop(columns=['dow'], inplace = True)
	boatdf.columns = ['dow_origin', 'dow_destination', 'lake_origin', 'county_origin',
		'infest_origin', 'lake_destination', 'county_destination', 
		'infest_destination']

	boatdf.loc[boatdf['dow_origin'].isin(['none', 'zm', 'ss', 'ew', 'zm_ss', 'zm_ew', 'ss_ew', 'zm_ss_ew']), 
	'county_origin'] = 'not '+ county
	boatdf.loc[boatdf['dow_destination'].isin(['none', 'zm', 'ss', 'ew', 'zm_ss', 'zm_ew', 'ss_ew', 'zm_ss_ew']), 
	'county_destination'] = 'not '+county
	boatdf = boatdf.loc[~((boatdf['county_origin'] == 'not ' + county) & (boatdf['county_destination'] == 'not ' + county))]
	
	for i in range(1, 21): 
		print(i)
		tmp_boat = pd.read_csv(path + '/boats'+str(i) +'.csv')
		tmp_boat = tmp_boat.loc[(tmp_boat['dow_origin'].isin(in_county_lakes)) | (tmp_boat['dow_destination'].isin(in_county_lakes))]
		tmp_boat.columns = ['dow_origin', 'dow_destination', 'weight']
		tmp_boat = pd.merge(tmp_boat, att[['dow', 'county_name', 'infest_cate']], how = 'left', left_on=['dow_origin'], right_on=['dow'])
		tmp_boat.drop(columns=['dow'], inplace = True)
		tmp_boat.columns = ['dow_origin', 'dow_destination', 'weight', 'county_origin', 'infest_origin']
		tmp_boat = pd.merge(tmp_boat, att[['dow', 'county_name', 'infest_cate']], how = 'left', left_on=['dow_destination'], right_on=['dow'])
		tmp_boat.drop(columns=['dow'], inplace = True)
		tmp_boat.columns = ['dow_origin', 'dow_destination', 'weight', 'county_origin', 'infest_origin', 
			'county_destination', 'infest_destination']

		# for movement that are from / to other counties by the type of species
		tmp_boat.loc[tmp_boat['county_origin'] != county, 'county_origin'] = 'not ' + county
		tmp_boat.loc[tmp_boat['county_destination'] != county, 'county_destination'] = 'not ' + county
		
		tmp_boat.loc[(tmp_boat['county_origin'] == 'not ' + county) & (tmp_boat['infest_origin'].isin(added_set)), 'dow_origin'] = \
			tmp_boat.loc[(tmp_boat['county_origin'] == 'not ' + county) & (tmp_boat['infest_origin'].isin(added_set)), 'infest_origin']

		tmp_boat.loc[(tmp_boat['county_destination'] == 'not ' + county) & (tmp_boat['infest_destination'].isin(added_set)), 'dow_destination'] = \
			tmp_boat.loc[(tmp_boat['county_destination'] == 'not ' + county) & (tmp_boat['infest_destination'].isin(added_set)), 'infest_destination']

		sum_df = tmp_boat.groupby(['dow_origin','dow_destination'])['weight'].agg('sum').reset_index()
		boatdf = pd.merge(boatdf, sum_df, how = 'left', left_on = ['dow_origin', 'dow_destination'], right_on = ['dow_origin', 'dow_destination'])
		boatdf['weight'].fillna(0, inplace = True)
		boatdf = boatdf.rename(columns = {'weight': 'weight'+str(i)})
		boatdf['weight'+str(i)].fillna(0, inplace = True)

	boatdf.loc[boatdf['infest_origin'].isnull(), 'infest_origin'] = boatdf.loc[boatdf['infest_origin'].isnull(), 'dow_origin']
	boatdf.loc[boatdf['infest_destination'].isnull(), 'infest_destination'] = boatdf.loc[boatdf['infest_destination'].isnull(), 'dow_destination']
	return boatdf


att = pd.read_csv('data/lake_attribute.csv')
att = att[['dow', 'lake_name', 'acre', 'utm_x', 'utm_y', 'county', 'county_name', 'inspect', 'id']]

infestW2019 = open("data/infestedwaterDOW.txt", "r").readlines()[0]
infestW2019 = ast.literal_eval(infestW2019)

att['zm2019'] = att['dow'].isin(infestW2019['zm']) * 1
att['ss2019'] = att['dow'].isin(infestW2019['ss']) * 1
att['ew2019'] = att['dow'].isin(infestW2019['ew']) * 1
att['infest_cate'] = 'none'
att.loc[(att['zm2019'] == 1) & (att['ss2019'] == 0) & (att['ew2019'] == 0), 'infest_cate'] = 'zm'
att.loc[(att['zm2019'] == 0) & (att['ss2019'] == 1) & (att['ew2019'] == 0), 'infest_cate'] = 'ss'
att.loc[(att['zm2019'] == 0) & (att['ss2019'] == 0) & (att['ew2019'] == 1), 'infest_cate'] = 'ew'
att.loc[(att['zm2019'] == 1) & (att['ss2019'] == 1) & (att['ew2019'] == 0), 'infest_cate'] = 'zm_ss'
att.loc[(att['zm2019'] == 1) & (att['ss2019'] == 0) & (att['ew2019'] == 1), 'infest_cate'] = 'zm_ew'
att.loc[(att['zm2019'] == 0) & (att['ss2019'] == 1) & (att['ew2019'] == 1), 'infest_cate'] = 'ss_ew'
att.loc[(att['zm2019'] == 1) & (att['ss2019'] == 1) & (att['ew2019'] == 1), 'infest_cate'] = 'zm_ss_ew'
att.drop(['zm2019', 'ss2019', 'ew2019'], axis = 1, inplace = True)

# att.to_csv('data/Annual boater net2/lake_data.csv', index = False)

for ct in ['ramsey', 'crow wing', 'stearns', 'meeker']:
	boatdf = extract_sim_boat(att, county = ct)
	boatdf.to_csv('data/Annual boater net/' + ct + '.csv', index = False)







