'''
MCMC
'''
import numpy as np
from scipy.misc import comb
from itertools import product
import pandas as pd
import pymc as pm
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

nchain = 5

att = pd.read_csv('data/lake_attribute.csv')
att['boat'].fillna(att['boat'].mean(), inplace=True)
att['infest'].fillna(0, inplace=True)
att['inspect'].fillna(0, inplace=True)
att['zm_suit'].fillna(0, inplace=True)
att['ss_suit'].fillna(0, inplace=True)
print(list(att.columns.values))

# creating data vectors 
lake_id = copy.deepcopy(att['id'].values)
infest_zm = copy.deepcopy(att['infest.zm'].values)
infest_ss = copy.deepcopy(att['infest.ss'].values)
infest_both = copy.deepcopy(infest_zm)
zm_suit = copy.deepcopy(att['zm_suit'].values)
ss_suit = copy.deepcopy(att['ss_suit'].values)

del att

# read networks
with open('data/boat_dict.txt') as json_file:  
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


river_net = pd.read_csv('data/river_net_sim.csv')
river_o = river_net['origin'].values
river_d = river_net['destination'].values
river_w = river_net['weight'].values


pars = [0.11033462, 0.00607339, 0.0402376,  0.31955123, 0.21533419, 0.12090607, 0.223175]
pars_name = ["factor_ss", "e_violate_zm", "e_violate_ss", "river_inf_zm", "river_inf_ss", "back_suit_zm", "back_suit_ss"]
pars_lb = [0, 0, 0, 0, 0, 0, 0]
pars_ub = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

scales = {"factor_ss":0.0001, "e_violate_zm":0.00005, "e_violate_ss":0.0001, \
"river_inf_zm": 0.0001, "river_inf_ss": 0.0001, "back_suit_zm": 0.0001, "back_suit_ss": 0.0001}

# scales = np.array([[1.69928101e-02,  9.04736734e-06, -3.09015741e-03,  2.01183966e-03, -3.09767631e-04,  1.70044864e-04,  1.06210898e-03],
#  [ 9.04736734e-06,  1.82939304e-05,  8.17770744e-06, -1.84628889e-04,  8.99596594e-05,  1.92853434e-05, -9.70630321e-05],
#  [-3.09015741e-03,  8.17770744e-06,  1.40449545e-02,  8.13327225e-04, -2.48090609e-03,  2.80038940e-04, -1.48763834e-04],
#  [ 2.01183966e-03, -1.84628889e-04,  8.13327225e-04,  1.88539519e-02, -6.50287718e-03,  1.24723229e-04,  1.76695360e-04],
#  [-3.09767631e-04,  8.99596594e-05, -2.48090609e-03, -6.50287718e-03,  2.04183609e-02,  3.39270490e-03,  1.83810934e-03],
#  [ 1.70044864e-04,  1.92853434e-05,  2.80038940e-04,  1.24723229e-04,  3.39270490e-03,  1.52959471e-02,  1.16373333e-03],
#  [ 1.06210898e-03, -9.70630321e-05, -1.48763834e-04,  1.76695360e-04,  1.83810934e-03,  1.16373333e-03,  2.01743613e-02]])

zm_count = {2012: 53,  2013: 68,  2014: 84,  2015: 99,  2016: 132,  2017: 167, 2018: 202}
ss_count = {2016: 8,  2017: 10, 2018: 13}

target_zm = np.array([val for key, val in zm_count.items() if key > 2012])
target_ss = np.array([val for key, val in ss_count.items() if key > 2016])
target_all = np.append(target_zm, target_ss)

sd_zm = 30
sd_ss = 1
sd_all = np.append(np.repeat(sd_zm, 6) , np.repeat(sd_ss, 2))


outcomes = AdaptMCMC.AdpativeMCMC(iters=50000, burn_in=10000, adapt_par=[1000,1000], \
	pars=pars, pars_name=pars_name, pars_lb=pars_lb, pars_ub=pars_ub, scales=scales, \
	target=target_all, target_sd=sd_all, verbose = 1, function = util.infest_outcome_func, \
	boat_net=boat_net, # small_prob=small_prob, 
	river_o=river_o, river_d=river_d, river_w=river_w, \
	lake_id=lake_id, infest_zm=infest_zm, infest_ss=infest_ss, infest_both=infest_both, \
	zm_suit=zm_suit, ss_suit=ss_suit)


with open('data/mcmc_results_'+str(nchain)+'.txt', 'w') as fout:
    json.dump(outcomes, fout)



import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()
import corner

with open('data/mcmc_results_'+str(nchain)+'.txt') as json_file:  
    outcomes = json.load(json_file)

samples = np.array([outcomes['factor_ss'], outcomes['e_violate_zm'], outcomes['e_violate_ss'], \
	outcomes['river_inf_zm'], outcomes['river_inf_ss'], outcomes['back_suit_zm'], outcomes['back_suit_ss']]).T
# samples = samples[0]
corner.corner(samples[:,:], labels=['factor_ss', 'e_violate_zm', 'e_violate_ss', 'river_inf_zm', 'river_inf_ss', \
	'back_suit_zm', 'back_suit_ss'])
plt.savefig('corner_plot_'+str(nchain)+'.png', format='png', dpi=900)


plt.plot(np.array(outcomes['e_violate_ss']))
plt.show()

plt.acorr(np.array(outcomes['e_violate_zm']), usevlines = True, maxlags = 100)
plt.show()



import numpy as np
from scipy.misc import comb
from itertools import product
import pandas as pd
import pymc as pm
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

nchain = 5

att = pd.read_csv('data/lake_attribute.csv')
att['boat'].fillna(att['boat'].mean(), inplace=True)
att['infest'].fillna(0, inplace=True)
att['inspect'].fillna(0, inplace=True)
att['zm_suit'].fillna(0, inplace=True)
att['ss_suit'].fillna(0, inplace=True)
print(list(att.columns.values))

# creating data vectors 
lake_id = copy.deepcopy(att['id'].values)
infest_zm = copy.deepcopy(att['infest.zm'].values)
infest_ss = copy.deepcopy(att['infest.ss'].values)
infest_both = copy.deepcopy(infest_zm)
zm_suit = copy.deepcopy(att['zm_suit'].values)
ss_suit = copy.deepcopy(att['ss_suit'].values)

del att

# read networks
with open('data/boat_dict1.txt') as json_file:  
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


with open('data/small_prob1.txt') as json_file:  
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

river_net = pd.read_csv('data/river_net_sim.csv')
river_o = river_net['origin'].values
river_d = river_net['destination'].values
river_w = river_net['weight'].values


with open('data/mcmc_results_'+str(nchain)+'.txt') as json_file:  
    outcomes = json.load(json_file)

samples = np.array([outcomes['factor_ss'], outcomes['e_violate_zm'], outcomes['e_violate_ss'], \
	outcomes['river_inf_zm'], outcomes['river_inf_ss'], outcomes['back_suit_zm'], outcomes['back_suit_ss']]).T


l = samples.shape[0]
x = np.random.randint(1, l+1)

sim_target = util.pre_infest_outcome_func(factor_ss = samples[x,0], 
			e_violate_zm = samples[x,1], e_violate_ss = samples[x,2], 
			river_inf_zm = samples[x,3], river_inf_ss = samples[x,4], 
			back_suit_zm = samples[x,5], back_suit_ss = samples[x,6], 
			boat_net=boat_net, small_prob=small_prob, river_o=river_o, river_d=river_d, river_w=river_w, 
			lake_id=lake_id, infest_zm=infest_zm, infest_ss=infest_ss, 
			infest_both=infest_both, zm_suit=zm_suit, ss_suit=ss_suit)
print(sim_target[0])

print(np.mean(samples, axis = 0))
print(np.median(samples, axis = 0))

print(np.cov(samples[20000:, ].T))







