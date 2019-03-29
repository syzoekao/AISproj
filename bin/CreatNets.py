import numpy as np
from scipy.misc import comb
from itertools import product
import pymc
import pandas as pd
import json
import timeit
import networkx as nx
import statsmodels
import os
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)

att = pd.read_csv('data/lake_attribute.csv')

y_col = ['dumBoat']


# Use coefficients are from R estimation
'''
coefs0 = { "constant": -9.43484224839827,  "log_distance": -1.24645990430623,  "log_acre_diff": -0.296350432150528,  
"same_county": 0.811503250847373,  "log_acre_o": 0.825486427259707,  "log_acre_d": 0.740293237438696,  
"log_boat_diff": -0.188837142631315,  "log_boat_o": 0.477240122901197,  "log_boat_d": 0.652646629657891,  
"infest_o": 0.432479925146412,  "infest_d": 0.583942696503004,  "same_infest": 0.141583356512359,  
"insp_d": 0.793705760443316,  "same_insp": 2.66145760606513,  "acre_diff_x_county": -0.076211540237906 }
'''

coefs5 = {"constant": -13.8142965660071,  "log_distance": -1.11736392426862,  "dist50": 1.12709657943131,  
"log_acre_diff": -0.283985394538909,  "log_acre_o": 0.980580267800405,  "log_acre_d": 0.893535512957915,  
"log_boat_diff": 0.498437785167016,  "log_boat_o": 0.918835143069213,  "log_boat_d": 1.23793527855556,  
"infest_o": 0.605462278818776,  "infest_d": 0.665941992280917,  "same_infest": 0.231411869775555,  
"insp_d": 0.898388438450318,  "same_insp": 1.80213836066291,  "log_ramp_acre_o": 0.841530622654791,  
"log_ramp_acre_d": 0.609970629026273,  "log_distance_x_dist50": -0.151654826904446}

coefs6 = {"constant": -13.8722663518705,  "log_distance": -1.09939163321157,  "dist50": 0.640370654390918,  
"log_acre_diff": -0.286750498320829,  "log_acre_o": 0.978458943608987,  "log_acre_d": 0.891434082574503,  
"log_boat_diff": 0.528579726160527,  "log_boat_o": 0.903314439868417,  "log_boat_d": 1.22582293036695,  
"infest_o": 0.621541898318226,  "infest_d": 0.682312476915083,  "same_infest": 0.228015198669238,  
"insp_d": 0.897133845920746,  "same_insp": 1.80280694339769,  "log_ramp_acre_o": 0.844159614862215,  
"log_ramp_acre_d": 0.61466377375799,  "same_county": 0.367068946540615,  
"log_distance_x_dist50": -0.0460109168138217}


x_col5 = [k for k,v in coefs5.items()]
x_col5 = x_col5[1:]

x_col6 = [k for k,v in coefs6.items()]
x_col6 = x_col6[1:]


att = pd.read_csv('data/lake_attribute.csv')
att['boat'].fillna(att['boat'].mean(), inplace=True)
att['infest'].fillna(0, inplace=True)
att['inspect'].fillna(0, inplace=True)
print(list(att.columns.values))
inspID = att.loc[att['inspect']==1, 'id'].values.tolist()

logit = np.empty((att.shape[0], att.shape[0]))

ids = att['id'].tolist()

model = 'model5'

for ix in ids: 
	print(ix)
	tempDF = pd.DataFrame([], columns = x_col6)
	tempDF['infest_o'] = np.repeat(att['infest'].iloc[ix], att.shape[0])
	tempDF['infest_d'] = att['infest']
	tempDF['same_county'] = 1*(att['county'] == att['county'].iloc[ix])
	utm_x = att['utm_x'].iloc[ix]
	utm_y = att['utm_y'].iloc[ix]
	tmp = np.sqrt((att['utm_x'] - utm_x)**2 + (att['utm_y'] - utm_y)**2)/1000
	tempDF['log_distance'] = np.log(tmp + 1)
	tempDF['dist50'] = 1*(tmp<50)
	tempDF['log_distance_x_dist50'] = tempDF['log_distance']*tempDF['dist50']
	tmp = att['acre'].iloc[ix]
	tempDF['log_acre_diff'] = np.log(np.absolute(att['acre'] - tmp) + 1)
	tempDF['log_acre_o'] = np.log(tmp + 1)
	tempDF['log_acre_d'] = np.log(att['acre'] + 1)
	tmp = att['boat'].iloc[ix]
	tempDF['log_boat_o'] = np.log(tmp + 1)
	tempDF['log_boat_d'] = np.log(att['acre'] + 1)
	tempDF['log_boat_diff'] = np.log(np.absolute(att['boat'] - tmp) + 1)
	tmp = att['infest'].iloc[ix]
	tempDF['same_infest'] = 1*(att['infest'] == tmp)

	tempDF['insp_d'] = att['inspect']
	tmp = att['inspect'].iloc[ix]
	tempDF['same_insp'] = 1*((att['inspect'] == tmp) & (tmp == 1))

	tmp = att['acre'].iloc[ix]
	tmp1 = att['ramp'].iloc[ix]
	tempDF['log_ramp_acre_o'] = np.log(tmp1/tmp + 1)
	tempDF['log_ramp_acre_d'] = np.log(att['ramp']/att['acre'] + 1)

	# tempDF['intercept'] = 1

	if model == 'model5':
		coefs = coefs5
		lp = coefs['constant']* np.ones(tempDF.shape[0]) + coefs['log_distance']*tempDF['log_distance'] + \
		coefs['dist50']*tempDF['dist50'] + coefs['log_acre_diff']*tempDF['log_acre_diff'] + \
		coefs['log_acre_o']*tempDF['log_acre_o'] + coefs['log_acre_d']*tempDF['log_acre_d'] + \
		coefs['log_boat_diff']*tempDF['log_boat_diff'] + coefs['log_boat_o']*tempDF['log_boat_o'] + \
		coefs['log_boat_d']*tempDF['log_boat_d'] + coefs['infest_o']*tempDF['infest_o'] + \
		coefs['infest_d']*tempDF['infest_d'] + coefs['same_infest']*tempDF['same_infest'] + \
		coefs['insp_d']*tempDF['insp_d'] + coefs['same_insp']*tempDF['same_insp'] + \
		coefs['log_ramp_acre_o']*tempDF['log_ramp_acre_o'] + coefs['log_ramp_acre_d']*tempDF['log_ramp_acre_d'] + \
		coefs['log_distance_x_dist50']*tempDF['log_distance_x_dist50'] 
	else: 
		coefs = coefs6
		lp = coefs['constant']* np.ones(tempDF.shape[0]) + coefs['log_distance']*tempDF['log_distance'] + \
		coefs['dist50']*tempDF['dist50'] + coefs['log_acre_diff']*tempDF['log_acre_diff'] + \
		coefs['log_acre_o']*tempDF['log_acre_o'] + coefs['log_acre_d']*tempDF['log_acre_d'] + \
		coefs['log_boat_diff']*tempDF['log_boat_diff'] + coefs['log_boat_o']*tempDF['log_boat_o'] + \
		coefs['log_boat_d']*tempDF['log_boat_d'] + coefs['infest_o']*tempDF['infest_o'] + \
		coefs['infest_d']*tempDF['infest_d'] + coefs['same_infest']*tempDF['same_infest'] + \
		coefs['insp_d']*tempDF['insp_d'] + coefs['same_insp']*tempDF['same_insp'] + \
		coefs['log_ramp_acre_o']*tempDF['log_ramp_acre_o'] + coefs['log_ramp_acre_d']*tempDF['log_ramp_acre_d'] + \
		coefs['same_county']*tempDF['same_county'] + \
		coefs['log_distance_x_dist50']*tempDF['log_distance_x_dist50'] 

	lp = np.array(lp)
	Y_prob = 1/(1+np.exp(-lp))
	logit[ix] = Y_prob




inspID = att.loc[att['inspect'] == 1, 'id'].tolist()
print(np.sum(logit[inspID]))

temp = 1*(logit > 0.0001) # remove values < 0.0001
logit2 = temp*logit
np.fill_diagonal(logit2, 1.0)


aa = timeit.default_timer()
np.save("data/logit", logit2)
print(timeit.default_timer() - aa)


del inspID, ids, logit, logit2


'''
gamma regression and splines
'''
import copy
import numpy as np
from scipy.misc import comb
from itertools import product
import pymc
import pandas as pd
import json
import timeit
import networkx as nx
import statsmodels
import os
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)

aa = timeit.default_timer()
move = pd.read_csv('data/movement_for_stat.csv')
print(timeit.default_timer() - aa)

move = move.drop(['Unnamed: 0'], axis = 1)
print(move.shape)

# move = pd.merge(move, counts, how = "left", left_on = ["dow.origin", "dow.destination"], right_on = ["dow.origin", "dow.destination"])
move = move.loc[move['dumBoat'] == 1]
move['counts'] = copy.deepcopy(move['avg.count'])
move['log_counts'] = np.log(move['counts'] + 1)


att = pd.read_csv('data/lake_attribute.csv')

move.columns = ['dow_destination', 'dow_origin', 'county_origin', 'county_destination', 'avg_count', 
'normBoats', 'normBoatsExtra', 'boat_o', 'boat_d', 'acre_o', 'acre_d', 'lat_o', 'long_o', 'lat_d', 
'long_d', 'distance', 'infest_o', 'infest_d', 'insp_o', 'insp_d', 'access_o', 'access_d', 'ramp_acre_o', 
'ramp_acre_d', 'same_county', 'dumBoat', 'acre_diff', 'boat_diff', 'self', 'log_distance', 
'log_acre_diff', 'log_acre_o', 'log_acre_d', 'log_normBoats', 'log_boat_o', 'log_boat_d', 
'log_boat_diff', 'same_infest', 'same_insp', 'same_access', 'dist50', 'dist100', 'log_ramp_acre_o', 
'log_ramp_acre_d', 'counts', 'log_counts']

'''
for ct in np.unique(att['county'].values).tolist(): 
	print(ct)
	move['ori_cty'+str(ct)] = 1.0*(move['county_origin'] == ct)
	move['des_cty'+str(ct)] = 1.0*(move['county_destination'] == ct)
'''

from patsy import dmatrix
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf


move['log_distance2'] = move['log_distance']**2
move['log_distance3'] = move['log_distance']**3

move['log_acre_diff2'] = move['log_acre_diff']**2
move['log_acre_diff3'] = move['log_acre_diff']**3

move['log_acre_o2'] = move['log_acre_o']**2

move['log_acre_d2'] = move['log_acre_d']**2
move['log_acre_d3'] = move['log_acre_d']**3

move['log_boat_diff2'] = move['log_boat_diff']**2
move['log_boat_diff3'] = move['log_boat_diff']**3

move['log_boat_o2'] = move['log_boat_o']**2
move['log_boat_o3'] = move['log_boat_o']**3

move['log_boat_d2'] = move['log_boat_d']**2
move['log_boat_d3'] = move['log_boat_d']**3


move['log_acre_diff_x_log_boat_diff'] = move['log_acre_diff']*move['log_boat_diff']
move['log_acre_o_x_log_boat_o'] = move['log_acre_o']*move['log_boat_o']
move['log_acre_d_x_log_boat_d'] = move['log_acre_d']*move['log_boat_d']
move['log_distance_x_dist50'] = move['log_distance']*move['dist50']


y_col = ['log_counts']

'''
x_col1 = ['log_distance', 'dist50', 'log_acre_diff', 'log_acre_o', 'log_acre_d', 'log_boat_diff', \
'log_boat_o', 'log_boat_d', 'infest_o', 'infest_d', 'same_infest', 'insp_o', 'insp_d', 'same_insp', \
'log_ramp_acre_o', 'log_ramp_acre_d', 'log_distance_x_dist50']
'''

x_col1 = ['log_distance', 'log_distance2', 'log_distance3', \
'log_acre_diff', 'log_acre_diff2', 'log_acre_diff3', 'log_acre_o', 'log_acre_o2', \
'log_acre_d', 'log_acre_d2', 'log_boat_diff', 'log_boat_diff2', 'log_boat_diff3', \
'log_boat_o', 'log_boat_o2', 'log_boat_o3', 'log_boat_d', 'log_boat_d2', 'log_boat_d3', \
'infest_o', 'infest_d', 'same_infest', 'insp_o', 'insp_d', 'same_insp', \
'log_ramp_acre_o', 'log_ramp_acre_d']

move_exog = move[x_col1]
move_exog = sm.add_constant(move_exog)
move_endog = move[y_col]
gamma_model1 = sm.GLM(move_endog, move_exog, family=sm.families.Gamma(link=sm.families.links.log))
gamma_results1 = gamma_model1.fit()

print(gamma_results1.summary())
print(gamma_results1.aic)
print(gamma_results1.bic)

move_exog = move[x_col1]
move_exog = sm.add_constant(move_exog)
Y_pred = np.exp(gamma_results1.predict(move_exog))

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(move['counts'], Y_pred))
print(rms)
print(np.corrcoef(Y_pred, move['counts'])[0, 1])

coef_table = gamma_results1.params
coef_table = pd.concat([coef_table, gamma_results1.conf_int()], axis=1)
coef_table = pd.concat([coef_table, gamma_results1.pvalues], axis=1)
coef_table.columns = ['coef', 'std err', 'z', 'P>|z|']
coef_table.to_csv("data/gamma_coefs1.csv")

move_dow = move.groupby(['dow_origin']).agg({'counts': 'sum'})
move1 = move[['dow_origin', 'dow_destination', 'counts']]
move1['Y_pred'] = Y_pred.tolist()
move1 = pd.merge(move1, move_dow, how = 'left', left_on = 'dow_origin', right_on = 'dow_origin')
move1.columns = ['dow_origin', 'dow_destination', 'counts', 'Y_pred', 'all_counts']
move1['pct_counts'] = move1['counts']/move1['all_counts']

pred_dow = move1.groupby(['dow_origin']).agg({'Y_pred': 'sum'})
move1 = pd.merge(move1, pred_dow, how = 'left', left_on = 'dow_origin', right_on = 'dow_origin')
move1.columns = ['dow_origin', 'dow_destination', 'counts', 'Y_pred', 'all_counts', 'pct_counts', 'all_pred']
move1['pct_pred'] = move1['Y_pred']/move1['all_pred']

rms1 = sqrt(mean_squared_error(move1['pct_counts'], move1['pct_pred']))
print(rms1)
print(np.corrcoef(move1['pct_pred'], move1['pct_counts'])[0, 1])

import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()

plt.scatter(move1['pct_pred'], move1['pct_counts'], facecolor='dodgerblue')
plt.show()


logit = np.load("data/logit.npy")

gamma_pred = np.zeros(logit.shape)

x_col1 = ['const']+x_col1

for ix in range(logit.shape[0]): 
	# aaa = timeit.default_timer()
	# aa = timeit.default_timer()
	print(ix)
	tempDF = pd.DataFrame([], columns = x_col1)
	tempDF['const'] = np.ones(att.shape[0])
	tempDF['infest_o'] = np.repeat(att['infest'].iloc[ix], att.shape[0])
	tempDF['infest_d'] = att['infest']
	tmp = att['county'].iloc[ix]
	utm_x = att['utm_x'].iloc[ix]
	utm_y = att['utm_y'].iloc[ix]
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====1")
	tmp = np.sqrt((att['utm_x'] - utm_x)**2 + (att['utm_y'] - utm_y)**2)/1000
	tempDF['log_distance'] = np.log(tmp + 1)
	tempDF['dist50'] = 1*(tmp<50)
	tempDF['log_distance_x_dist50'] = tempDF['log_distance']*tempDF['dist50']
	tempDF['log_distance2'] = tempDF['log_distance']**2
	tempDF['log_distance3'] = tempDF['log_distance']**3
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====2")
	tmp = att['acre'].iloc[ix]
	tempDF['log_acre_diff'] = np.log(np.absolute(att['acre'] - tmp) + 1)
	tempDF['log_acre_diff2'] = tempDF['log_acre_diff']**2
	tempDF['log_acre_diff3'] = tempDF['log_acre_diff']**3
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====3")
	tempDF['log_acre_o'] = np.log(tmp + 1)
	tempDF['log_acre_o2'] = tempDF['log_acre_o']**2 
	tempDF['log_acre_d'] = np.log(att['acre'] + 1)
	tempDF['log_acre_d2'] = tempDF['log_acre_d']**2
	tempDF['log_acre_d3'] = tempDF['log_acre_d']**3
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====4")
	tmp = att['boat'].iloc[ix]
	tempDF['log_boat_diff'] = np.log(np.absolute(att['boat'] - tmp) + 1)
	tempDF['log_boat_diff2'] = tempDF['log_boat_diff']**2
	tempDF['log_boat_diff3'] = tempDF['log_boat_diff']**3
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====5")
	tempDF['log_boat_o'] = np.log(tmp + 1)
	tempDF['log_boat_o2'] = tempDF['log_boat_o']**2
	tempDF['log_boat_o3'] = tempDF['log_boat_o']**3
	tempDF['log_boat_d'] = np.log(att['boat'] + 1)
	tempDF['log_boat_d2'] = tempDF['log_boat_d']**2
	tempDF['log_boat_d3'] = tempDF['log_boat_d']**3
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====6")
	tmp = att['infest'].iloc[ix]
	tempDF['same_infest'] = 1*(att['infest'] == tmp)
	tmp = att['inspect'].iloc[ix]
	tempDF['insp_o'] = tmp
	tempDF['insp_d'] = att['inspect']
	tempDF['same_insp'] = 1*((att['inspect'] == tmp) & (tmp == 1))
	# print(timeit.default_timer() - aa)

	tmp1 = att['ramp'].iloc[ix]
	tmp2 = att['acre'].iloc[ix]
	tempDF['log_ramp_acre_o'] = np.log(tmp1/tmp2+1)
	tempDF['log_ramp_acre_d'] = np.log(att['ramp']/att['acre']+1)

	# aa = timeit.default_timer()
	# print("====8")
	tempDF = tempDF[x_col1]
	# print(timeit.default_timer() - aa)

	aa = timeit.default_timer()
	# print("====9")
	Y_pred = np.exp(gamma_results1.predict(tempDF))
	# Y_pred[Y_pred>850] = 850
	# Y_pred = Y_pred*(1*(logit[ix] > 0.0001))
	gamma_pred[ix] = Y_pred
	# gamma_pred[np.where(np.isnan(gamma_pred))] = 0
	# print(timeit.default_timer() - aa)
	# print(timeit.default_timer() - aaa)


np.fill_diagonal(gamma_pred, 0)
gamma_pred[gamma_pred>850] = 850
gamma_pred = gamma_pred*(1*(logit > 0.0001))
gamma_pred[np.where(np.isnan(gamma_pred))] = 0


del move, tempDF, Y_pred

# logit = np.load("data/logit.npy")
# gamma_pred = np.load("data/gamma_pred.npy")

avg_weight = logit*gamma_pred
aa = timeit.default_timer()
np.save("data/avg_weight", avg_weight)
print(timeit.default_timer() - aa)

aa = timeit.default_timer()
np.save("data/gamma_pred", gamma_pred)
print(timeit.default_timer() - aa)


'''
Prediction of proportion of self-loop
'''
import numpy as np
from scipy.misc import comb
from itertools import product
import pymc
import pandas as pd
import json
import timeit
import networkx as nx
import statsmodels
import os
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)

att = pd.read_csv('data/lake_attribute.csv')

# att['mile5'] = mile5
att = att[['dow', 'lake_name', 'acre', 'utm_x', 'utm_y', 'county', 'county.name', 'infest', 'inspect', \
'infest.zm', 'infest.ss', 'zm_suit', 'ss_suit', 'accessible', 'id', 'ramp']]

att.columns = ['dow', 'lake_name', 'acre', 'utm_x', 'utm_y', 'county', 'county_name', 'infest', 
'inspect', 'infest_zm', 'infest_ss', 'zm_suit', 'ss_suit', 'accessible', 'id', 'ramp']

dist50 = np.zeros(att.shape[0])
for ix in range(att.shape[0]):
	print(ix)
	utm_x = att.loc[att['id'] == ix, 'utm_x'].values	
	utm_y = att.loc[att['id'] == ix, 'utm_y'].values		
	tmp = np.sqrt((att['utm_x'] - utm_x)**2 + (att['utm_y'] - utm_y)**2)/1000
	dist50[ix] = np.sum(tmp<50)

att['dist50'] = dist50
att['ramp_per_acre'] = att['ramp']/att['acre']
att.to_csv('/Users/szu-yukao/Documents/FishProject/Writing/some_lake_attributes.csv', index = False)


inspID = att.loc[att['inspect'] == 1, 'dow'].values.tolist()

move_self = pd.read_csv('data/self_loop.csv')
move_self = move_self[['dow', 'p_out_boats']]
move_self = move_self.loc[move_self['dow'].isin(inspID)]
move_self2 = pd.merge(move_self, att[['dow', 'id', 'lake_name', 'acre', 'infest', 'accessible', 'ramp']], how = 'left', left_on = 'dow', 
	right_on = 'dow')
move_self2.loc[move_self2['p_out_boats'] >= 1, 'p_out_boats'] = 0.999
move_self2['log_acre'] = np.log(move_self2['acre'] + 1)
move_self2['log_acre2'] = move_self2['log_acre']**2
move_self2['log_acre3'] = move_self2['log_acre']**3
move_self2['log_ramp_acre'] = np.log(move_self2['ramp']/move_self2['acre'] + 1)
# move_self2['log_mile5'] = np.log(att['mile5']+1)
move_self2["acre_x_infest"] = move_self2['log_acre']*move_self2['infest']
# move_self2["acre_x_mile5"] = move_self2['log_acre']*att['mile5']
# move_self2["infest_x_mile5"] = move_self2['infest']*att['mile5']
move_self2['tr_p_out_boats'] = np.log((move_self2['p_out_boats'])/(1-move_self2['p_out_boats']))

'''
for ct in np.unique(att['county'].values).tolist(): 
	print(ct)
	move_self2['cty'+str(ct)] = 1.0*(move_self2['county'] == ct)


import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()

plt.scatter(move_self2['tr_p_out_boats'], move_self2['p_out_boats'], facecolor='dodgerblue')
plt.show()
'''

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

y_col = ['tr_p_out_boats']
x_col1 = ['infest', 'log_acre']

move_exog = move_self2[x_col1]
move_exog = sm.add_constant(move_exog)
move_endog = move_self2[y_col]
linear_model1 = smf.OLS(move_endog, move_exog)
linear_results1 = linear_model1.fit()

print(linear_results1.summary())
# print(linear_results1.aic)

Y_pred = linear_results1.predict(move_exog)
Y_pred = np.exp(Y_pred)/(1+np.exp(Y_pred))

coef_table = linear_results1.params
coef_table = pd.concat([coef_table, linear_results1.conf_int()], axis=1)
coef_table = pd.concat([coef_table, linear_results1.pvalues], axis=1)
coef_table.columns = ['coef', 'std err', 'z', 'P>|z|']
coef_table.to_csv("data/linear_results1.csv")

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = np.sqrt(mean_squared_error(move_self2['p_out_boats'], Y_pred))
print(rms)

import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()

plt.scatter(move_self2['p_out_boats'], Y_pred, facecolor='dodgerblue')
plt.show()


# choose model 1 and make prediction (because other parameters are not significant...)
att['p_self_boat'] = 0
att['log_acre'] = np.log(att['acre'] + 1)
X_var = att[x_col1]
X_var = sm.add_constant(X_var)
Y_pred = linear_results1.predict(X_var)
Y_pred = np.exp(Y_pred)/(1+np.exp(Y_pred))
att['p_self_boat'] = Y_pred

att[['p_self_boat']].to_csv('data/p_self_boat.csv', index = False)


'''
Creating 20 networks (annual)
'''
import numpy as np
from scipy.misc import comb
import scipy
from itertools import product
import pymc
import pandas as pd
import json
import timeit
import igraph
import copy
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
tot = np.round(att['boat_ann']/2)
logit = np.load('data/logit.npy')
gamma = np.load('data/gamma_pred.npy')
p_self = pd.read_csv('data/p_self_boat.csv')

inspID = att.loc[att['inspect'] == 1, 'id'].values.tolist()

del traffic

for j in range(1, 21): 
	aa = timeit.default_timer()
	temp_dum = np.random.binomial(n = 1, p = logit)
	# print(timeit.default_timer()-aa)
	np.fill_diagonal(temp_dum, 1)

	gamma[np.where(np.isnan(gamma))]=0

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
	boat_net[boat_net < 1] = 0

	ids = att['dow'].values

	boat_o = []
	boat_d = []
	boat_w = []

	# aa = timeit.default_timer()
	for i in range(0, boat_net.shape[0]):
		# print(i)
		temp_to = np.where((boat_net[i] > 0) & ~np.isnan(boat_net[i]))[0].tolist()
		temp_from = np.array(len(temp_to)*[i]).tolist()
		temp_weight = boat_net[i, temp_to].astype(float)
		temp_weight = temp_weight.tolist()
		temp_to = ids[temp_to].tolist()
		temp_from = ids[temp_from].tolist()

		boat_o.append(temp_from)
		boat_d.append(temp_to)
		boat_w.append(temp_weight)
	print(timeit.default_timer()-aa)

	boat_o = [val for sublist in boat_o for val in sublist]
	boat_d = [val for sublist in boat_d for val in sublist]
	boat_w = [val for sublist in boat_w for val in sublist]

	output = [boat_o, boat_d, boat_w]

	temp_move = {'id_origin': boat_o, 'id_destination': boat_d, 'weight': boat_w}
	temp_move = pd.DataFrame(temp_move)
	temp_move.to_csv('data/Annual boater net/boats'+str(j)+'.csv', index = False)


'''
Model performance (get true positive and true negative)
'''
import numpy as np
from scipy.misc import comb
import scipy
from itertools import product
import pymc
import pandas as pd
import json
import timeit
import igraph
import statsmodels
import os
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)

att = pd.read_csv('data/lake_attribute.csv')
att = att[['dow', 'lake_name', 'acre', 'utm_x', 'utm_y', 'county', 'county.name', 'infest', 'inspect', \
'infest.zm', 'infest.ss', 'zm_suit', 'ss_suit', 'id']]
traffic = pd.read_csv("data/traffics.csv")
att = pd.merge(att, traffic[['DOW', 'ExtraBoats']], how = 'left', left_on = 'dow', right_on = 'DOW')
att.columns = ['dow', 'lake_name', 'acre', 'utm_x', 'utm_y', 'county', 'county.name', 'infest', 'inspect', 
'infest.zm', 'infest.ss', 'zm_suit', 'ss_suit', 'id', 'DOW', 'boat_ann']
tot = np.round(att['boat_ann']/2)
logit = np.load('data/logit.npy')
gamma = np.load('data/gamma_pred.npy')
p_self = pd.read_csv('data/p_self_boat.csv')

inspID = att.loc[att['inspect'] == 1, 'id'].values.tolist()

aa = timeit.default_timer()
move = pd.read_csv('data/movement_for_stat.csv')
print(timeit.default_timer() - aa)

move = move.drop(['Unnamed: 0'], axis = 1)
print(move.shape)

move = move[['dow.origin', 'dow.destination', 'avg.count', 'dumBoat']]
move.columns = ['dow_origin', 'dow_destination', 'avg_count', 'dumBoat']


mean_boat = np.zeros(20)
max_boat = np.zeros(20)
std_boat = np.zeros(20)
true_pos_vec = np.zeros(20)
true_neg_vec = np.zeros(20)

for run in range(0, 20): 
	aa = timeit.default_timer()
	temp_dum = np.random.binomial(n = 1, p = logit)
	np.fill_diagonal(temp_dum, 1)

	# gamma[np.where(np.isnan(gamma))]=0

	out_net = gamma*temp_dum
	sum_out = np.sum(out_net, axis = 1)
	sum_out = np.tile(sum_out, (out_net.shape[0], 1)).T 
	self_loop = np.tile(p_self.values.T, (out_net.shape[0], 1)).T 
	p_out = out_net/sum_out * (1-self_loop)

	kk = np.diag_indices(p_out.shape[0])
	p_out[kk[0], kk[1]] = p_self.values.T

	tot_mat = np.tile(tot.values.T, (out_net.shape[0], 1)).T

	boat_net = tot_mat*p_out
	boat_net = np.ceil(boat_net)

	ids = att['dow'].values

	boat_o = []
	boat_d = []
	boat_w = []


	# aa = timeit.default_timer()
	for i in range(0, boat_net.shape[0]):
		# print(i)
		if i in inspID: 
			temp_to = np.where((boat_net[i] >= 1) & ~np.isnan(boat_net[i]))[0].tolist()
			temp_from = np.array(len(temp_to)*[i]).tolist()
			temp_weight = boat_net[i, temp_to].astype(float)
			temp_weight = temp_weight.tolist()
			temp_to = ids[temp_to].tolist()
			temp_from = ids[temp_from].tolist()
		else: 
			temp_to = np.where((boat_net[i] >= 1) & ~np.isnan(boat_net[i]))[0]
			temp_to = temp_to[np.in1d(temp_to, inspID)].tolist()
			temp_from = np.array(len(temp_to)*[i]).tolist()
			temp_weight = boat_net[i, temp_to].astype(float)
			temp_weight = temp_weight.tolist()
			temp_to = ids[temp_to].tolist()
			temp_from = ids[temp_from].tolist()

		boat_o.append(temp_from)
		boat_d.append(temp_to)
		boat_w.append(temp_weight)
	# print(timeit.default_timer()-aa)

	boat_o = [val for sublist in boat_o for val in sublist]
	boat_d = [val for sublist in boat_d for val in sublist]
	boat_w = [val for sublist in boat_w for val in sublist]

	output = [boat_o, boat_d, boat_w]

	temp_move = {'dow_origin': boat_o, 'dow_destination': boat_d, 'weight': boat_w}
	temp_move = pd.DataFrame(temp_move)
	temp_move.to_csv('data/Annual boater net/boats'+str(run+1)+'.csv', index = False)
	temp_move['sim_conn'] = np.where(temp_move['weight']>0, 1, 0)

	temp_move = pd.merge(temp_move, move[['dow_origin', 'dow_destination', 'dumBoat', 'avg_count']], 
		how = 'outer', left_on = ['dow_origin', 'dow_destination'], right_on = ['dow_origin', 'dow_destination'])
	temp_move['dumBoat'].fillna(0, inplace=True)
	temp_move['sim_conn'].fillna(0, inplace=True)
	# temp_move = temp_move.loc[temp_move['dow_origin'] != temp_move['dow_destination']]

	'''
	print(cross_table)
	print(cross_table/cross_table.ix['All'])
	print(cross_table/cross_table.ix['All']['All'])

	pd.set_option('display.expand_frame_repr', False)
	print(temp_move[['avg_count', 'dumBoat', 'sim_conn']].groupby(['dumBoat', 'sim_conn']).describe(percentiles = [0.7, 0.8, 0.9, 0.95]).T)
	print(temp_move[['weight', 'dumBoat', 'sim_conn']].groupby(['dumBoat', 'sim_conn']).describe(percentiles = [0.7, 0.8, 0.9, 0.95]).T)
	'''

	temp_move = temp_move.loc[temp_move['dow_origin'] != temp_move['dow_destination']]

	tmp = temp_move.loc[temp_move['sim_conn']==1, 'weight'].describe()
	mean_boat[run] = tmp.ix['mean']
	max_boat[run] = tmp.ix['max']
	std_boat[run] = tmp.ix['std']

	cross_table = pd.crosstab(index=temp_move['sim_conn'], 
	                           columns=temp_move['dumBoat'], 
	                           margins=True)

	tmp_table = cross_table/cross_table.ix['All']

	true_pos_vec[run] = tmp_table.iloc[1,1]
	true_neg_vec[run] = tmp_table.iloc[0,0]

	print(timeit.default_timer()-aa)


sim_net_data = pd.DataFrame({"mean_boat": mean_boat, 'max_boat': max_boat, 'std_boat': std_boat, 
	'true_pos': true_pos_vec, 'true_neg': true_neg_vec})
sim_net_data.to_csv('data/Annual boater net/sim_net_summary.csv')



inspID = att.loc[att['inspect'] == 1, 'dow'].values.tolist()
n_true_pos = []
n_false_neg = []
n_true_neg = []
n_false_pos = []
true_pos_rate = []
true_neg_rate = []

for i in range(len(inspID)): 
	j = inspID[i]
	tempSet = temp_move.loc[(temp_move['dow_origin'] == j) | (temp_move['dow_destination'] == j)]
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

	n_true_pos.append(n_true_pos0)
	n_false_neg.append(n_false_neg0)
	n_true_neg.append(n_true_neg0)
	n_false_pos.append(n_false_pos0)
	true_pos_rate.append(true_pos_rate0)
	true_neg_rate.append(true_neg_rate0)


temp_out = dict(zip(['dow', 'out_n_true_pos', 'out_n_false_neg', 'out_n_true_neg', 'out_n_false_pos', 'out_true_pos_rate', 'out_true_neg_rate'], 
	[inspID, n_true_pos, n_false_neg, n_true_neg, n_false_pos, true_pos_rate, true_neg_rate]))

temp_out = pd.DataFrame(temp_out)

att = pd.merge(att[['dow']], temp_out, how = 'left', left_on = 'dow', right_on = 'dow')
att.to_csv('/Users/szu-yukao/Documents/FishProject/Writing/from_estimation_weight.csv', index = False)

del temp_out, move, n_true_pos, n_false_neg, n_true_neg, n_false_pos, true_pos_rate, true_neg_rate



'''
creating network for calibration (weekly)
'''
import numpy as np
from scipy.misc import comb
import scipy
from itertools import product
import pymc
import pandas as pd
import json
import timeit
import igraph
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

tot = np.round(att[['boat_ann']]/2)
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
import pymc
import pandas as pd
import json
import timeit
import igraph
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
tot = np.round(att['boat_ann']/2)
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










