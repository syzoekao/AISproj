
def model_performance(model, data_set, target, predict_names, pred_type, path, filename): 
	feature_importance = model.feature_importances_
	ix = np.argsort(-feature_importance)
	sort_predictor_col = np.array(predict_names)[ix].tolist()
	feature_importance = feature_importance[ix]

	plt.clf()
	fig = plt.figure(figsize=(12,6))
	x_pos = np.arange(len(feature_importance))
	plt.bar(x_pos, feature_importance, align='center', color = 'cornflowerblue', edgecolor = 'grey')
	plt.xticks(x_pos, sort_predictor_col, rotation=90)
	plt.ylabel('Feature importance')
	plt.xlabel('Features')
	fig.subplots_adjust(bottom = 0.3)
	fig.tight_layout()
	plt.savefig(path + filename + '.eps', format='eps', dpi=1000)

	if pred_type == 'binary': 
		y_pred = model.predict_proba(data_set)[:, 1]
		x_vec0 = np.arange(0.2, 0.51, 0.01)
		accuracy_vec = np.zeros(x_vec0.shape[0])
		sens_vec = np.zeros(x_vec0.shape[0])

		for x in range(x_vec0.shape[0]): 
			y_hat = 1 * (y_pred >= x_vec0[x])
			accuracy_vec[x] = accuracy_score(target, y_hat)
			conf_mat = confusion_matrix(target, y_hat)
			spec_sens = np.round(np.diagonal(conf_mat) / np.sum(conf_mat, axis = 1), 5).tolist()
			sens_vec[x] = spec_sens[1]	
				
		x_sol = np.where(sens_vec == np.min(sens_vec[sens_vec > 0.95]))[0]
		x_vec = x_vec0[np.where(sens_vec == np.min(sens_vec[sens_vec > 0.95]))[0]]
		# y_hat = 1 * (y_pred >= x_vec)
		y_hat = 1 * (y_pred >= 0.5)
		accuracy = accuracy_score(target, y_hat)
		auc = roc_auc_score(target, y_hat)

		conf_mat = confusion_matrix(target, y_hat)
		spec_sens = np.round(np.diagonal(conf_mat) / np.sum(conf_mat, axis = 1), 5).tolist()
		conf_mat_df = pd.DataFrame(conf_mat.T)
		conf_mat_df = conf_mat_df.reset_index()
		conf_mat_df.columns = ["prediction", "0", "1"]

		text_file = open(path + filename + ".txt", "w")
		# text_file.write('threshold: ' + str(np.round(x_vec[0], 2)) + '\n' + 
		text_file.write('threshold: ' + str(0.5) + '\n' + 
			'confustion matrix\n\t\t\tdata\n' + 
			conf_mat_df.to_string() + '\n\n' + 
			'specificity: ' + str(spec_sens[0]) + '\n' +
			'sensitivity: ' + str(spec_sens[1]) + '\n' + 
			'accuracy: ' + str(np.round(accuracy, 3)) + '\n' + 
			'auc: ' + str(np.round(auc, 3)))
		text_file.close()	
	
		plt.clf()
		fig = plt.figure(figsize=(12,6))
		plt.plot(x_vec0, accuracy_vec, color = 'cornflowerblue', label = 'accuracy')
		plt.plot(x_vec0, sens_vec, color = 'limegreen', label = 'sensitivity')
		plt.hlines(0.95, np.min(x_vec0), np.max(x_vec0), color = 'black', linestyles = 'dashed', label = '0.95')
		plt.hlines(0.92, np.min(x_vec0), np.max(x_vec0), color = 'black', linestyles = 'dashdot', label = '0.92')
		plt.ylabel('value')
		plt.xlabel('threshold')
		plt.legend()
		fig.tight_layout()
		plt.savefig(path + filename + '(accuracy and sensitivity).eps', format='eps', dpi=1000)

	else: 
		y_pred = model.predict(data_set)
		y_pred = np.exp(y_pred) - 1
		y_pred[y_pred < 1] = 1
		target_tr = np.exp(target) - 1
		corr = np.round(np.corrcoef(target_tr, y_pred)[0, 1], 3)
		print(corr)

		plt.clf()
		fig = plt.figure(figsize=(6,4))
		plt.scatter(target_tr, y_pred, s = 10, 
			facecolors='none', edgecolors='cornflowerblue')
		plt.ylabel('predicted boats')
		plt.xlabel('boats')
		plt.text(500, 500, "corr = " + str(corr))
		fig.tight_layout()
		plt.savefig(path + filename + '(pred vs actual).png', format='png', dpi=500)

	plt.clf()
	fig = plt.figure(figsize=(6,4))
	plt.hist(y_pred, facecolor = 'cornflowerblue', bins = 50, alpha = 0.7, label = 'predicted')
	if pred_type == 'continuous': 
		plt.hist(target_tr, facecolor = 'limegreen', bins = 50, alpha = 0.7, label = 'data')
	plt.ylabel('frequency')
	plt.xlabel('predicted value')
	plt.legend()
	fig.tight_layout()
	plt.savefig(path + filename + '(distribution).png', format='png', dpi=500)

	plt.close('all')


# Predict annual arrival boats

import numpy as np
import pandas as pd
import json
import timeit
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import re
import os
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)

import psutil

import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()

att = pd.read_csv('data/lake_attribute.csv')
att['log_boats'] = np.log(att['AnnualTraffic'])
att['log_acre'] = np.log(att['acre'])
att['log_nLake30'] = np.log(att['nLake30'])
att['log_pop'] = np.log(att['countypop'])
att['log_d2hwy'] = np.log(att['d2HWYmeter'] + 1)
att['log_acre_pop'] = np.log(att['acre'] * att['countypop'])
att['log_infest_nLake30'] = att['infest'] * np.log(att['nLake30'])
att['log_infest_acre'] = att['infest'] * np.log(att['acre'])
att['log_infest_pop'] = att['infest'] * np.log(att['countypop'])
att['log_infest_d2hwy'] = att['infest'] * np.log(att['d2HWYmeter'] + 1)
att['log_ramp_acre'] = np.log(att['ramp'] / att['acre'] * 1000 + 1)

tmp_att = att.loc[att['include_insp'] == 1]

y_col = ['log_boats']

x_col = ['log_acre', 'infest', 'accessible', 'log_nLake30', 'log_pop', 'log_d2hwy', 'log_acre_pop', 
'log_infest_acre', 'log_infest_d2hwy', 'log_ramp_acre']


train_ix = np.sort(np.random.choice(np.arange(0, tmp_att.shape[0], 1), 
	size = np.round(tmp_att.shape[0] * 0.9).astype(int), replace = False))
test_ix = np.setdiff1d(np.arange(0, tmp_att.shape[0], 1), train_ix)

train_set = tmp_att.iloc[train_ix]
train_set = train_set[x_col].to_numpy()
test_set = tmp_att.iloc[test_ix]
test_set = test_set[x_col].to_numpy()

train_Y = tmp_att.iloc[train_ix]
train_Y = train_Y[y_col].to_numpy().T[0]
test_Y = tmp_att.iloc[test_ix]
test_Y = test_Y[y_col].to_numpy().T[0]


xgb1 = XGBRegressor(
	learning_rate = 0.1,
	n_estimators = 5000,
	max_depth = 10,
	min_child_weight = 19,
	gamma = 0.3,
	eta = 0.3, 
	subsample = 0.5,
	colsample_bytree = 0.8,
	objective = 'reg:squarederror',
	booster = 'gbtree', 
	nthread = os.cpu_count() - 1,
	eval_metric = 'rmse')


param_test1 = {
	'max_depth': [5, 6, 7, 8, 9, 10],
	# 'min_child_weight': [x for x in range(18, 21, 1)],
	# 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5],
	# 'subsample': [0.3, 0.4, 0.5, 0.6],
	# 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]
}

aa = timeit.default_timer()
gsearch1 = GridSearchCV(estimator = xgb1, param_grid = param_test1, 
	scoring='neg_mean_squared_error', iid = False, cv = 3, verbose = 2)
gsearch1.fit(train_set, train_Y)
print(timeit.default_timer() -  aa)

print(gsearch1.best_params_)
print(gsearch1.best_score_)
print(pd.Series(gsearch1.cv_results_))


xgb1 = XGBRegressor(
	learning_rate = 0.1,
	n_estimators = 167,
	max_depth = 6,
	min_child_weight = 19,
	gamma = 0.5,
	eta = 0.3, 
	subsample = 0.5,
	colsample_bytree = 0.8,
	objective = 'reg:squarederror',
	booster = 'gbtree', 
	nthread = os.cpu_count() - 1,
	eval_metric = 'rmse')

aa = timeit.default_timer()
eval_set = [(test_set, test_Y)]
xgbmodel = xgb1.fit(train_set, train_Y, eval_set = eval_set, verbose = True, early_stopping_rounds = 100)
print(timeit.default_timer() -  aa)

model_performance(xgbmodel, test_set, test_Y, x_col, pred_type = 'continuous', 
	path = 'boater_gen_out/', filename = 'annualTraffic')


tempDF = att[x_col]
pred_boat = xgbmodel.predict(tempDF.to_numpy())
pred_boat = np.exp(pred_boat)

import copy
pred_boat_copy = copy.deepcopy(pred_boat)

att = pd.read_csv('data/lake_attribute.csv')
att['annualTraffic2'] = pred_boat

tmp_real = att.loc[att['include_insp'] == 1, 'annualTraffic'].to_numpy()
tmp_pred = att.loc[att['include_insp'] == 1, 'annualTraffic2'].to_numpy()

path = 'boater_gen_out/'
filename = 'annualTraffic'

corr = np.round(np.corrcoef(tmp_real, tmp_pred)[0, 1], 3)
print(corr)

plt.clf()
fig = plt.figure(figsize=(6,4))
plt.scatter(tmp_real, tmp_pred, s = 10, 
	facecolors='none', edgecolors='cornflowerblue')
plt.ylabel('predicted boats')
plt.xlabel('boats')
plt.text(500, 500, "corr = " + str(corr))
fig.tight_layout()
plt.savefig(path + filename + '(pred vs actual).png', format='png', dpi=500)

att.to_csv('data/lake_attribute.csv', index = False)


import numpy as np
import pandas as pd
import json
import timeit
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import re
import os
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)

import psutil

import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()

aa = timeit.default_timer()
move = pd.read_csv('data/movement_for_stat.csv')
print(timeit.default_timer() - aa)

move = move.drop(['Unnamed: 0'], axis = 1)
move = move.loc[move["dow_origin"] != move["dow_destination"]]
move['log_dist_x_dist30'] = move['log_distance'] * move['dist30']

sel_col = ['dumBoat', 'log_distance', 'gravity', 'log_acre_o', 'log_acre_d', 
'infest_o', 'infest_d', 'both_infest', 'both_insp', 'insp_d', 'insp_o', 
'nLake30_origin', 'nLake30_destination', 'gravity_nLake', 
'dist30', 'log_ramp_acre_o', 'log_ramp_acre_d', 'both_insp_infest', 
'gravity_both_insp', 'gravity_both_infest', 'grLake_both_insp', 
'grLake_both_infest', 'log_dist_x_dist30']

predictor_col = ['log_distance', 'gravity', 'log_acre_o', 'log_acre_d', 
'infest_o', 'infest_d', 'both_infest', 'both_insp', 'insp_d', 'insp_o', 
'nLake30_origin', 'nLake30_destination', 'gravity_nLake', 
'dist30', 'log_ramp_acre_o', 'log_ramp_acre_d', 'both_insp_infest', 
'gravity_both_insp', 'gravity_both_infest', 'grLake_both_insp', 
'grLake_both_infest', 'log_dist_x_dist30']

outcome_col = ['dumBoat']

process = psutil.Process(os.getpid())
print(process.memory_info().rss)

train_ix = np.sort(np.random.choice(np.arange(0, move.shape[0], 1), 
	size = np.round(move.shape[0] * 0.8).astype(int), replace = False))
test_ix = np.setdiff1d(np.arange(0, move.shape[0], 1), train_ix)

train_set = move.iloc[train_ix]
train_set = train_set[predictor_col].to_numpy()
test_set = move.iloc[test_ix]
test_set = test_set[predictor_col].to_numpy()

train_Y = move.iloc[train_ix]
train_Y = train_Y[outcome_col].to_numpy().T[0]
test_Y = move.iloc[test_ix]
test_Y = test_Y[outcome_col].to_numpy().T[0]


xgb1 = XGBClassifier(
	learning_rate = 0.1,
	n_estimators = 1000,
	max_depth = 8,
	min_child_weight = 28,
	gamma = 0.4,
	eta = 0.3, 
	subsample = 0.4,
	colsample_bytree = 0.7,
	objective = 'binary:logistic',
	booster = 'gbtree', 
	nthread = os.cpu_count() - 1,
	eval_metric = 'auc', 
	scale_pos_weight = 168)

# tuning parameters

param_test1 = {
	'max_depth': [5, 6, 7, 8, 9]
	'min_child_weight': [27, 28, 29, 30, 31]
	'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]
	'subsample': [0.3, 0.4, 0.5, 0.6]
	'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]
}

aa = timeit.default_timer()
gsearch1 = GridSearchCV(estimator = xgb1, param_grid = param_test1, 
	scoring='roc_auc', iid = False, cv = 3, verbose = 2)
gsearch1.fit(train_set, train_Y)
print(timeit.default_timer() -  aa)

print(gsearch1.best_params_)
print(gsearch1.best_score_)
print(pd.Series(gsearch1.cv_results_))

# Final model for binary outcomes (link prediction)
xgb1 = XGBClassifier(
	learning_rate = 0.1,
	n_estimators = 232,
	max_depth = 8,
	min_child_weight = 28,
	gamma = 0.4,
	eta = 0.3, 
	subsample = 0.4,
	colsample_bytree = 0.7,
	objective = 'binary:logistic',
	booster = 'gbtree', 
	nthread = os.cpu_count() - 1,
	eval_metric = 'auc', 
	scale_pos_weight = 168)


aa = timeit.default_timer()
eval_set = [(test_set, test_Y)]
xgbmodel = xgb1.fit(train_set, train_Y, eval_set = eval_set, verbose = True, early_stopping_rounds = 5)
print(timeit.default_timer() -  aa)


plt.close('all')
model_performance(xgbmodel, test_set, test_Y, predictor_col, pred_type = 'binary', 
	path = 'boater_gen_out/', filename = 'boater_link_prediction')


# building data
del move, train_ix, test_ix, train_set, test_set, train_Y, test_Y

att = pd.read_csv('data/lake_attribute.csv')

inspID = att.loc[att['inspect']==1, 'id'].values.tolist()
ids = att['id'].tolist()
pred_prob = np.empty((att.shape[0], att.shape[0]))


for ix in ids: 
	print(ix)
	tempDF = pd.DataFrame([], columns = predictor_col)
	tempDF['infest_o'] = np.repeat(att['infest'].iloc[ix], att.shape[0])
	tempDF['infest_d'] = att['infest']
	utm_x = att['utm_x'].iloc[ix]
	utm_y = att['utm_y'].iloc[ix]
	tmp_dist = np.sqrt((att['utm_x'] - utm_x) ** 2 + (att['utm_y'] - utm_y) ** 2) / 1000
	tempDF['log_distance'] = np.log(tmp_dist + 1)
	tempDF['dist30'] = 1 * (tmp_dist < 30)
	tempDF['log_dist_x_dist30'] = tempDF['log_distance'] * tempDF['dist30']

	tmp = att['acre'].iloc[ix]
	tempDF['log_acre_diff'] = np.log(np.absolute(att['acre'] - tmp) + 1)
	tempDF['log_acre_o'] = np.log(tmp + 1)
	tempDF['log_acre_d'] = np.log(att['acre'] + 1)
	tmp = att['infest'].iloc[ix]
	tempDF['both_infest'] = 1*(att['infest'] * tmp)

	tempDF['insp_d'] = att['inspect']
	tmp = att['inspect'].iloc[ix]
	tempDF['insp_o'] = np.repeat(tmp, att.shape[0])
	tempDF['both_insp'] = 1 * (att['inspect'] * tmp)

	tmp = att['acre'].iloc[ix]
	tmp1 = att['ramp'].iloc[ix]
	tempDF['log_ramp_acre_o'] = np.log(tmp1/tmp * 1000 + 1)
	tempDF['log_ramp_acre_d'] = np.log(att['ramp']/att['acre'] * 1000 + 1)

	tmp = att['acre'].iloc[ix]
	tempDF['gravity'] = np.log(tmp * att['acre'] / (tmp_dist + 1))

	tempDF['nLake30_origin'] = np.repeat(att['nLake30'].iloc[ix], att.shape[0])
	tempDF['nLake30_destination'] = att['nLake30']
	tempDF['gravity_nLake'] = np.log((tempDF['nLake30_origin'] * tempDF['nLake30_destination'])/(tmp_dist + 1))

	tempDF['both_insp_infest'] = tempDF['both_infest'] * tempDF['both_insp']
	tempDF['gravity_both_insp'] = tempDF['gravity'] * tempDF['both_insp']
	tempDF['gravity_both_infest'] = tempDF['gravity'] * tempDF['both_infest']
	tempDF['grLake_both_insp'] = tempDF['gravity_nLake'] * tempDF['both_insp']
	tempDF['grLake_both_infest'] = tempDF['gravity_nLake'] * tempDF['both_infest']

	data_set = tempDF[predictor_col].to_numpy()
	y_pred = xgbmodel.predict_proba(data_set)[:, 1]
	pred_prob[ix] = y_pred


inspID = att.loc[att['inspect'] == 1, 'id'].tolist()
print(np.sum(pred_prob[inspID] >= 0.5))

aa = timeit.default_timer()
np.save("data/predprob", pred_prob)
print(timeit.default_timer() - aa)

aa = timeit.default_timer()
move = pd.read_csv('data/movement_for_stat.csv')
print(timeit.default_timer() - aa)

move = move.drop(['Unnamed: 0'], axis = 1)
move = move.loc[move["dow_origin"] != move["dow_destination"]]
move = move.loc[move["dumBoat"] == 1]

move['log_dist_x_dist30'] = move['log_distance'] * move['dist30']

pred_prob = np.load("data/predprob.npy")

pred_dum = 1 * (pred_prob >= 0.5)
np.fill_diagonal(pred_dum, 0)

real_data = np.zeros(pred_dum.shape)
for ix in range(move.shape[0]): 
	ix_origin = move.iloc[ix]['id_origin'].astype(int)
	ix_dest = move.iloc[ix]['id_destination'].astype(int)
	real_data[ix_origin, ix_dest] = 1

np.fill_diagonal(real_data, 0)
real_ix = np.where(real_data == 1)
print((np.sum(pred_dum[real_ix])) / real_ix[0].shape[0])


'''
predicting weight: number of boaters
'''

import numpy as np
import pandas as pd
import json
import timeit
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import re
import os
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)

import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()

aa = timeit.default_timer()
move = pd.read_csv('data/movement_for_stat.csv')
print(timeit.default_timer() - aa)

move = move.drop(['Unnamed: 0'], axis = 1)
move = move.loc[(move["inBoat"] == 1)]
# move = move.loc[move["self_loop"] == 0]
# move = move.loc[(move['normBoats'] < 5000)]

# move['log_distance2'] = move['log_distance'] ** 2
# move['log_distance3'] = move['log_distance'] ** 3

# move['log_acre_diff2'] = move['log_acre_diff'] ** 2
# move['log_acre_diff3'] = move['log_acre_diff'] ** 3

# move['log_acre_o2'] = move['log_acre_o'] ** 2

# move['log_acre_d2'] = move['log_acre_d'] ** 2
# move['log_acre_d3'] = move['log_acre_d'] ** 3

move['log_distance_x_dist30'] = move['log_distance'] * move['dist30']

move['log_normBoats'] = np.log(move['normBoats'])

y_col = ['log_normBoats']

x_col = ['log_distance', 'log_acre_o', 'log_acre_d', \
'infest_o', 'infest_d', 'both_infest', 'insp_o', 'both_insp', \
'log_ramp_acre_o', 'log_ramp_acre_d', 'log_distance_x_dist30', \
'gravity', 'nLake30_origin', 'nLake30_destination', 'gravity_nLake', 'both_insp_infest', \
'gravity_both_insp', 'gravity_both_infest', 'grLake_both_insp', 'grLake_both_infest']


train_ix = np.sort(np.random.choice(np.arange(0, move.shape[0], 1), 
	size = np.round(move.shape[0] * 0.8).astype(int), replace = False))
test_ix = np.setdiff1d(np.arange(0, move.shape[0], 1), train_ix)

train_set = move.iloc[train_ix]
train_set = train_set[x_col].to_numpy()
test_set = move.iloc[test_ix]
test_set = test_set[x_col].to_numpy()

train_Y = move.iloc[train_ix]
train_Y = train_Y[y_col].to_numpy().T[0]
test_Y = move.iloc[test_ix]
test_Y = test_Y[y_col].to_numpy().T[0]


xgb1 = XGBRegressor(
	learning_rate = 0.1,
	n_estimators = 5000,
	max_depth = 8,
	min_child_weight = 19,
	gamma = 0.3,
	eta = 0.3, 
	subsample = 0.5,
	colsample_bytree = 1,
	objective = 'reg:squarederror',
	booster = 'gbtree', 
	nthread = os.cpu_count() - 1,
	eval_metric = 'rmse')

aa = timeit.default_timer()
eval_set = [(test_set, test_Y)]
xgbmodel = xgb1.fit(train_set, train_Y, eval_set = eval_set, verbose = True, early_stopping_rounds = 10)
print(timeit.default_timer() -  aa)

# tuning parameters

param_test1 = {
	'max_depth': [5, 6, 7, 8, 9, 10],
	'min_child_weight': [x for x in range(18, 21, 1)],
	'gamma': [0.1, 0.2, 0.3, 0.4, 0.5],
	'subsample': [0.6, 0.7, 0.8, 0.9, 1],
	'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]
}

aa = timeit.default_timer()
gsearch1 = GridSearchCV(estimator = xgb1, param_grid = param_test1, 
	scoring='neg_mean_squared_error', iid = False, cv = 3, verbose = 2)
gsearch1.fit(train_set, train_Y)
print(timeit.default_timer() -  aa)

print(gsearch1.best_params_)
print(gsearch1.best_score_)
print(pd.Series(gsearch1.cv_results_))


# final model 

xgb1 = XGBRegressor(
	learning_rate = 0.1,
	n_estimators = 552,
	max_depth = 8,
	min_child_weight = 19,
	gamma = 0.3,
	eta = 0.3, 
	subsample = 0.5,
	colsample_bytree = 1,
	objective = 'reg:squarederror',
	booster = 'gbtree', 
	nthread = os.cpu_count() - 1,
	eval_metric = 'rmse')

aa = timeit.default_timer()
eval_set = [(test_set, test_Y)]
xgbmodel = xgb1.fit(train_set, train_Y, eval_set = eval_set, verbose = True)
print(timeit.default_timer() -  aa)

model_performance(xgbmodel, test_set, test_Y, x_col, pred_type = 'continuous', 
	path = 'boater_gen_out/', filename = 'boaters_prediction')


# building data
del move, train_ix, test_ix, train_set, test_set, train_Y, test_Y

att = pd.read_csv('data/lake_attribute.csv')
ids = att['id'].tolist()
pred_boat = np.empty((att.shape[0], att.shape[0]))

for ix in range(pred_boat.shape[0]): 
	# aaa = timeit.default_timer()
	# aa = timeit.default_timer()
	print(ix)
	tempDF = pd.DataFrame([], columns = x_col)
	tempDF['infest_d'] = np.repeat(att['infest'].iloc[ix], att.shape[0])
	tempDF['infest_o'] = att['infest']
	tmp = att['county'].iloc[ix]
	utm_x = att['utm_x'].iloc[ix]
	utm_y = att['utm_y'].iloc[ix]
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====1")
	tmp_dist = np.sqrt((att['utm_x'] - utm_x) ** 2 + (att['utm_y'] - utm_y) ** 2) / 1000
	tempDF['log_distance'] = np.log(tmp_dist + 1)
	tempDF['dist30'] = 1 * (tmp_dist < 30)
	tempDF['log_distance_x_dist30'] = tempDF['log_distance']*tempDF['dist30']
	# tempDF['log_distance2'] = tempDF['log_distance'] ** 2
	# tempDF['log_distance3'] = tempDF['log_distance'] ** 3
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====2")
	tmp = att['acre'].iloc[ix]
	# tempDF['log_acre_diff'] = np.log(np.absolute(att['acre'] - tmp) + 1)
	# tempDF['log_acre_diff2'] = tempDF['log_acre_diff'] ** 2
	# tempDF['log_acre_diff3'] = tempDF['log_acre_diff'] ** 3
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====3")
	tempDF['log_acre_d'] = np.log(tmp + 1)
	# tempDF['log_acre_o2'] = tempDF['log_acre_o'] ** 2 
	tempDF['log_acre_o'] = np.log(att['acre'] + 1)
	# tempDF['log_acre_d2'] = tempDF['log_acre_d'] ** 2
	# tempDF['log_acre_d3'] = tempDF['log_acre_d'] ** 3
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====6")
	tmp = att['infest'].iloc[ix]
	tempDF['both_infest'] = 1 * (att['infest'] * tmp)
	tmp = att['inspect'].iloc[ix]
	tempDF['insp_d'] = tmp
	tempDF['insp_o'] = att['inspect']
	tempDF['both_insp'] = 1 * (att['inspect'] * tmp)
	# print(timeit.default_timer() - aa)

	tmp1 = att['ramp'].iloc[ix]
	tmp2 = att['acre'].iloc[ix]
	tempDF['log_ramp_acre_d'] = np.log(tmp1 / tmp2 * 1000 +1)
	tempDF['log_ramp_acre_o'] = np.log(att['ramp'] / att['acre'] * 1000 + 1)

	# tempDF['self_loop'] = 0
	# tempDF.iloc[ix]['self_loop'] = 1

	tmp = att['acre'].iloc[ix]
	tempDF['gravity'] = np.log(tmp * att['acre'] / (tmp_dist + 1))

	tempDF['nLake30_destination'] = np.repeat(att['nLake30'].iloc[ix], att.shape[0])
	tempDF['nLake30_origin'] = att['nLake30']
	tempDF['gravity_nLake'] = np.log((tempDF['nLake30_origin'] * tempDF['nLake30_destination'])/(tmp_dist + 1))

	tempDF['both_insp_infest'] = tempDF['both_infest'] * tempDF['both_insp']
	tempDF['gravity_both_insp'] = tempDF['gravity'] * tempDF['both_insp']
	tempDF['gravity_both_infest'] = tempDF['gravity'] * tempDF['both_infest']
	tempDF['grLake_both_insp'] = tempDF['gravity_nLake'] * tempDF['both_insp']
	tempDF['grLake_both_infest'] = tempDF['gravity_nLake'] * tempDF['both_infest']

	tempDF = tempDF[x_col]
	pred_boat[:, ix] = xgbmodel.predict(tempDF.to_numpy())


pred_boat = np.ceil(np.exp(pred_boat) - 1)
pred_boat[pred_boat < 1] = 1

aa = timeit.default_timer()
np.save("data/predboat", pred_boat)
print(timeit.default_timer() - aa)


aa = timeit.default_timer()
move = pd.read_csv('data/movement_for_stat.csv')
print(timeit.default_timer() - aa)

move = move.drop(['Unnamed: 0'], axis = 1)
move = move.loc[move["inBoat"] == 1]

pred_Y = np.zeros(move.shape[0])
for x in range(pred_Y.shape[0]): 
	ix_x = move['id_origin'].iloc[x].astype(int)
	ix_y = move['id_destination'].iloc[x].astype(int)
	pred_Y[x] = pred_boat[ix_x, ix_y]

filename = 'boaters_prediction'
path = 'boater_gen_out/'

tmp_ix = np.where(move[['dow_destination']].to_numpy().T[0] != move[['dow_origin']].to_numpy().T[0])[0]
# tmp_ix = np.intersect1d(np.where(move[['normBoats']].to_numpy().T[0] < 2000)[0], tmp_ix)

tmp_real = move[['normBoats']].to_numpy().T[0]

corr = np.round(np.corrcoef(pred_Y, tmp_real)[0, 1], 3)
print(corr)

plt.clf()
fig = plt.figure(figsize=(6,4))
plt.scatter(tmp_real, pred_Y, s = 10, 
	facecolors='none', edgecolors='cornflowerblue')
plt.ylabel('predicted boats')
plt.xlabel('boats')
plt.text(500, 1000, "corr = " + str(corr))
fig.tight_layout()
plt.savefig(path + filename + '(pred vs actual) post.png', format='png', dpi=500)


corr = np.round(np.corrcoef(pred_Y[tmp_ix], tmp_real[tmp_ix])[0, 1], 3)
print(corr)

plt.clf()
fig = plt.figure(figsize=(6,4))
plt.scatter(tmp_real[tmp_ix], pred_Y[tmp_ix], s = 10, 
	facecolors='none', edgecolors='cornflowerblue')
plt.ylabel('predicted boats')
plt.xlabel('boats')
plt.text(500, 500, "corr = " + str(corr))
fig.tight_layout()
plt.savefig(path + filename + '(pred vs actual) post no self loop.png', format='png', dpi=500)




