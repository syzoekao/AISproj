
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
		accuracy_vec = np.zeros(x_vec.shape[0])
		sens_vec = np.zeros(x_vec.shape[0])

		for x in range(x_vec.shape[0]): 
			y_hat = 1 * (y_pred >= x_vec[x])
			accuracy_vec[x] = accuracy_score(target, y_hat)
			conf_mat = confusion_matrix(target, y_hat)
			spec_sens = np.round(np.diagonal(conf_mat) / np.sum(conf_mat, axis = 1), 5).tolist()
			sens_vec[x] = spec_sens[1]	
				
		x_sol = np.where(sens_vec == np.min(sens_vec[sens_vec > 0.95]))[0]
		x_vec = x_vec0[np.where(sens_vec == np.min(sens_vec[sens_vec > 0.95]))[0]]
		y_hat = 1 * (y_pred >= x_vec)
		accuracy = accuracy_score(target, y_hat)
		auc = roc_auc_score(target, y_hat)

		conf_mat = confusion_matrix(target, y_hat)
		spec_sens = np.round(np.diagonal(conf_mat) / np.sum(conf_mat, axis = 1), 5).tolist()
		conf_mat_df = pd.DataFrame(conf_mat.T)
		conf_mat_df = conf_mat_df.reset_index()
		conf_mat_df.columns = ["prediction", "0", "1"]

		text_file = open(path + filename + ".txt", "w")
		text_file.write('threshold: ' + str(np.round(x_vec[0], 2)) + '\n' + 
			'confustion matrix\n\t\t\tdata\n' + 
			conf_mat_df.to_string() + '\n\n' + 
			'specificity: ' + str(spec_sens[0]) + '\n' +
			'sensitivity: ' + str(spec_sens[1]) + '\n' + 
			'accuracy: ' + str(np.round(accuracy, 3)) + '\n' + 
			'auc: ' + str(np.round(auc, 3)))
		text_file.close()	
	
		plt.clf()
		fig = plt.figure(figsize=(12,6))
		plt.plot(x_vec, accuracy_vec, color = 'cornflowerblue', label = 'accuracy')
		plt.plot(x_vec, sens_vec, color = 'limegreen', label = 'sensitivity')
		plt.hlines(0.95, np.min(x_vec), np.max(x_vec), color = 'black', linestyles = 'dashed', label = '0.95')
		plt.hlines(0.92, np.min(x_vec), np.max(x_vec), color = 'black', linestyles = 'dashdot', label = '0.92')
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

		plt.clf()
		fig = plt.figure(figsize=(6,4))
		plt.scatter(target_tr, y_pred, s = 10, 
			facecolors='none', edgecolors='cornflowerblue')
		plt.ylabel('predicted boats')
		plt.xlabel('boats')
		plt.text(500, 7000, "corr = " + str(corr))
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
move['log_dist_x_dist50'] = move['log_distance'] * move['dist50']

sel_col = ['dumBoat', 'log_distance', 'log_acre_diff', 'log_acre_o', 
'log_acre_d', 'log_boat_o', 'log_boat_d','log_boat_diff', 'infest_o', 'infest_d', 
'same_insp', 'dist50', 'insp_d', 'log_ramp_acre_o', 'log_ramp_acre_d', 
'log_dist_x_dist50', 'both_infest']

predictor_col = ['log_distance', 'log_acre_diff', 'log_acre_o', 
	'log_acre_d', 'log_boat_o', 'log_boat_d','log_boat_diff', 'infest_o', 'infest_d', 
	'same_insp', 'dist50', 'insp_d', 'log_ramp_acre_o', 'log_ramp_acre_d', 
	'log_dist_x_dist50', 'both_infest']

outcome_col = ['dumBoat']

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


aa = timeit.default_timer()
clf = LogisticRegression().fit(train_set, train_Y)
print(timeit.default_timer() -  aa)

y_pred = clf.predict_proba(test_set)[:, 1]
y_hat = 1 * (y_pred > 0.001)
conf_mat = pd.crosstab(test_Y, y_hat, rownames=['Actual'], colnames=['Predicted'])
accuracy = accuracy_score(test_Y, y_hat)
auc = roc_auc_score(test_Y, y_pred)
sens = conf_mat.iloc[1, 1] / (conf_mat.iloc[1, 0] + conf_mat.iloc[1, 1])
spec = conf_mat.iloc[0, 0] / (conf_mat.iloc[0, 0] + conf_mat.iloc[0, 1])
print(conf_mat)
print("sens: " + str(np.round(sens, 3)) + "; spec: " + str(np.round(spec, 3)))

xgb1 = XGBClassifier(
	learning_rate = 0.1,
	n_estimators = 20,
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
	# 'max_depth': [5, 6, 7, 8, 9]
	# 'min_child_weight': [27, 28, 29, 30, 31]
	# 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]
	# 'subsample': [0.3, 0.4, 0.5, 0.6]
	# 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]
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
	n_estimators = 500,
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
xgbmodel = xgb1.fit(train_set, train_Y, eval_set = eval_set, early_stopping_rounds = 5, verbose = True)
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
	tmp = np.sqrt((att['utm_x'] - utm_x) ** 2 + (att['utm_y'] - utm_y) ** 2) / 1000
	tempDF['log_distance'] = np.log(tmp + 1)
	tempDF['dist50'] = 1 * (tmp < 50)
	tempDF['log_dist_x_dist50'] = tempDF['log_distance'] * tempDF['dist50']
	tmp = att['acre'].iloc[ix]
	tempDF['log_acre_diff'] = np.log(np.absolute(att['acre'] - tmp) + 1)
	tempDF['log_acre_o'] = np.log(tmp + 1)
	tempDF['log_acre_d'] = np.log(att['acre'] + 1)
	tmp = att['boat'].iloc[ix]
	tempDF['log_boat_o'] = np.log(tmp + 1)
	tempDF['log_boat_d'] = np.log(att['boat'] + 1)
	tempDF['log_boat_diff'] = np.log(np.absolute(att['boat'] - tmp) + 1)
	tmp = att['infest'].iloc[ix]
	tempDF['both_infest'] = 1*(att['infest'] * tmp)

	tempDF['insp_d'] = att['inspect']
	tmp = att['inspect'].iloc[ix]
	tempDF['same_insp'] = 1*((att['inspect'] == tmp) & (tmp == 1))

	tmp = att['acre'].iloc[ix]
	tmp1 = att['ramp'].iloc[ix]
	tempDF['log_ramp_acre_o'] = np.log(tmp1/tmp * 1000 + 1)
	tempDF['log_ramp_acre_d'] = np.log(att['ramp']/att['acre'] * 1000 + 1)

	data_set = tempDF[predictor_col].to_numpy()
	y_pred = xgbmodel.predict_proba(data_set)[:, 1]
	pred_prob[ix] = y_pred

inspID = att.loc[att['inspect'] == 1, 'id'].tolist()
print(np.sum(pred_prob[inspID] >= 0.35))

aa = timeit.default_timer()
np.save("data/predprob", pred_prob)
print(timeit.default_timer() - aa)

aa = timeit.default_timer()
move = pd.read_csv('data/movement_for_stat.csv')
print(timeit.default_timer() - aa)

move = move.drop(['Unnamed: 0'], axis = 1)
move = move.loc[move["dow_origin"] != move["dow_destination"]]
move = move.loc[move["dumBoat"] == 1]

move['log_dist_x_dist50'] = move['log_distance'] * move['dist50']

pred_prob = np.load("data/predprob.npy")

pred_dum = 1 * (pred_prob >= 0.35)
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
move = move.loc[move["dumBoat"] == 1]

move['log_distance2'] = move['log_distance'] ** 2
move['log_distance3'] = move['log_distance'] ** 3

move['log_acre_diff2'] = move['log_acre_diff'] ** 2
move['log_acre_diff3'] = move['log_acre_diff'] ** 3

move['log_acre_o2'] = move['log_acre_o'] ** 2

move['log_acre_d2'] = move['log_acre_d'] ** 2
move['log_acre_d3'] = move['log_acre_d'] ** 3

move['log_boat_diff2'] = move['log_boat_diff'] ** 2
move['log_boat_diff3'] = move['log_boat_diff'] ** 3

move['log_boat_o2'] = move['log_boat_o'] ** 2
move['log_boat_o3'] = move['log_boat_o'] ** 3

move['log_boat_d2'] = move['log_boat_d'] ** 2
move['log_boat_d3'] = move['log_boat_d'] ** 3

move['log_acre_diff_x_log_boat_diff'] = move['log_acre_diff'] * move['log_boat_diff']
move['log_acre_o_x_log_boat_o'] = move['log_acre_o'] * move['log_boat_o']
move['log_acre_d_x_log_boat_d'] = move['log_acre_d'] * move['log_boat_d']
move['log_distance_x_dist50'] = move['log_distance'] * move['dist50']

move['log_normBoatsExtra'] = np.log(move['normBoatsExtra'] + 1)

y_col = ['log_normBoatsExtra']

x_col = ['log_distance', 'log_distance2', 'log_distance3', 'self_loop', \
'log_acre_diff', 'log_acre_diff2', 'log_acre_diff3', 'log_acre_o', 'log_acre_o2', \
'log_acre_d', 'log_acre_d2', 'log_boat_diff', 'log_boat_diff2', 'log_boat_diff3', \
'log_boat_o', 'log_boat_o2', 'log_boat_o3', 'log_boat_d', 'log_boat_d2', 'log_boat_d3', \
'infest_o', 'infest_d', 'same_infest', 'insp_o', 'insp_d', 'same_insp', \
'log_ramp_acre_o', 'log_ramp_acre_d']


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
	n_estimators = 617,
	max_depth = 8,
	min_child_weight = 19,
	gamma = 0.3,
	eta = 0.3, 
	subsample = 0.9,
	colsample_bytree = 1,
	objective = 'reg:squarederror',
	booster = 'gbtree', 
	nthread = os.cpu_count() - 1,
	eval_metric = 'rmse')

aa = timeit.default_timer()
eval_set = [(test_set, test_Y)]
xgbmodel = xgb1.fit(train_set, train_Y, eval_set = eval_set, verbose = True, early_stopping_rounds = 5)
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
	n_estimators = 640,
	max_depth = 8,
	min_child_weight = 19,
	gamma = 0.3,
	eta = 0.3, 
	subsample = 0.9,
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
	tempDF['infest_o'] = np.repeat(att['infest'].iloc[ix], att.shape[0])
	tempDF['infest_d'] = att['infest']
	tmp = att['county'].iloc[ix]
	utm_x = att['utm_x'].iloc[ix]
	utm_y = att['utm_y'].iloc[ix]
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====1")
	tmp = np.sqrt((att['utm_x'] - utm_x) ** 2 + (att['utm_y'] - utm_y) ** 2)/1000
	tempDF['log_distance'] = np.log(tmp + 1)
	tempDF['dist50'] = 1 * (tmp < 50)
	tempDF['log_distance_x_dist50'] = tempDF['log_distance']*tempDF['dist50']
	tempDF['log_distance2'] = tempDF['log_distance'] ** 2
	tempDF['log_distance3'] = tempDF['log_distance'] ** 3
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====2")
	tmp = att['acre'].iloc[ix]
	tempDF['log_acre_diff'] = np.log(np.absolute(att['acre'] - tmp) + 1)
	tempDF['log_acre_diff2'] = tempDF['log_acre_diff'] ** 2
	tempDF['log_acre_diff3'] = tempDF['log_acre_diff'] ** 3
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====3")
	tempDF['log_acre_o'] = np.log(tmp + 1)
	tempDF['log_acre_o2'] = tempDF['log_acre_o'] ** 2 
	tempDF['log_acre_d'] = np.log(att['acre'] + 1)
	tempDF['log_acre_d2'] = tempDF['log_acre_d'] ** 2
	tempDF['log_acre_d3'] = tempDF['log_acre_d'] ** 3
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====4")
	tmp = att['boat'].iloc[ix]
	tempDF['log_boat_diff'] = np.log(np.absolute(att['boat'] - tmp) + 1)
	tempDF['log_boat_diff2'] = tempDF['log_boat_diff'] ** 2
	tempDF['log_boat_diff3'] = tempDF['log_boat_diff'] ** 3
	# print(timeit.default_timer() - aa)

	# aa = timeit.default_timer()
	# print("====5")
	tempDF['log_boat_o'] = np.log(tmp + 1)
	tempDF['log_boat_o2'] = tempDF['log_boat_o'] ** 2
	tempDF['log_boat_o3'] = tempDF['log_boat_o'] ** 3
	tempDF['log_boat_d'] = np.log(att['boat'] + 1)
	tempDF['log_boat_d2'] = tempDF['log_boat_d'] ** 2
	tempDF['log_boat_d3'] = tempDF['log_boat_d'] ** 3
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
	tempDF['log_ramp_acre_o'] = np.log(tmp1 / tmp2 * 1000 +1)
	tempDF['log_ramp_acre_d'] = np.log(att['ramp'] / att['acre'] * 1000 + 1)

	tempDF['self_loop'] = 0
	tempDF.iloc[ix]['self_loop'] = 1

	tempDF = tempDF[x_col]
	pred_boat[ix] = xgbmodel.predict(tempDF.to_numpy())


pred_boat = np.ceil(np.exp(pred_boat) - 1)
pred_boat[pred_boat < 1] = 1

aa = timeit.default_timer()
np.save("data/predboat", pred_boat)
print(timeit.default_timer() - aa)


aa = timeit.default_timer()
move = pd.read_csv('data/movement_for_stat.csv')
print(timeit.default_timer() - aa)

move = move.drop(['Unnamed: 0'], axis = 1)
move = move.loc[move["dumBoat"] == 1]

pred_Y = np.zeros(move.shape[0])
for x in range(pred_Y.shape[0]): 
	ix_x = move['id_origin'].iloc[x]
	ix_y = move['id_destination'].iloc[x]
	pred_Y[x] = pred_boat[ix_x, ix_y]

filename = 'boaters_prediction'
path = 'boater_gen_out/'

corr = np.round(np.corrcoef(pred_Y, move[['normBoatsExtra']].to_numpy().T)[0, 1], 3)

plt.clf()
fig = plt.figure(figsize=(6,4))
plt.scatter(move[['normBoatsExtra']].to_numpy().T, pred_Y, s = 10, 
	facecolors='none', edgecolors='cornflowerblue')
plt.ylabel('predicted boats')
plt.xlabel('boats')
plt.text(500, 10000, "corr = " + str(corr))
fig.tight_layout()
plt.savefig(path + filename + '(pred vs actual) post.png', format='png', dpi=500)





