import numpy as np
from scipy.special import comb
from itertools import product
import pandas as pd
import copy
import json
import timeit
import scipy.stats as stats
import os
import AISsim.UtilFunction as util
import ABCSMC.ABCSMC as abc
import ABCSMC.data as abcdata
import ABCSMC.genfiles as gfile
import ABCSMC.managefile as mfile 
import ABCSMC.SampleTasks as task
import matplotlib as mpl
print(mpl.rcParams['backend'])
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()
import corner
import seaborn as sns

myenv = "mylaptop"
abcdata.SetDirectory(myenv)

gen_t = 7

pars_name = abcdata.DrawParams().pars_name

with open('genout/gen' + str(gen_t) + '.txt') as json_file:  
	outcomes = json.load(json_file)

pars = np.array(outcomes['par'])
wgt = np.array(outcomes['weight'])
distance = np.array(outcomes['distance'])
sim_target = np.array(outcomes['sim_target'])

pars = pd.DataFrame(pars, 
	columns = pars_name)

samples = pars[['factor_ss', 'e_violate_zm', 'e_violate_ss', 'river_inf_zm', 'river_inf_ss', 'back_suit_zm', 'back_suit_ss']]
samples['wgt'] = wgt
samples = np.array(samples)
np.save("data/param_sample", samples)

pars["weight"] = wgt
pars["violate_ss"] = 0.18 * pars['factor_ss'] * pars['e_violate_ss']
pars["violate_zm"] = 0.0075 * pars['e_violate_zm']


fig = plt.figure(figsize=(18,10))

ax1 = fig.add_subplot(231)
temp0 = pars["violate_zm"]
temp = np.histogram(pars["violate_zm"], weights = pars['weight'], bins = 10)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])
xlabels = np.round(np.linspace(-0.5, np.round(np.max(tmp_x)*100000, 0), 6), 1)

ax1.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax1.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
ax1.set_title('violate_zm', fontsize =18)
ax1.set_xlabel('$10^{-5}$',fontsize=14)
ax1.set_ylabel('density',fontsize=14)
ax1.set_xticklabels(xlabels)
plt.xticks(rotation=0)
ax1.axvline(np.mean(temp0), color=sns.xkcd_rgb["midnight"], linewidth=2)
ax1.text(np.mean(temp0), 0.08, str(np.round(np.mean(temp0)*100000,2))+"x$10^{-5}$", fontsize=15)

ax2 = fig.add_subplot(232)
temp0 = pars["river_inf_zm"]
temp = np.histogram(pars["river_inf_zm"], weights = pars['weight'], bins = 10)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

ax2.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax2.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
ax2.set_title('river_inf_zm', fontsize =18)
ax2.set_xlabel('',fontsize=14)
ax2.set_ylabel('density',fontsize=14)
plt.xticks(rotation=0)
ax2.axvline(np.mean(temp0), color=sns.xkcd_rgb["midnight"], linewidth=2)
ax2.text(np.mean(temp0), 0.05, np.round(np.mean(temp0),2), fontsize=15)

ax3 = fig.add_subplot(233)
temp0 = pars["back_suit_zm"]
temp = np.histogram(pars["back_suit_zm"], weights = pars['weight'], bins = 10)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

ax3.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax3.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
ax3.set_title('back_suit_zm', fontsize =18)
ax3.set_xlabel('',fontsize=14)
ax3.set_ylabel('density',fontsize=14)
plt.xticks(rotation=0)
ax3.axvline(np.mean(temp0), color=sns.xkcd_rgb["midnight"], linewidth=2)
ax3.text(np.mean(temp0), 0.05, np.round(np.mean(temp0),2), fontsize=15)

ax4 = fig.add_subplot(234)
temp0 = pars["violate_ss"]
temp = np.histogram(pars["violate_ss"], weights = pars['weight'], bins = 10)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])
xlabels = np.round(np.linspace(-2, np.ceil(np.max(tmp_x)*1000), 10), 0)

ax4.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax4.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
ax4.set_title('violate_ss', fontsize =18)
ax4.set_xlabel('$10^{-3}$',fontsize=14)
ax4.set_ylabel('density',fontsize=14)
ax4.set_xticklabels(xlabels)
plt.xticks(rotation=0)
ax4.axvline(np.mean(temp0), color=sns.xkcd_rgb["midnight"], linewidth=2)
ax4.text(np.mean(temp0), 0.05, str(np.round(np.mean(temp0)*1000,2))+"x$10^{-3}$", fontsize=15)

ax5 = fig.add_subplot(235)
temp0 = pars["river_inf_ss"]
temp = np.histogram(pars['river_inf_ss'], weights = pars['weight'], bins = 10)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

ax5.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax5.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
ax5.set_title('river_inf_ss', fontsize =18)
ax5.set_xlabel('',fontsize=14)
ax5.set_ylabel('density',fontsize=14)
plt.xticks(rotation=0)
ax5.axvline(np.mean(temp0), color=sns.xkcd_rgb["midnight"], linewidth=2)
ax5.text(np.mean(temp0), 0.06, np.round(np.mean(temp0),2), fontsize=15)

ax6 = fig.add_subplot(236)
temp0 = pars["back_suit_ss"]
temp = np.histogram(pars['back_suit_ss'], weights = pars['weight'], bins = 10)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

ax6.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax6.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
ax6.set_title('back_suit_ss', fontsize =18)
ax6.set_xlabel('',fontsize=14)
ax6.set_ylabel('density',fontsize=14)
plt.xticks(rotation=0)
ax6.axvline(np.mean(temp0), color=sns.xkcd_rgb["midnight"], linewidth=2)
ax6.text(np.mean(temp0), 0.05, np.round(np.mean(temp0),2), fontsize=15)

plt.tight_layout()
plt.savefig("results/figure (paper)(gen" + str(gen_t) + ").eps", format='eps', dpi=1000)
plt.close()


samples = pars[['violate_ss', 'river_inf_ss', 'violate_zm', 'river_inf_zm', 
	'back_suit_zm', 'back_suit_ss']].to_numpy()

pars[['violate_ss', 'river_inf_ss', 'violate_zm', 'river_inf_zm', 
	'back_suit_zm', 'back_suit_ss', 'weight']].to_csv('results/pars_for_r.csv', index = False)

# # samples = samples[0]
# tmp = corner.corner(samples[:,:], labels=['violate_ss', 'river_inf_ss', 'violate_zm', 'river_inf_zm', 
# 	'back_suit_zm', 'back_suit_ss'])
# plt.savefig('results/corner_plot (gen' + str(gen_t) + ').eps', format='eps', dpi=1200)


def mysum(x): 
	return np.mean(x, axis = 0).tolist(), np.percentile(x, q = 10, axis = 0).tolist(), \
		np.percentile(x, q = 90, axis = 0).tolist(), np.std(x, axis = 0).tolist()

lake_id = abcdata.LoadData(myenv).getLakeAttributes()[0]
target = abcdata.Target(lake_id)
target_all = target.target_all

dis_ix = np.argsort(distance)
# sim_target = sim_target[dis_ix[:10]]

# calibration period summary
pre_summary = {'zm': dict(zip(['mean', 'lb','ub','sd'], mysum(sim_target[:, 0:7]))), 
	'ss': dict(zip(['mean', 'lb','ub','sd'], mysum(sim_target[:, 7:])))}

zm_count = dict(zip([x for x in range(2013, 2020)], target_all[:7].tolist()))
ss_count = dict(zip([x for x in range(2017, 2020)], target_all[7:].tolist()))


fig = plt.figure(figsize=(12,5))
# ax1 = fig.add_subplot(111)
ax1 = fig.add_subplot(121)
yr = np.array([x for x in range(2013, 2020)])
temp_mean =  np.array(pre_summary['zm']['mean'])
temp_lower = np.array(pre_summary['zm']['lb'])
temp_upper = np.array(pre_summary['zm']['ub'])
target = np.array([val for key, val in zm_count.items()])
temp_line_cal = []
for x in range(0, 7): 
	temp_line_cal += [(yr[x], yr[x]), (temp_lower[x], temp_upper[x])]
ax1.plot(*temp_line_cal, color = sns.xkcd_rgb["midnight"])
# ax1.plot(*temp_line_pred, color = sns.xkcd_rgb["grey"])
ax1.scatter(yr, temp_mean, marker = 'o', facecolors='white', edgecolors=sns.xkcd_rgb["grey"], s=50, linewidth=2)
ax1.scatter(yr, temp_lower, marker = '_', color = sns.xkcd_rgb["grey"])
ax1.scatter(yr, temp_upper, marker = '_', color = sns.xkcd_rgb["grey"])
ax1.scatter(yr[0:7], temp_mean[0:7], marker = 'o', facecolors='white', edgecolors=sns.xkcd_rgb["midnight"], s=50, linewidth=2)
ax1.scatter(yr[0:7], temp_lower[0:7], marker = '_', color = sns.xkcd_rgb["midnight"])
ax1.scatter(yr[0:7], temp_upper[0:7], marker = '_', color = sns.xkcd_rgb["midnight"])
ax1.scatter(yr[0:7], target, marker = '*', facecolors=sns.xkcd_rgb["red orange"], edgecolors = sns.xkcd_rgb["red orange"], s=80, linewidth=3)
# for x in range(len(temp_line)): 
ax1.set_title('(A) Zebra mussels', fontsize =18)
ax1.set_xlabel('year',fontsize=14)
ax1.set_ylabel('# of lakes',fontsize=14)

# fig = plt.figure(figsize=(12,5))
# ax2 = fig.add_subplot(111)
ax2 = fig.add_subplot(122)
yr = np.array([x for x in range(2017, 2020)])
temp_mean =  np.array(pre_summary['ss']['mean'])
temp_lower = np.array(pre_summary['ss']['lb'])
temp_upper = np.array(pre_summary['ss']['ub'])
target = np.array([val for key, val in ss_count.items()])
temp_line_cal = []
for x in range(0, 3): 
	temp_line_cal += [(yr[x], yr[x]), (temp_lower[x], temp_upper[x])]
ax2.plot(*temp_line_cal, color = sns.xkcd_rgb["midnight"])
# ax2.plot(*temp_line_pred, color = sns.xkcd_rgb["grey"])
ax2.scatter(yr, temp_mean, marker = 'o', facecolors='white', edgecolors=sns.xkcd_rgb["grey"], s=50, linewidth=2)
ax2.scatter(yr, temp_lower, marker = '_', color = sns.xkcd_rgb["grey"])
ax2.scatter(yr, temp_upper, marker = '_', color = sns.xkcd_rgb["grey"])
ax2.scatter(yr[0:3], temp_mean[0:3], marker = 'o', facecolors='white', edgecolors=sns.xkcd_rgb["midnight"], s=50, linewidth=2)
ax2.scatter(yr[0:3], temp_lower[0:3], marker = '_', color = sns.xkcd_rgb["midnight"])
ax2.scatter(yr[0:3], temp_upper[0:3], marker = '_', color = sns.xkcd_rgb["midnight"])
ax2.scatter(yr[0:3], target, marker = '*', facecolors=sns.xkcd_rgb["red orange"], edgecolors = sns.xkcd_rgb["red orange"], s=80, linewidth=3)
# for x in range(len(temp_line)): ax2.set_title('(B) Proportion of partnerships in which\n>= 1 person discloses their HIV status', fontsize =18)
ax2.set_title('(B) Starry stonewort', fontsize =18)
ax2.set_xlabel('year',fontsize=14)
ax2.set_ylabel('# of lakes',fontsize=14)

plt.tight_layout()
plt.savefig('results/summary plot annual_no prediction (gen' + str(gen_t) + ').eps', format='eps', dpi=1000)
plt.close()



