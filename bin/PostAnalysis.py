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

import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()
import corner
import seaborn as sns


with open('data/mcmc_results_1.txt') as json_file:  
    outcomes1 = json.load(json_file)

with open('data/mcmc_results_2.txt') as json_file:  
    outcomes2 = json.load(json_file)

with open('data/mcmc_results_3.txt') as json_file:  
    outcomes3 = json.load(json_file)

with open('data/mcmc_results_4.txt') as json_file:  
    outcomes4 = json.load(json_file)

with open('data/mcmc_results_5.txt') as json_file:  
    outcomes5 = json.load(json_file)


keys = [k for k,v in outcomes4.items()]

for k in keys[:-1]: 
	fig = plt.figure(figsize=(12,5))
	# ax1 = fig.add_subplot(111)
	ax1 = fig.add_subplot(121)
	ax1.plot(np.array(outcomes1[k]), color = sns.xkcd_rgb["grey green"])
	ax1.plot(np.array(outcomes2[k]), color = sns.xkcd_rgb["denim"])
	ax1.plot(np.array(outcomes3[k]), color = sns.xkcd_rgb["lightish blue"])
	ax1.plot(np.array(outcomes4[k]), color = sns.xkcd_rgb["reddish"])
	ax1.plot(np.array(outcomes5[k]), color = sns.xkcd_rgb["pistachio"])
	ax1.set_title(k, fontsize =18)
	ax1.set_xlabel('iterations',fontsize=14)
	ax1.set_ylabel(k,fontsize=14)

	ax2 = fig.add_subplot(122)
	temp0 = np.array(outcomes1[k] + outcomes2[k] + outcomes3[k] + outcomes4[k] + outcomes5[k])
	# temp0 = np.array(outcomes1[k] + outcomes2[k] + outcomes3[k])
	temp = np.histogram(temp0, bins = 25)
	tmp_y = temp[0]/np.sum(temp[0])
	tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

	ax2.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["apple green"], linewidth = 1)
	ix = tmp_x
	iy = tmp_y
	ax2.fill_between(ix,iy, color = sns.xkcd_rgb["apple green"])
	ax2.set_title(k, fontsize =18)
	ax2.set_xlabel('value',fontsize=14)
	ax2.set_ylabel('density',fontsize=14)
	plt.xticks(rotation=30)
	ax2.axvline(np.mean(temp0), color=sns.xkcd_rgb["light red"], linewidth=2)
	ax2.text(np.mean(temp0), 0.04, np.round(np.mean(temp0),5), fontsize=12)

	plt.tight_layout()
	plt.savefig("results/"+k+'(new2).eps', format='eps', dpi=1000)




fig = plt.figure(figsize=(12,5))
tmp1 = 0.18*np.array(outcomes1['factor_ss'])*np.array(outcomes1['e_violate_ss'])
tmp2 = 0.18*np.array(outcomes2['factor_ss'])*np.array(outcomes2['e_violate_ss'])
tmp3 = 0.18*np.array(outcomes3['factor_ss'])*np.array(outcomes3['e_violate_ss'])
tmp4 = 0.18*np.array(outcomes4['factor_ss'])*np.array(outcomes4['e_violate_ss'])
tmp5 = 0.18*np.array(outcomes5['factor_ss'])*np.array(outcomes5['e_violate_ss'])

ax1 = fig.add_subplot(121)
ax1.plot(tmp1, color = sns.xkcd_rgb["grey green"])
ax1.plot(tmp2, color = sns.xkcd_rgb["denim"])
ax1.plot(tmp3, color = sns.xkcd_rgb["lightish blue"])
ax1.plot(tmp4, color = sns.xkcd_rgb["reddish"])
ax1.plot(tmp5, color = sns.xkcd_rgb["pistachio"])
ax1.set_title("violate_ss", fontsize =18)
ax1.set_xlabel('iterations',fontsize=14)
ax1.set_ylabel("violate_ss",fontsize=14)

ax2 = fig.add_subplot(122)
temp0 = np.hstack([tmp1, tmp2, tmp3, tmp4, tmp5])
temp = np.histogram(temp0, bins = 50)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

ax2.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["apple green"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax2.fill_between(ix,iy, color = sns.xkcd_rgb["apple green"])
ax2.set_title('violate_ss', fontsize =18)
ax2.set_xlabel('value',fontsize=14)
ax2.set_ylabel('density',fontsize=14)
plt.xticks(rotation=30)
ax2.axvline(np.mean(temp0), color=sns.xkcd_rgb["light red"], linewidth=2)
ax2.text(np.mean(temp0), 0.1, np.round(np.mean(temp0),5), fontsize=12)

plt.tight_layout()
plt.savefig("results/violate_ss(new).eps", format='eps', dpi=1000)


fig = plt.figure(figsize=(12,5))
tmp1 = 0.0075*np.array(outcomes1['e_violate_zm'])
tmp2 = 0.0075*np.array(outcomes2['e_violate_zm'])
tmp3 = 0.0075*np.array(outcomes3['e_violate_zm'])
tmp4 = 0.0075*np.array(outcomes4['e_violate_zm'])
tmp5 = 0.0075*np.array(outcomes5['e_violate_zm'])

ax1 = fig.add_subplot(121)
ax1.plot(tmp1, color = sns.xkcd_rgb["grey green"])
ax1.plot(tmp2, color = sns.xkcd_rgb["denim"])
ax1.plot(tmp3, color = sns.xkcd_rgb["lightish blue"])
ax1.plot(tmp4, color = sns.xkcd_rgb["reddish"])
ax1.plot(tmp5, color = sns.xkcd_rgb["pistachio"])
ax1.set_title("violate_zm", fontsize =18)
ax1.set_xlabel('iterations',fontsize=14)
ax1.set_ylabel("violate_zm",fontsize=14)

ax2 = fig.add_subplot(122)
# temp0 = np.hstack([tmp1, tmp2, tmp3, tmp4, tmp5])
temp0 = np.hstack([tmp3, tmp4, tmp5])
temp = np.histogram(temp0, bins = 25)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

ax2.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["apple green"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax2.fill_between(ix,iy, color = sns.xkcd_rgb["apple green"])
ax2.set_title('violate_zm', fontsize =18)
ax2.set_xlabel('value',fontsize=14)
ax2.set_ylabel('density',fontsize=14)
plt.xticks(rotation=30)
ax2.axvline(np.mean(temp0), color=sns.xkcd_rgb["light red"], linewidth=2)
ax2.text(np.mean(temp0), 0.05, np.round(np.mean(temp0),5), fontsize=12)

plt.tight_layout()
plt.savefig("results/violate_zm(new).eps", format='eps', dpi=1000)


## figures on paper

fig = plt.figure(figsize=(18,10))

ax1 = fig.add_subplot(231)
tmp1 = 0.0075*np.array(outcomes1['e_violate_zm'])
tmp2 = 0.0075*np.array(outcomes2['e_violate_zm'])
tmp3 = 0.0075*np.array(outcomes3['e_violate_zm'])
tmp4 = 0.0075*np.array(outcomes4['e_violate_zm'])
tmp5 = 0.0075*np.array(outcomes5['e_violate_zm'])
temp0 = np.hstack([tmp1, tmp2, tmp3, tmp4, tmp5])
temp = np.histogram(temp0, bins = 30)
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
tmp1 = np.array(outcomes1['river_inf_zm'])
tmp2 = np.array(outcomes2['river_inf_zm'])
tmp3 = np.array(outcomes3['river_inf_zm'])
tmp4 = np.array(outcomes4['river_inf_zm'])
tmp5 = np.array(outcomes5['river_inf_zm'])
temp0 = np.hstack([tmp1, tmp2, tmp3, tmp4, tmp5])
temp = np.histogram(temp0, bins = 20)
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
tmp1 = np.array(outcomes1['back_suit_zm'])
tmp2 = np.array(outcomes2['back_suit_zm'])
tmp3 = np.array(outcomes3['back_suit_zm'])
tmp4 = np.array(outcomes4['back_suit_zm'])
tmp5 = np.array(outcomes5['back_suit_zm'])
temp0 = np.hstack([tmp1, tmp2, tmp3, tmp4, tmp5])
temp = np.histogram(temp0, bins = 20)
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
tmp1 = 0.18*np.array(outcomes1['factor_ss'])*np.array(outcomes1['e_violate_ss'])
tmp2 = 0.18*np.array(outcomes2['factor_ss'])*np.array(outcomes2['e_violate_ss'])
tmp3 = 0.18*np.array(outcomes3['factor_ss'])*np.array(outcomes3['e_violate_ss'])
tmp4 = 0.18*np.array(outcomes4['factor_ss'])*np.array(outcomes4['e_violate_ss'])
tmp5 = 0.18*np.array(outcomes5['factor_ss'])*np.array(outcomes5['e_violate_ss'])
temp0 = np.hstack([tmp1, tmp2, tmp3, tmp4, tmp5])
temp = np.histogram(temp0, bins = 20)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])
xlabels = np.round(np.linspace(-2, np.ceil(np.max(tmp_x)*100000), 10), 0)

ax4.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax4.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
ax4.set_title('violate_ss', fontsize =18)
ax4.set_xlabel('$10^{-5}$',fontsize=14)
ax4.set_ylabel('density',fontsize=14)
ax4.set_xticklabels(xlabels)
plt.xticks(rotation=0)
ax4.axvline(np.mean(temp0), color=sns.xkcd_rgb["midnight"], linewidth=2)
ax4.text(np.mean(temp0), 0.22, str(np.round(np.mean(temp0)*100000,2))+"x$10^{-5}$", fontsize=15)

ax5 = fig.add_subplot(235)
tmp1 = np.array(outcomes1['river_inf_ss'])
tmp2 = np.array(outcomes2['river_inf_ss'])
tmp3 = np.array(outcomes3['river_inf_ss'])
tmp4 = np.array(outcomes4['river_inf_ss'])
tmp5 = np.array(outcomes5['river_inf_ss'])
temp0 = np.hstack([tmp1, tmp2, tmp3, tmp4, tmp5])
temp = np.histogram(temp0, bins = 20)
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
tmp1 = np.array(outcomes1['back_suit_ss'])
tmp2 = np.array(outcomes2['back_suit_ss'])
tmp3 = np.array(outcomes3['back_suit_ss'])
tmp4 = np.array(outcomes4['back_suit_ss'])
tmp5 = np.array(outcomes5['back_suit_ss'])
temp0 = np.hstack([tmp1, tmp2, tmp3, tmp4, tmp5])
temp = np.histogram(temp0, bins = 20)
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
plt.savefig("results/figure (paper)(new).eps", format='eps', dpi=1000)



violate_ss = 0.18*np.array(outcomes1['factor_ss']+outcomes2['factor_ss']+outcomes3['factor_ss']+outcomes4['factor_ss']+outcomes5['factor_ss'])* \
np.array(outcomes1['e_violate_ss']+outcomes2['e_violate_ss']+outcomes3['e_violate_ss']+outcomes4['e_violate_ss']+outcomes5['e_violate_ss'])

samples = np.array([outcomes1['factor_ss']+outcomes2['factor_ss']+outcomes3['factor_ss']+outcomes4['factor_ss']+outcomes5['factor_ss'], 
	outcomes1['e_violate_zm']+outcomes2['e_violate_zm']+outcomes3['e_violate_zm']+outcomes4['e_violate_zm']+outcomes5['e_violate_zm'], 
	outcomes1['e_violate_ss']+outcomes2['e_violate_ss']+outcomes3['e_violate_ss']+outcomes4['e_violate_ss']+outcomes5['e_violate_ss'], \
	outcomes1['river_inf_zm']+outcomes2['river_inf_zm']+outcomes3['river_inf_zm']+outcomes4['river_inf_zm']+outcomes5['river_inf_zm'], 
	outcomes1['river_inf_ss']+outcomes2['river_inf_ss']+outcomes3['river_inf_ss']+outcomes4['river_inf_ss']+outcomes5['river_inf_ss'], 
	outcomes1['back_suit_zm']+outcomes2['back_suit_zm']+outcomes3['back_suit_zm']+outcomes4['back_suit_zm']+outcomes5['back_suit_zm'], 
	outcomes1['back_suit_ss']+outcomes2['back_suit_ss']+outcomes3['back_suit_ss']+outcomes4['back_suit_ss']+outcomes5['back_suit_ss']]).T

tmp = corner.corner(samples[:,:], labels=['factor_ss', 'e_violate_zm', 'e_violate_ss', 'river_inf_zm', 'river_inf_ss', 
	'back_suit_zm', 'back_suit_ss'])
plt.savefig('results/corner_plot.png', format='png', dpi=1200)


violate_ss = 0.18*np.array(outcomes1['factor_ss']+outcomes2['factor_ss']+outcomes3['factor_ss']+outcomes4['factor_ss']+outcomes5['factor_ss'])* \
np.array(outcomes1['e_violate_ss']+outcomes2['e_violate_ss']+outcomes3['e_violate_ss']+outcomes4['e_violate_ss']+outcomes5['e_violate_ss'])

violate_zm = 0.0075*np.array(outcomes1['e_violate_zm']+outcomes2['e_violate_zm']+outcomes3['e_violate_zm']+outcomes4['e_violate_zm']+outcomes5['e_violate_zm'])

samples = np.array([violate_ss, \
	outcomes1['river_inf_ss']+outcomes2['river_inf_ss']+outcomes3['river_inf_ss']+outcomes4['river_inf_ss']+outcomes5['river_inf_ss'], \
	violate_zm, \
	outcomes1['river_inf_zm']+outcomes2['river_inf_zm']+outcomes3['river_inf_zm']+outcomes4['river_inf_zm']+outcomes5['river_inf_zm'], 
	outcomes1['back_suit_zm']+outcomes2['back_suit_zm']+outcomes3['back_suit_zm']+outcomes4['back_suit_zm']+outcomes5['back_suit_zm'], 
	outcomes1['back_suit_ss']+outcomes2['back_suit_ss']+outcomes3['back_suit_ss']+outcomes4['back_suit_ss']+outcomes5['back_suit_ss']]).T

# samples = samples[0]
tmp = corner.corner(samples[:,:], labels=['violate_ss', 'river_inf_ss', 'violate_zm', 'river_inf_zm', 
	'back_suit_zm', 'back_suit_ss'])
plt.savefig('results/corner_plot (simplified).eps', format='eps', dpi=1200)


print(np.corrcoef(violate_zm, np.array(outcomes1['river_inf_zm']+outcomes2['river_inf_zm']+outcomes3['river_inf_zm']+outcomes4['river_inf_zm']+outcomes5['river_inf_zm'])))
print(np.corrcoef(violate_ss, np.array(outcomes1['river_inf_ss']+outcomes2['river_inf_ss']+outcomes3['river_inf_ss']+outcomes4['river_inf_ss']+outcomes5['river_inf_ss'])))
print(np.corrcoef(violate_zm, np.array(outcomes1['back_suit_zm']+outcomes2['back_suit_zm']+outcomes3['back_suit_zm']+outcomes4['back_suit_zm']+outcomes5['back_suit_zm'])))
print(np.corrcoef(violate_ss, np.array(outcomes1['back_suit_ss']+outcomes2['back_suit_ss']+outcomes3['back_suit_ss']+outcomes4['back_suit_ss']+outcomes5['back_suit_ss'])))

# summary statistics
samples = np.array([outcomes1['factor_ss']+outcomes2['factor_ss']+outcomes3['factor_ss']+outcomes4['factor_ss']+outcomes5['factor_ss'], 
	outcomes1['e_violate_zm']+outcomes2['e_violate_zm']+outcomes3['e_violate_zm']+outcomes4['e_violate_zm']+outcomes5['e_violate_zm'], 
	outcomes1['e_violate_ss']+outcomes2['e_violate_ss']+outcomes3['e_violate_ss']+outcomes4['e_violate_ss']+outcomes5['e_violate_ss'], 
	outcomes1['river_inf_zm']+outcomes2['river_inf_zm']+outcomes3['river_inf_zm']+outcomes4['river_inf_zm']+outcomes5['river_inf_zm'], 
	outcomes1['river_inf_ss']+outcomes2['river_inf_ss']+outcomes3['river_inf_ss']+outcomes4['river_inf_ss']+outcomes5['river_inf_ss'], 
	outcomes1['back_suit_zm']+outcomes2['back_suit_zm']+outcomes3['back_suit_zm']+outcomes4['back_suit_zm']+outcomes5['back_suit_zm'], 
	outcomes1['back_suit_ss']+outcomes2['back_suit_ss']+outcomes3['back_suit_ss']+outcomes4['back_suit_ss']+outcomes5['back_suit_ss'], 
	outcomes1['target0']+outcomes2['target0']+outcomes3['target0']+outcomes4['target0']+outcomes5['target0'], 
	outcomes1['target1']+outcomes2['target1']+outcomes3['target1']+outcomes4['target1']+outcomes5['target1'], 
	outcomes1['target2']+outcomes2['target2']+outcomes3['target2']+outcomes4['target2']+outcomes5['target2'], 
	outcomes1['target3']+outcomes2['target3']+outcomes3['target3']+outcomes4['target3']+outcomes5['target3'], 
	outcomes1['target4']+outcomes2['target4']+outcomes3['target4']+outcomes4['target4']+outcomes5['target4'], 
	outcomes1['target5']+outcomes2['target5']+outcomes3['target5']+outcomes4['target5']+outcomes5['target5'], 
	outcomes1['target6']+outcomes2['target6']+outcomes3['target6']+outcomes4['target6']+outcomes5['target6'], 
	outcomes1['target7']+outcomes2['target7']+outcomes3['target7']+outcomes4['target7']+outcomes5['target7']]).T


samples = pd.DataFrame(samples)
samples.columns = ['factor_ss', 'e_violate_zm', 'e_violate_ss', 
'river_inf_zm', 'river_inf_ss', 'back_suit_zm', 'back_suit_ss', 
'zm_yr2013', 'zm_yr2014', 'zm_yr2015', 'zm_yr2016', 'zm_yr2017', 'zm_yr2018', 
'ss_yr2017', 'ss_yr2018']
samples.describe().T.to_csv("results/parameter summary.csv")


samples = samples[['factor_ss', 'e_violate_zm', 'e_violate_ss', 'river_inf_zm', 'river_inf_ss', 'back_suit_zm', 'back_suit_ss']]
samples = np.array(samples)
np.save("data/param_sample", samples)

samples = np.array([outcomes1['factor_ss'][60000:]+outcomes2['factor_ss'][60000:]+outcomes3['factor_ss'][60000:]+outcomes4['factor_ss'][60000:]+outcomes5['factor_ss'][60000:], \
	outcomes1['e_violate_zm'][60000:]+outcomes2['e_violate_zm'][60000:]+outcomes3['e_violate_zm'][60000:]+outcomes4['e_violate_zm'][60000:]+outcomes5['e_violate_zm'][60000:], \
	outcomes1['e_violate_ss'][60000:]+outcomes2['e_violate_ss'][60000:]+outcomes3['e_violate_ss'][60000:]+outcomes4['e_violate_ss'][60000:]+outcomes5['e_violate_ss'][60000:], \
	outcomes1['river_inf_zm'][60000:]+outcomes2['river_inf_zm'][60000:]+outcomes3['river_inf_zm'][60000:]+outcomes4['river_inf_zm'][60000:]+outcomes5['river_inf_zm'][60000:], \
	outcomes1['river_inf_ss'][60000:]+outcomes2['river_inf_ss'][60000:]+outcomes3['river_inf_ss'][60000:]+outcomes4['river_inf_ss'][60000:]+outcomes5['river_inf_ss'][60000:], \
	outcomes1['back_suit_zm'][60000:]+outcomes2['back_suit_zm'][60000:]+outcomes3['back_suit_zm'][60000:]+outcomes4['back_suit_zm'][60000:]+outcomes5['back_suit_zm'][60000:], \
	outcomes1['back_suit_ss'][60000:]+outcomes2['back_suit_ss'][60000:]+outcomes3['back_suit_ss'][60000:]+outcomes4['back_suit_ss'][60000:]+outcomes5['back_suit_ss'][60000:]]).T

print(np.cov(samples.T))



'''
Simulation results
'''

import numpy as np
import pandas as pd
import copy
import json
import timeit
import os
import UtilFunction as util
os.chdir("/Users/szu-yukao/Documents/FishProject/virsim")
cwd = os.getcwd()
print(cwd)

import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()
import corner
import seaborn as sns

att = pd.read_csv('data/lake_attribute.csv')
att['boat'].fillna(att['boat'].mean(), inplace=True)
att['infest'].fillna(0, inplace=True)
att['inspect'].fillna(0, inplace=True)
att['zm_suit'].fillna(0, inplace=True)
att['ss_suit'].fillna(0, inplace=True)
print(list(att.columns.values))

zm_dow = pd.read_csv('data/zm_dow.csv')
ss_dow = pd.read_csv('data/ss_dow.csv')

att = pd.merge(att, zm_dow, how = 'left', left_on = 'dow', right_on = 'dow')
att = pd.merge(att, ss_dow, how = 'left', left_on = 'dow', right_on = 'dow')


# creating data vectors 
lake_id = copy.deepcopy(att['id'].values)

# pre-management scenarios 
# zebra mussels

temp = []
for i in range(1, 101): 
	with open('results/pre_res_zm_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))

pre_res_zm = [val for sublist in temp for val in sublist]

risk_counts = np.zeros(lake_id.shape[0])
risk_boat_counts = np.zeros(lake_id.shape[0])
risk_river_counts = np.zeros(lake_id.shape[0])

for i in range(len(pre_res_zm)): 
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

output = np.column_stack((lake_id, risk_counts/10000, risk_boat_counts/risk_counts, risk_river_counts/risk_counts))
output[np.isnan(output)] = 0
output = pd.DataFrame(output)
output.columns = ['id', 'pre_p_infest', 'pre_p_boat', 'pre_p_river']
output['id'].astype(int)

pre_risk_zm = pd.merge(att[['id','dow','lake_name', 'county.name', 'utm_x', 'utm_y', 'acre', 'infest.zm', 'zm_infest2018']], output, how = 'left', 
	left_on = 'id', right_on = 'id')


# starry stonewort
temp = []
for i in range(1, 101): 
	with open('results/pre_res_ss_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))

pre_res_ss = [val for sublist in temp for val in sublist]

risk_counts = np.zeros(lake_id.shape[0])
risk_boat_counts = np.zeros(lake_id.shape[0])
risk_river_counts = np.zeros(lake_id.shape[0])

for i in range(len(pre_res_ss)): 
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

output = np.column_stack((lake_id, risk_counts/10000, risk_boat_counts/risk_counts, risk_river_counts/risk_counts))
output[np.isnan(output)] = 0
output = pd.DataFrame(output)
output.columns = ['id', 'pre_p_infest', 'pre_p_boat', 'pre_p_river']
output['id'].astype(int)

pre_risk_ss = pd.merge(att[['id','dow','lake_name', 'county.name', 'utm_x', 'utm_y', 'acre', 'infest.ss', 'ss_infest2018']], output, how = 'left', 
	left_on = 'id', right_on = 'id')


# risk of zm for each lake 
temp = []
for i in range(1, 101): 
	with open('results/scenarios_res_zm_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))

scenarios_res_zm = [val for sublist in temp for val in sublist]

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

	output = np.column_stack((lake_id, risk_counts/10000, risk_boat_counts/risk_counts, \
		risk_river_counts/risk_counts, risk_boat_counts/10000, risk_river_counts/10000))
	output[np.isnan(output)] = 0
	output = pd.DataFrame(output)
	output.columns = ['id', s+'_p_infest', s+'_p_boat', s+'_p_river', s+'_r_boat', s+'_r_river']
	output['id'].astype(int)

	lake_risk_dict.update({s: output})

lake_risk_zm = pd.merge(pre_risk_zm, lake_risk_dict['S'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_zm = pd.merge(lake_risk_zm, lake_risk_dict['E'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_zm = pd.merge(lake_risk_zm, lake_risk_dict['P'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_zm = pd.merge(lake_risk_zm, lake_risk_dict['D'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_zm = pd.merge(lake_risk_zm, lake_risk_dict['T'], how = 'left', 
	left_on = 'id', right_on = 'id')

lake_risk_zm.to_csv('results/risk table (zm)(new).csv')


# risk of ss for each lake 
temp = []
for i in range(1, 101): 
	with open('results/scenarios_res_ss_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))

scenarios_res_ss = [val for sublist in temp for val in sublist]

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

	output = np.column_stack((lake_id, risk_counts/10000, risk_boat_counts/risk_counts, \
		risk_river_counts/risk_counts, risk_boat_counts/10000, risk_river_counts/10000))
	output[np.isnan(output)] = 0
	output = pd.DataFrame(output)
	output.columns = ['id', s+'_p_infest', s+'_p_boat', s+'_p_river', s+'_r_boat', s+'_r_river']
	output['id'].astype(int)

	lake_risk_dict.update({s: output})

lake_risk_ss = pd.merge(pre_risk_ss, lake_risk_dict['S'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_ss = pd.merge(lake_risk_ss, lake_risk_dict['E'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_ss = pd.merge(lake_risk_ss, lake_risk_dict['P'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_ss = pd.merge(lake_risk_ss, lake_risk_dict['D'], how = 'left', 
	left_on = 'id', right_on = 'id')
lake_risk_ss = pd.merge(lake_risk_ss, lake_risk_dict['T'], how = 'left', 
	left_on = 'id', right_on = 'id')

lake_risk_ss.to_csv('results/risk table (ss)(new).csv')

# management scenario effect
# zebra mussels
def mysum(x): 
	return np.mean(x, axis = 0).tolist(), np.percentile(x, q = 10, axis = 0).tolist(), \
	np.percentile(x, q = 90, axis = 0).tolist(), np.std(x, axis = 0).tolist()
	

temp = []
for i in range(1, 101): 
	with open('results/scenarios_ann_zm_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))

scenarios_ann_zm = [val for sublist in temp for val in sublist]

scenario_zm = dict()
summary_zm = dict()
for s in ['S', 'E', 'P', 'D', 'T']: 
	temp = np.zeros((len(scenarios_ann_zm), 7))
	for i in range(temp.shape[0]): 
		temp[i] = scenarios_ann_zm[i][s]
	scenario_zm.update({s: temp.tolist()})
	summary_zm.update({s: dict(zip(['mean', 'lb','ub','sd'], mysum(scenario_zm[s])))})

with open('results/effect summary stats (zm)(new).txt', 'w') as fout:
    json.dump(summary_zm, fout)

with open('results/scenario effect (zm)(new).txt', 'w') as fout:
    json.dump(scenario_zm, fout)

# starry stonewort
temp = []
for i in range(1, 101): 
	with open('results/scenarios_ann_ss_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))

scenarios_ann_ss = [val for sublist in temp for val in sublist]

scenario_ss = dict()
summary_ss = dict()

for s in ['S', 'E', 'P', 'D', 'T']: 
	temp = np.zeros((len(scenarios_ann_ss), 7))
	for i in range(temp.shape[0]): 
		temp[i] = scenarios_ann_ss[i][s]
	scenario_ss.update({s: temp.tolist()})
	summary_ss.update({s: dict(zip(['mean', 'lb','ub','sd'], mysum(scenario_ss[s])))})

with open('results/effect summary stats (ss)(new).txt', 'w') as fout:
    json.dump(summary_ss, fout)

with open('results/scenario effect (ss)(new).txt', 'w') as fout:
    json.dump(scenario_ss, fout)


# calibration period summary
temp = []
for i in range(1, 101): 
	with open('results/pre_ann_out_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))
pre_ann = [val for sublist in temp for val in sublist]
pre_ann = np.array(pre_ann)
pre_summary = {'zm': dict(zip(['mean', 'lb','ub','sd'], mysum(pre_ann[:, 0:6]))), 
'ss': dict(zip(['mean', 'lb','ub','sd'], mysum(pre_ann[:, 6:])))}


zm_count = {2012: 53,  2013: 68,  2014: 84,  2015: 99,  2016: 132,  2017: 167, 2018: 196}
ss_count = {2016: 8,  2017: 10, 2018: 13}


fig = plt.figure(figsize=(12,5))
# ax1 = fig.add_subplot(111)
ax1 = fig.add_subplot(121)
yr = np.array([x for x in range(2013, 2026)])
temp_mean =  np.array(pre_summary['zm']['mean'] + summary_zm['S']['mean'])
temp_lower = np.array(pre_summary['zm']['lb'] + summary_zm['S']['lb'])
temp_upper = np.array(pre_summary['zm']['ub'] + summary_zm['S']['ub'])
target = np.array([val for key, val in zm_count.items()])[1:]
temp_line_cal = []
for x in range(0, 6): 
	temp_line_cal += [(yr[x], yr[x]), (temp_lower[x], temp_upper[x])]
temp_line_pred = []
for x in range(6, 13): 
	temp_line_pred += [(yr[x], yr[x]), (temp_lower[x], temp_upper[x])]
ax1.plot(*temp_line_cal, color = sns.xkcd_rgb["midnight"])
ax1.plot(*temp_line_pred, color = sns.xkcd_rgb["grey"])
ax1.scatter(yr, temp_mean, marker = 'o', facecolors='white', edgecolors=sns.xkcd_rgb["grey"], s=50, linewidth=2)
ax1.scatter(yr, temp_lower, marker = '_', color = sns.xkcd_rgb["grey"])
ax1.scatter(yr, temp_upper, marker = '_', color = sns.xkcd_rgb["grey"])
ax1.scatter(yr[0:6], temp_mean[0:6], marker = 'o', facecolors='white', edgecolors=sns.xkcd_rgb["midnight"], s=50, linewidth=2)
ax1.scatter(yr[0:6], temp_lower[0:6], marker = '_', color = sns.xkcd_rgb["midnight"])
ax1.scatter(yr[0:6], temp_upper[0:6], marker = '_', color = sns.xkcd_rgb["midnight"])
ax1.scatter(yr[0:6], target, marker = '*', facecolors=sns.xkcd_rgb["red orange"], edgecolors = sns.xkcd_rgb["red orange"], s=80, linewidth=3)
# for x in range(len(temp_line)): 
ax1.set_title('(A) Zebra mussels', fontsize =18)
ax1.set_xlabel('year',fontsize=14)
ax1.set_ylabel('# of lakes',fontsize=14)

# fig = plt.figure(figsize=(12,5))
# ax2 = fig.add_subplot(111)
ax2 = fig.add_subplot(122)
yr = np.array([x for x in range(2017, 2026)])
temp_mean =  np.array(pre_summary['ss']['mean'] + summary_ss['S']['mean'])
temp_lower = np.array(pre_summary['ss']['lb'] + summary_ss['S']['lb'])
temp_upper = np.array(pre_summary['ss']['ub'] + summary_ss['S']['ub'])
target = np.array([val for key, val in ss_count.items()])[1:]
temp_line_cal = []
for x in range(0, 2): 
	temp_line_cal += [(yr[x], yr[x]), (temp_lower[x], temp_upper[x])]
temp_line_pred = []
for x in range(2, 9): 
	temp_line_pred += [(yr[x], yr[x]), (temp_lower[x], temp_upper[x])]
ax2.plot(*temp_line_cal, color = sns.xkcd_rgb["midnight"])
ax2.plot(*temp_line_pred, color = sns.xkcd_rgb["grey"])
ax2.scatter(yr, temp_mean, marker = 'o', facecolors='white', edgecolors=sns.xkcd_rgb["grey"], s=50, linewidth=2)
ax2.scatter(yr, temp_lower, marker = '_', color = sns.xkcd_rgb["grey"])
ax2.scatter(yr, temp_upper, marker = '_', color = sns.xkcd_rgb["grey"])
ax2.scatter(yr[0:2], temp_mean[0:2], marker = 'o', facecolors='white', edgecolors=sns.xkcd_rgb["midnight"], s=50, linewidth=2)
ax2.scatter(yr[0:2], temp_lower[0:2], marker = '_', color = sns.xkcd_rgb["midnight"])
ax2.scatter(yr[0:2], temp_upper[0:2], marker = '_', color = sns.xkcd_rgb["midnight"])
ax2.scatter(yr[0:2], target, marker = '*', facecolors=sns.xkcd_rgb["red orange"], edgecolors = sns.xkcd_rgb["red orange"], s=80, linewidth=3)
# for x in range(len(temp_line)): ax2.set_title('(B) Proportion of partnerships in which\n>= 1 person discloses their HIV status', fontsize =18)
ax2.set_title('(B) Starry stonewort', fontsize =18)
ax2.set_xlabel('year',fontsize=14)
ax2.set_ylabel('# of lakes',fontsize=14)

plt.tight_layout()
plt.savefig('results/summary plot annual(new).eps', format='eps', dpi=1000)



# infestation through boaters
# zebra mussels

temp = []
for i in range(1, 101): 
	with open('results/scenarios_res_zm_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))

scenarios_res_zm = [val for sublist in temp for val in sublist]

s = 'S'
p_boat = np.zeros(len(scenarios_res_zm))
p_river = np.zeros(len(scenarios_res_zm))

for i in range(len(scenarios_res_zm)): 
	# i = 0
	p_boat[i] = np.sum(np.array(scenarios_res_zm[i][s]['boat']))/len(scenarios_res_zm[i]['S']['boat'])
	p_river[i] = np.sum(np.array(scenarios_res_zm[i][s]['river']))/len(scenarios_res_zm[i]['S']['boat'])

p_boat[np.isnan(p_boat)] = 0
p_boat_zm = p_boat

temp = []
for i in range(1, 101): 
	with open('results/scenarios_res_ss_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))

scenarios_res_ss = [val for sublist in temp for val in sublist]

s = 'S'
p_boat = np.zeros(len(scenarios_res_ss))
p_river = np.zeros(len(scenarios_res_ss))

for i in range(len(scenarios_res_ss)): 
	# i = 0
	p_boat[i] = np.sum(np.array(scenarios_res_ss[i][s]['boat']))/len(scenarios_res_ss[i]['S']['boat'])
	p_river[i] = np.sum(np.array(scenarios_res_ss[i][s]['river']))/len(scenarios_res_ss[i]['S']['boat'])

p_boat[np.isnan(p_boat)] = 0
p_boat_ss = p_boat



fig = plt.figure(figsize=(12,4.5))
ax1 = fig.add_subplot(121)
temp = p_boat_zm
temp = np.histogram(temp, bins = 30)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

ax1.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax1.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
ax1.set_title("(A) Zebra mussels", fontsize =18)
ax1.set_xlabel('proportion',fontsize=14)
ax1.set_ylabel('density',fontsize=14)
ax1.axvline(np.mean(p_boat_zm), color=sns.xkcd_rgb["midnight"], linewidth=2)
ax1.text(np.mean(p_boat_zm), 0.035, np.round(np.mean(p_boat_zm),2), fontsize=12)

ax1 = fig.add_subplot(122)
temp = p_boat_ss
temp = np.histogram(temp, bins = 30)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

ax1.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax1.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
ax1.set_title("(B) Starry stonewort", fontsize =18)
ax1.set_xlabel('proportion',fontsize=14)
ax1.set_ylabel('density',fontsize=14)
ax1.axvline(np.mean(p_boat_ss), color=sns.xkcd_rgb["midnight"], linewidth=2)
ax1.text(np.mean(p_boat_ss), 0.05, np.round(np.mean(p_boat_ss),2), fontsize=12)

plt.tight_layout()
plt.savefig("results/boat_infestation(new).eps", format='eps', dpi=1000)



# infestation through rivers
# zebra mussels

temp = []
for i in range(1, 101): 
	with open('results/scenarios_res_zm_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))

scenarios_res_zm = [val for sublist in temp for val in sublist]

s = 'S'
p_boat = np.zeros(len(scenarios_res_zm))
p_river = np.zeros(len(scenarios_res_zm))

for i in range(len(scenarios_res_zm)): 
	# i = 0
	p_boat[i] = np.sum(np.array(scenarios_res_zm[i]['S']['boat']))/len(scenarios_res_zm[i]['S']['boat'])
	p_river[i] = np.sum(np.array(scenarios_res_zm[i]['S']['river']))/len(scenarios_res_zm[i]['S']['boat'])

p_river[np.isnan(p_river)] = 0
p_river_zm = p_river

temp = []
for i in range(1, 101): 
	with open('results/scenarios_res_ss_'+str(i)+'.txt') as json_file:  
	    temp.append(json.load(json_file))

scenarios_res_ss = [val for sublist in temp for val in sublist]

s = 'S'
p_boat = np.zeros(len(scenarios_res_ss))
p_river = np.zeros(len(scenarios_res_ss))

for i in range(len(scenarios_res_ss)): 
	# i = 0
	p_boat[i] = np.sum(np.array(scenarios_res_ss[i]['S']['boat']))/len(scenarios_res_ss[i]['S']['boat'])
	p_river[i] = np.sum(np.array(scenarios_res_ss[i]['S']['river']))/len(scenarios_res_ss[i]['S']['boat'])

p_river[np.isnan(p_river)] = 0
p_river_ss = p_river



fig = plt.figure(figsize=(12,4.5))
ax1 = fig.add_subplot(121)
temp = p_river_zm
temp = np.histogram(temp, bins = 30)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

ax1.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax1.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
ax1.set_title("(A) Zebra mussels", fontsize =18)
ax1.set_xlabel('value',fontsize=14)
ax1.set_ylabel('density',fontsize=14)
ax1.axvline(np.mean(p_river_zm), color=sns.xkcd_rgb["midnight"], linewidth=2)
ax1.text(np.mean(p_river_zm), 0.035, np.round(np.mean(p_river_zm),2), fontsize=12)

ax1 = fig.add_subplot(122)
temp = p_boat_ss
temp = np.histogram(temp, bins = 30)
tmp_y = temp[0]/np.sum(temp[0])
tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

ax1.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
ix = tmp_x
iy = tmp_y
ax1.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
ax1.set_title("(B) Starry stonewort", fontsize =18)
ax1.set_xlabel('value',fontsize=14)
ax1.set_ylabel('density',fontsize=14)
ax1.axvline(np.mean(p_river_ss), color=sns.xkcd_rgb["midnight"], linewidth=2)
ax1.text(np.mean(p_river_ss), 0.05, np.round(np.mean(p_river_ss),2), fontsize=12)

plt.tight_layout()
plt.savefig("results/river_infestation(new).eps", format='eps', dpi=1000)



# (Other strategies) infestation through boaters 
# zebra mussels
for s in ['E', 'P', 'D', 'T']: 
	temp = []
	for i in range(1, 101): 
		with open('results/scenarios_res_zm_'+str(i)+'.txt') as json_file:  
		    temp.append(json.load(json_file))

	scenarios_res_zm = [val for sublist in temp for val in sublist]

	p_boat = np.zeros(len(scenarios_res_zm))
	p_river = np.zeros(len(scenarios_res_zm))

	for i in range(len(scenarios_res_zm)): 
		# i = 0
		p_boat[i] = np.sum(np.array(scenarios_res_zm[i][s]['boat']))
		p_river[i] = np.sum(np.array(scenarios_res_zm[i][s]['river']))

	p_boat[np.isnan(p_boat)] = 0
	p_boat_zm = p_boat

	temp = []
	for i in range(1, 101): 
		with open('results/scenarios_res_ss_'+str(i)+'.txt') as json_file:  
		    temp.append(json.load(json_file))

	scenarios_res_ss = [val for sublist in temp for val in sublist]

	p_boat = np.zeros(len(scenarios_res_ss))
	p_river = np.zeros(len(scenarios_res_ss))

	for i in range(len(scenarios_res_ss)): 
		# i = 0
		p_boat[i] = np.sum(np.array(scenarios_res_ss[i][s]['boat']))
		p_river[i] = np.sum(np.array(scenarios_res_ss[i][s]['river']))

	p_boat[np.isnan(p_boat)] = 0
	p_boat_ss = p_boat

	fig = plt.figure(figsize=(12,4.5))
	ax1 = fig.add_subplot(121)
	temp = p_boat_zm
	temp = np.histogram(temp, bins = 30)
	tmp_y = temp[0]/np.sum(temp[0])
	tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

	ax1.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
	ix = tmp_x
	iy = tmp_y
	ax1.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
	ax1.set_title("(A) Zebra mussels", fontsize =18)
	ax1.set_xlabel('value',fontsize=14)
	ax1.set_ylabel('density',fontsize=14)
	ax1.axvline(np.mean(p_boat_zm), color=sns.xkcd_rgb["midnight"], linewidth=2)
	ax1.text(np.mean(p_boat_zm), 0.035, np.round(np.mean(p_boat_zm),2), fontsize=12)

	ax1 = fig.add_subplot(122)
	temp = p_boat_ss
	temp = np.histogram(temp, bins = 30)
	tmp_y = temp[0]/np.sum(temp[0])
	tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

	ax1.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
	ix = tmp_x
	iy = tmp_y
	ax1.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
	ax1.set_title("(B) Starry stonewort", fontsize =18)
	ax1.set_xlabel('value',fontsize=14)
	ax1.set_ylabel('density',fontsize=14)
	ax1.axvline(np.mean(p_boat_ss), color=sns.xkcd_rgb["midnight"], linewidth=2)
	ax1.text(np.mean(p_boat_ss), 0.05, np.round(np.mean(p_boat_ss),2), fontsize=12)

	plt.tight_layout()
	plt.savefig("results/boat_infestation("+s+")(new).eps", format='eps', dpi=1000)




# (Other strategies) infestation through rivers
# zebra mussels
for s in ['E', 'P', 'D', 'T']: 
	temp = []
	for i in range(1, 101): 
		with open('results/scenarios_res_zm_'+str(i)+'.txt') as json_file:  
		    temp.append(json.load(json_file))

	scenarios_res_zm = [val for sublist in temp for val in sublist]

	p_boat = np.zeros(len(scenarios_res_zm))
	p_river = np.zeros(len(scenarios_res_zm))

	for i in range(len(scenarios_res_zm)): 
		# i = 0
		p_boat[i] = np.sum(np.array(scenarios_res_zm[i][s]['boat']))
		p_river[i] = np.sum(np.array(scenarios_res_zm[i][s]['river']))

	p_river[np.isnan(p_river)] = 0
	p_river_zm = p_river

	temp = []
	for i in range(1, 101): 
		with open('results/scenarios_res_ss_'+str(i)+'.txt') as json_file:  
		    temp.append(json.load(json_file))

	scenarios_res_ss = [val for sublist in temp for val in sublist]

	p_boat = np.zeros(len(scenarios_res_ss))
	p_river = np.zeros(len(scenarios_res_ss))

	for i in range(len(scenarios_res_ss)): 
		# i = 0
		p_boat[i] = np.sum(np.array(scenarios_res_ss[i][s]['boat']))
		p_river[i] = np.sum(np.array(scenarios_res_ss[i][s]['river']))

	p_river[np.isnan(p_river)] = 0
	p_river_ss = p_river



	fig = plt.figure(figsize=(12,4.5))
	ax1 = fig.add_subplot(121)
	temp = p_river_zm
	temp = np.histogram(temp, bins = 30)
	tmp_y = temp[0]/np.sum(temp[0])
	tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

	ax1.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
	ix = tmp_x
	iy = tmp_y
	ax1.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
	ax1.set_title("(A) Zebra mussels", fontsize =18)
	ax1.set_xlabel('value',fontsize=14)
	ax1.set_ylabel('density',fontsize=14)
	ax1.axvline(np.mean(p_river_zm), color=sns.xkcd_rgb["midnight"], linewidth=2)
	ax1.text(np.mean(p_river_zm), 0.035, np.round(np.mean(p_river_zm),2), fontsize=12)

	ax1 = fig.add_subplot(122)
	temp = p_river_ss
	temp = np.histogram(temp, bins = 30)
	tmp_y = temp[0]/np.sum(temp[0])
	tmp_x = np.array([(temp[1][x]+temp[1][x+1])/2 for x in range(temp[1][1:].shape[0])])

	ax1.plot(tmp_x, tmp_y, color = sns.xkcd_rgb["grey"], linewidth = 1)
	ix = tmp_x
	iy = tmp_y
	ax1.fill_between(ix,iy, color = sns.xkcd_rgb["light grey"])
	ax1.set_title("(B) Starry stonewort", fontsize =18)
	ax1.set_xlabel('value',fontsize=14)
	ax1.set_ylabel('density',fontsize=14)
	ax1.axvline(np.mean(p_river_ss), color=sns.xkcd_rgb["midnight"], linewidth=2)
	ax1.text(np.mean(p_river_ss), 0.05, np.round(np.mean(p_river_ss),2), fontsize=12)

	plt.tight_layout()
	plt.savefig("results/river_infestation("+s+")(new).eps", format='eps', dpi=1000)




'''
Management scenarios
'''

fig = plt.figure(figsize=(12,5))
# ax1 = fig.add_subplot(111)
ax1 = fig.add_subplot(121)
sc = np.array([x for x in range(0, 5)])
temp_mean =  np.array([summary_zm['S']['mean'][-1], summary_zm['E']['mean'][-1], summary_zm['P']['mean'][-1], \
	summary_zm['D']['mean'][-1], summary_zm['T']['mean'][-1]])
temp_lower = np.array([summary_zm['S']['lb'][-1], summary_zm['E']['lb'][-1], summary_zm['P']['lb'][-1], \
	summary_zm['D']['lb'][-1], summary_zm['T']['lb'][-1]])
temp_upper = np.array([summary_zm['S']['ub'][-1], summary_zm['E']['ub'][-1], summary_zm['P']['ub'][-1], \
	summary_zm['D']['ub'][-1], summary_zm['T']['ub'][-1]])
temp_line = []
for x in range(len(sc)): 
	temp_line += [(sc[x], sc[x]), (temp_lower[x], temp_upper[x])]
ax1.plot(*temp_line, color = sns.xkcd_rgb["blue"])
ax1.scatter(sc, temp_lower, marker = '_', color = sns.xkcd_rgb["blue"])
ax1.scatter(sc, temp_upper, marker = '_', color = sns.xkcd_rgb["blue"])
ax1.scatter(sc, temp_mean, marker = 'o', facecolors='white', edgecolors=sns.xkcd_rgb["blue"], s=100, linewidth=3)
for x in range(len(sc)): 
	ax1.text(sc[x], temp_mean[x], np.round(temp_mean[x]).astype(int), fontsize=12)
ax1.set_title('(A) Zebra mussels', fontsize =18)
plt.xticks(sc, ('StatusQuo', 'Education', 'Penalty', 'MandDecon', 'RedTraffic'))
ax1.set_xlabel('scenarios',fontsize=14)
ax1.set_ylabel('# of lakes',fontsize=14)

ax2 = fig.add_subplot(122)
sc = np.array([x for x in range(0, 5)])
temp_mean =  np.array([summary_ss['S']['mean'][-1], summary_ss['E']['mean'][-1], summary_ss['P']['mean'][-1], \
	summary_ss['D']['mean'][-1], summary_ss['T']['mean'][-1]])
temp_lower = np.array([summary_ss['S']['lb'][-1], summary_ss['E']['lb'][-1], summary_ss['P']['lb'][-1], \
	summary_ss['D']['lb'][-1], summary_ss['T']['lb'][-1]])
temp_upper = np.array([summary_ss['S']['ub'][-1], summary_ss['E']['ub'][-1], summary_ss['P']['ub'][-1], \
	summary_ss['D']['ub'][-1], summary_ss['T']['ub'][-1]])
temp_line = []
for x in range(len(sc)): 
	temp_line += [(sc[x], sc[x]), (temp_lower[x], temp_upper[x])]
ax2.plot(*temp_line, color = sns.xkcd_rgb["blue"])
ax2.scatter(sc, temp_lower, marker = '_', color = sns.xkcd_rgb["blue"])
ax2.scatter(sc, temp_upper, marker = '_', color = sns.xkcd_rgb["blue"])
ax2.scatter(sc, temp_mean, marker = 'o', facecolors='white', edgecolors=sns.xkcd_rgb["blue"], s=100, linewidth=3)
for x in range(len(sc)): 
	ax2.text(sc[x], temp_mean[x], np.round(temp_mean[x]).astype(int), fontsize=12)
ax2.set_title('(B) Starry stonewort', fontsize =18)
plt.xticks(sc, ('StatusQuo', 'Education', 'Penalty', 'MandDecon', 'RedTraffic'))
ax2.set_xlabel('scenarios',fontsize=14)
ax2.set_ylabel('# of lakes',fontsize=14)

plt.tight_layout()
plt.savefig('results/management strategies(new).eps', format='eps', dpi=1000)



# management summary
print(dict(zip([k for k, v in summary_zm.items()], [v['mean'][-1] for k, v in summary_zm.items()])))
print(dict(zip([k for k, v in summary_zm.items()], [v['mean'][-1] for k, v in summary_ss.items()])))


'''
# read networks
with open('data/boat_dict.txt') as json_file:  
    boat_dict = json.load(json_file)

boat_o = []
boat_d = []
boat_w = []
tmp_key = [k for k, v in boat_dict.items()]
for k in tmp_key: 
	k2, v2 = [[key for key, val in boat_dict[k].items()], [val for key, val in boat_dict[k].items()]]
	k2 = [int(x) for x in k2]
	if len(k2) > 0:
		tmp_o = len(k2)*[int(k)]
		tmp_d = k2
		tmp_w = v2
		boat_o.append(tmp_o)
		boat_d.append(tmp_d)
		boat_w.append(tmp_w)


boat_o = [val for sublist in boat_o for val in sublist]
boat_d = [val for sublist in boat_d for val in sublist]
boat_w = [val for sublist in boat_w for val in sublist]

boat_net = np.column_stack((boat_o, boat_d, boat_w))

boat_net = pd.DataFrame(boat_net)
boat_net.columns = ['id.origin', 'id.destination', 'weight']
boat_net['id.origin'].astype(int)
boat_net['id.destination'].astype(int)

att = pd.read_csv('data/lake_attribute.csv')
att['boat'].fillna(att['boat'].mean(), inplace=True)
att['infest'].fillna(0, inplace=True)
att['inspect'].fillna(0, inplace=True)

boat_net = pd.merge(boat_net, att[['id','dow','lake_name']], how = 'left', left_on = 'id.origin', right_on = 'id')
boat_net = boat_net[['id.origin', 'id.destination', 'weight', 'dow', 'lake_name']]
boat_net.columns = ['id.origin', 'id.destination', 'weight', 'dow.origin', 'lake.origin']

boat_net = pd.merge(boat_net, att[['id','dow','lake_name']], how = 'left', left_on = 'id.destination', right_on = 'id')
boat_net = boat_net[['id.origin', 'id.destination', 'weight', 'dow.origin', 'lake.origin', 'dow', 'lake_name']]
boat_net.columns = ['id.origin', 'id.destination', 'weight', 'dow.origin', 'lake.origin', \
'dow.destination', 'lake.destination']

boat_net.to_csv("data/boat_net_for_R.csv")


river_net = pd.read_csv('data/river_net_sim.csv')

ssID = [20898, 17743, 721, 11938, 12002, 12007, 12131, 21039, 21043, 23548]
np.in1d(river_net.loc[river_net['dow.destination'] == 18068000, ['id.origin']], ssID)

# check whitefish lake group
wf = [18064400, 18031000, 18031100, 18068000, 18035400, 18035500, 18026600, 18026800, 18036600, 18026900, 18027000, 18031200, 18063900, 18031500, 18037800, 18027100]

print(lake_risk_ss.loc[lake_risk_ss['dow'].isin(wf), ['lake_name', 'acre', 'S_p_infest', 'S_p_boat', 'S_p_river']])


zmID = att.loc[att['zm_infest2017'] == 1, 'id'].tolist()
np.in1d(river_net.loc[river_net['dow.destination'] == 18039900, ['id.origin']], zmID)

kk = river_net.loc[river_net['dow.destination'] == 18039900, ['id.origin']][np.in1d(river_net.loc[river_net['dow.destination'] == 18039900, ['id.origin']], zmID)].values.T.tolist()[0]
print(att.loc[att['id'].isin(kk), ['dow', 'lake_name']])
print(river_net.loc[(river_net['dow.destination'] == 18039900) & \
river_net['id.origin'].isin(kk), ['dow.origin', 'dow.destination', 'weight']])
'''



