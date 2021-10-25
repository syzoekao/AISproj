import numpy as np
import pandas as pd
import copy
import json
import timeit
import re
import os
os.chdir("/Users/zoekao/Documents/FishProject/virsim2")
cwd = os.getcwd()
print(cwd)

pars_name = ["factor_ss", "e_violate_zm", "e_violate_ss", "river_inf_zm", "river_inf_ss", \
	"back_suit_zm", "back_suit_ss"]

i = 10

data = open("data/MH" + str(i) + ".txt", "r")
sim = data.readlines()
data.close()

out = []
tmp_repeat = []

for x in range(len(sim)): 
	if "Accept" in sim[x]: 
		tmp_repeat.append(int(re.findall("\d+", sim[x])[0]))
		temp = str.split(sim[x], 'Accepted:  [')[1].rstrip().split(' ') + str.strip(sim[x + 1], ' []\n ').split(' ')
		temp = [y for y in temp if len(y) > 0]
		temp = [[float(x) for x in temp]]
		out = out + temp

n_repeat = np.array(tmp_repeat[1:]) - np.array(tmp_repeat[0:-1])
out = np.array(out)
out = out[:-1]

tmp_out = np.repeat(out, n_repeat, axis = 0)

tmp_out = tmp_out[5000:30000]

sim_param = {}

for x in range(len(pars_name)):
	tmp_name = pars_name[x] 
	sim_param[tmp_name] = tmp_out[:, x].tolist()

with open('data/mcmc_results_' + str(i) + '.txt', 'w') as fout:
	json.dump(sim_param, fout)



