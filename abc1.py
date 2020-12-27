import numpy as np
import pandas as pd
import timeit
import copy
import json
import os
import ABCSMC.ABCSMC as abc
import ABCSMC.data as abcdata
import ABCSMC.SampleTasks as task

myenv = "msi"

lake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit, boat_net, \
	river_o, river_d, river_w, target_all, sd_all = abcdata.AllData(env = myenv).get_all_data()

t = 1
t_minus1 = 0
n_samp = 5
i_file = 1

with open('genout/gen' + str(t_minus1) + '.txt') as json_file:
	last_gen = json.load(json_file)

for i in range((n_samp * (i_file - 1) + 1), (n_samp * i_file + 1)):
	print('i = ', str(i))
	aa = timeit.default_timer()
	output = task.particle_sample(last_gen, t, \
		lake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit, boat_net, \
		river_o, river_d, river_w, target_all, sd_all)

	with open('genout/genout' + str(t) + '_' + str(i) + '.txt', 'w') as fout:
		json.dump(output, fout)
	print(timeit.default_timer()-aa)
