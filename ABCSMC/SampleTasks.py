import os
import numpy as np
import sys
import re
import ABCSMC.ABCSMC as abc
import AISsim.UtilFunction as util
import ABCSMC.data as data
import time
import json

'''
app = Celery('tasks', 
	broker='redis://localhost:6379/0', 
	backend='redis://localhost:6379/0')
@celery.signals.worker_process_init.connect()
def seed_rng(**_):
	"""
	Seeds the numpy random number generator.
	"""
	np.random.seed()
'''

# @app.task
def init_simulation_task(run, env):
	return abc.init_particle(run, env)

# @app.task
def summary_for_init_wrapper(res_from_init, n_sel = 1000): 
	particles = np.array([res_from_init[x]['par'] for x in range(len(res_from_init))])
	distance = np.array([res_from_init[x]['distance'] for x in range(len(res_from_init))])
	llk = np.array([res_from_init[x]['llk'] for x in range(len(res_from_init))])
	weight = np.array([res_from_init[x]['weight'] for x in range(len(res_from_init))])
	sim_target = np.array([res_from_init[x]['sim_target'] for x in range(len(res_from_init))])
	# n_sel = 1000
	ix = np.argsort(distance)[:n_sel]
	particles = particles[ix]
	distance = distance[ix]
	sim_target = sim_target[ix]
	weight = weight[ix]
	weight = weight/weight.sum()
	min_ix = np.argsort(distance)[:100]
	varcov = abc.var_cov(particles, weight, min_ix, 0)
	return {'par':particles.tolist(), 'distance': distance.tolist(), \
		'llk': llk.tolist(), 'weight': weight.tolist(), \
		'varcov': varcov.tolist(), 'sim_target': sim_target.tolist()}

# @app.task
def summary_wrapper(res_from_particle, t): 
	particles = np.array([res_from_particle[x]['par'] for x in range(len(res_from_particle))])
	weight = np.array([res_from_particle[x]['weight'] for x in range(len(res_from_particle))])
	weight = weight/weight.sum()
	distance = np.array([res_from_particle[x]['distance'] for x in range(len(res_from_particle))])
	min_ix = np.argsort(distance)[:100]
	varcov = abc.var_cov(particles, weight, min_ix, t).tolist()
	weight = weight.tolist()
	distance = distance.tolist()
	sim_target = [res_from_particle[x]['sim_target'] for x in range(len(res_from_particle))]
	return {'par':particles.tolist(), 'weight': weight, 'varcov': varcov, 'distance': distance, \
		"sim_target": sim_target}


# @app.task
def particle_sample(res_from_prev, gen_t, \
	lake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit, boat_net, \
	river_o, river_d, river_w, target_all, sd_all): 
	'''
	seed = np.random.choice([x for x in range(10000)], 1)
	print(seed)
	np.random.seed(seed)
	'''
	theta_minus1 = np.array(res_from_prev['par'])
	weight_minus1 = np.array(res_from_prev['weight'])
	varcov_minus1 = np.array(res_from_prev['varcov'])
	# epsilon = abc.tolerance(method = 'linear', 1000, 200, generation)[gen_t]
	epsilon = abc.tolerance(method = [40, 30, 20, 10, 5, 4, 3, 2])[gen_t - 1]

	distance = epsilon + 100
	while distance > epsilon:
		tmp_new_theta = theta_minus1[np.random.choice(theta_minus1.shape[0], 1, 
			replace = False, p = weight_minus1)[0]]
		new_theta = np.random.multivariate_normal(tmp_new_theta, varcov_minus1)
		priorProb = abc.prior_prob(new_theta)

		if priorProb == 0: 
			distance = epsilon + 100
		else: 
			res_array = np.zeros((5, target_all.shape[0]))
			for i in range(5): 
				sim_out = util.infest_outcome_func(factor_ss = new_theta[0], 
					e_violate_zm = new_theta[1], e_violate_ss = new_theta[2], river_inf_zm = new_theta[3], 
					river_inf_ss = new_theta[4], back_suit_zm = new_theta[5], back_suit_ss = new_theta[6], 
					boat_net = boat_net, \
					river_o = river_o, river_d = river_d, river_w = river_w, \
					lake_id = lake_id, infest_zm = infest_zm, infest_ss = infest_ss, infest_both = infest_both, \
					zm_suit = zm_suit, ss_suit = ss_suit)
				if isinstance(sim_out, str): 
					distance = epsilon + 100
				else: 
					res_array[i] = sim_out
			sim_res = np.mean(res_array, axis = 0)
			distance = abc.distance_measure(target_all, sim_res, sd_all)
	wgt = abc.part_weight(new_theta, priorProb, varcov_minus1, theta_minus1, weight_minus1)
	return {"par": new_theta.tolist(), "distance": distance, "weight": wgt, "sim_target": sim_res.tolist()}


'''
def calc_abc_measures(scndir, gen_t): 
	if gen_t == 0:
		tmp_dir = os.listdir("simdata/" + scndir)
		tmp_num = [int(val) for sublist in \
			[re.findall('outputfile_(.*).txt', x) for x in tmp_dir] for val in sublist]
		outputfile = []
		for i in tmp_num: 
			with open('simdata/' + scndir + '/outputfile' + str(i) + '.txt') as json_file:
				outputfile += json.load(json_file)
		ret = summary_for_init_wrapper(temp, 1000)
	else: 
		tmp_dir = os.listdir("simdata/" + scndir)
		tmp_num = [int(val) for sublist in \
			[re.findall('genout' + str(gen_t) + '_(.*).txt', x) for x in tmp_dir] for val in sublist]
		result = []
		for i in tmp_num: 
			with open('simdata/' + scndir + '/genout' + str(gen_t) + '_' + str(i) + '.txt') as json_file:  
				temp = [json.load(json_file)]
			result += temp
		ret = summary_wrapper(result, gen_t)
	with open('simdata/' + scndir + '/gen' + str(gen_t) + '.txt', 'w') as fout:
		json.dump(ret, fout)
'''


def post_sample(new_theta, \
	lake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit, boat_net, \
	river_o, river_d, river_w, target_all, sd_all): 

	tmpout = util.infest_outcome_func(factor_ss = new_theta[0], \
		e_violate_zm = new_theta[1], e_violate_ss = new_theta[2], river_inf_zm = new_theta[3], \
		river_inf_ss = new_theta[4], back_suit_zm = new_theta[5], back_suit_ss = new_theta[6], \
		boat_net = boat_net, \
		river_o = river_o, river_d = river_d, river_w = river_w, \
		lake_id = lake_id, infest_zm = infest_zm, infest_ss = infest_ss, infest_both = infest_both, \
		zm_suit = zm_suit, ss_suit = ss_suit)

	tmpout = {'zm': tmpout[:7].tolist(), 'ss': tmpout[7:].tolist()}

	tmpout.update({'par': new_theta.tolist()})
	return tmpout


'''
def post_summary(ret): 
	tmp_dd_m = ret['dd_m']
	tmp_dd_m = np.reshape(tmp_dd_m, (-1, int(len(tmp_dd_m)/10)))
	tmp_dd_f = ret['dd_f']
	tmp_dd_f = np.reshape(tmp_dd_f, (-1, int(len(tmp_dd_f)/10)))
	tmp_concur = np.round(ret['concurrency']['all'], 4)
	tmp_turnover = np.round(ret['turnover'], 4)
	tmp_prev = np.array(ret['STIprev'])
	return tmp_dd_m, tmp_dd_f, tmp_concur, tmp_turnover, tmp_prev


def post_summary_new(ret): 
	tmp_dd_m = ret['dd_m']
	tmp_dd_m = np.reshape(tmp_dd_m, (-1, int(len(tmp_dd_m)/10)))
	tmp_dd_f = ret['dd_f']
	tmp_dd_f = np.reshape(tmp_dd_f, (-1, int(len(tmp_dd_f)/10)))
	tmp_concur = np.round(ret['concurrency']['all2'], 4)
	tmp_concur_yr = ret['concurrency']['all']
	tmp_turnover = np.round(ret['turnover'], 4)
	tmp_prev = np.array(ret['STIprev'])
	tmp_part = np.array(ret['meanPart'])
	return tmp_dd_m, tmp_dd_f, tmp_concur, tmp_concur_yr, tmp_turnover, tmp_prev, tmp_part


def sample_SN(x, scenario, gen_t): 
	scndir = {1: 'scenario1', 2: 'scenario2', 3: 'scenario3', 4: 'scenario4'}
	with open('simdata/'+scndir[scenario]+'/gen'+str(gen_t)+'.txt') as json_file:  
		temp = json.load(json_file)
	particles = np.array(temp['par'])
	new_theta = theta[x]
	tmpout = ng.het_network_gen(pPrimary = new_theta[0], gapRel_m = new_theta[1], \
		gapRel_f = new_theta[2], dissFact = new_theta[3], formFactorWPri = new_theta[4], \
		formFactor = new_theta[5], durRelCas1 = new_theta[6], durRelCas2 = new_theta[7], \
		durRelPri1 = new_theta[8], durRelPri2 = new_theta[9], pInf = new_theta[10], \
		N_m = 1000, scenario = scenario, post = True, exportSN = True)
	return tmpout
'''
