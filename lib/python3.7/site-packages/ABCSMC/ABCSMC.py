import numpy as np
import ABCSMC.data as abcdata
import AISsim.UtilFunction as util
import scipy.stats 
from sklearn.covariance import ledoit_wolf
from itertools import product


def prior_draw(): 
	out_samp = abcdata.DrawParams().pars
	return out_samp

def prior_prob(prior_v): 
	out_prob = scipy.stats.uniform.pdf(prior_v, 0, 0.5)
	out_prob = np.prod(out_prob)
	return out_prob


def tolerance(method, epsMax = 60, epsMin = 5, generation = 10): 
	if method == 'constant': 
		return np.repeat(espMin, generation)
	elif method == "linear": 
		return np.linspace(epsMax,epsMin,num=generation)
	elif method == "exponential": 
		return np.logspace(np.log10(epsMax), np.log10(epsMin), num=generation)
	else: 
		return np.array(method)

def distance_measure(target, simOut, std): 
	return np.sum((target - simOut) ** 2 / (2 * std ** 2))

def var_cov(particles, wgt, min_ix, t): 
	'''
	https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance
	Used the reliability weights to calculate the weighted covariance because 
	the sample is non-random. 
	'''
	if t == 0:
		return 2. * np.cov(particles.T)
	else:
		n = particles.shape[1]
		cov = np.zeros((n, n)) 
		w = wgt.sum()/(wgt.sum()**2 - (wgt**2).sum()) 
		average = np.average(particles[min_ix], axis = 0, weights = wgt[min_ix])
		# average = particles[min_ix]
		for j in range(n): 
			for k in range(n): 
				cov[j, k] = w * np.sum(wgt * (particles[:,j] - average[j]) * (particles[:,k] - average[k]))
	return 2 * cov

def kernel(cov, new_theta):
	# if np.linalg.det(cov) < 1.E-15: 
	# 	'''
	# 	if the covariance matrix is sigular, use Ledoit-Wolf estimator
	# 	'''
	# 	cov, _ = ledoit_wolf(theta_minus1) 
	return scipy.stats.multivariate_normal(mean = new_theta, cov = cov, allow_singular = True).pdf


def part_weight(new_theta, priorProb, cov, theta_minus1, wgt_minus1): 
	'''
	calculating the particle weight for each particle i (pid)
	'''
	k = kernel(cov, new_theta)
	kernel_v = k(theta_minus1)
	wgt = priorProb / np.sum(wgt_minus1 * kernel_v)
	return wgt


def init_particle(run, lake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit, boat_net, \
		river_o, river_d, river_w, target_all, sd_all):
	# myenv = "mylaptop"

	# Loading data
	# lake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit, boat_net, \
	# 	river_o, river_d, river_w, target_all, sd_all = \
	# 	abcdata.AllData(env = myenv).get_all_data()

	sim_target = "toss away sample"
	while isinstance(sim_target, str): 
		k_X = prior_draw()
		# k_X = [0.03623706, 0.02003358, 0.01008539, 0.07430393, 0.03270713, 0.13896991, 0.36345734]
		sim_target = util.infest_outcome_func(factor_ss = k_X[0], 
			e_violate_zm = k_X[1], e_violate_ss = k_X[2], river_inf_zm = k_X[3], 
			river_inf_ss = k_X[4], back_suit_zm = k_X[5], back_suit_ss = k_X[6], 
			boat_net = boat_net, 
			river_o = river_o, river_d = river_d, river_w = river_w, \
			lake_id = lake_id, infest_zm = infest_zm, infest_ss = infest_ss, infest_both = infest_both, \
			zm_suit = zm_suit, ss_suit = ss_suit)
	
	print(k_X)

	distance = distance_measure(target_all, sim_target, sd_all)
	llk = np.sum(scipy.stats.norm.logpdf(sim_target, loc = target_all, scale = sd_all)) 
	weight = 1
	return {"par": k_X, "distance": distance, "llk": llk, "weight": weight, 'sim_target': sim_target.tolist()}

