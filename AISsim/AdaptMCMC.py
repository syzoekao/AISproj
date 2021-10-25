import numpy as np
import scipy.stats as stats
import AISsim.UtilFunction as util
import timeit


def prior_bounds(prop, lb, ub): 
	out_ = np.array([stats.uniform.logpdf(prop[x], loc=lb[x], scale=ub[x]) for x in range(prop.shape[0])])
	if np.isinf(np.sum(out_)): 
		out_ = np.zeros(prop.shape[0])
	return np.sum(out_)



def AdpativeMCMC(iters, burn_in, adapt_par, pars, pars_name, pars_lb, pars_ub, scales, target, target_sd, \
	verbose, function, **kwargs): 
	'''
	pars = [0.03623706, 0.02003358, 0.01008539, 0.07430393, 0.03270713, 0.13896991, 0.36345734]
	pars_name = ["factor_ss", "e_violate_zm", "e_violate_ss", "river_inf_zm", "river_inf_ss", "back_suit_zm", "back_suit_ss"]
	pars_lb = [0, 0, 0, 0, 0, 0, 0]
	pars_ub = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

	scales = {"factor_ss":0.0001, "e_violate_zm":0.00005, "e_violate_ss":0.0001, \
	"river_inf_zm": 0.0001, "river_inf_ss": 0.0001, "back_suit_zm": 0.0001, "back_suit_ss": 0.0001}

	iters = 1010
	burn_in = 10
	adapt_par = [100, 100]
	verbose = 1

	zm_count = {2010: 37,  2011: 41,  2012: 54,  2013: 69,  2014: 85,  2015: 100,  2016: 133,  2017: 165}
	ss_count = {2016: 8,  2017: 10}

	target_zm = np.array([val for key, val in zm_count.items() if key > 2012])
	target_ss = np.array([val for key, val in ss_count.items() if key > 2016])
	target_all = np.append(target_zm, target_ss)

	sd_zm = np.std(target_zm)
	sd_ss = target_ss
	sd_all = np.append(np.repeat(sd_zm, 5) , sd_ss)
	'''

	adapt_step = [adapt_par[0]] + [adapt_par[0] + adapt_par[1]*x for x in range(1, int((iters-adapt_par[0])/adapt_par[1]))]

	trace = np.empty((iters+1, len(pars)))
	trace2 = np.empty((iters+1, target.shape[0]+1))

	means = np.zeros(len(pars)).astype(float)

	if type(scales) is dict: 
		cov = np.zeros((len(pars), len(pars)))
		ix = np.diag_indices(cov.shape[0])
		cov[ix] = np.array([val for key, val in scales.items()])
	else: 
		cov = scales # if scales should be a squared array

	k_X = np.array(pars)

	lprior = prior_bounds(prop = k_X, lb = pars_lb, ub = pars_ub)
	
	sim_target = function(factor_ss = k_X[0], 
			e_violate_zm = k_X[1], e_violate_ss = k_X[2], river_inf_zm = k_X[3], 
			river_inf_ss = k_X[4], back_suit_zm = k_X[5], back_suit_ss = k_X[6], 
			boat_net = kwargs["boat_net"], # small_prob = kwargs["small_prob"], 
			river_o=kwargs["river_o"], river_d=kwargs["river_d"], river_w=kwargs["river_w"], 
			lake_id=kwargs["lake_id"], infest_zm=kwargs["infest_zm"], infest_ss=kwargs["infest_ss"], 
			infest_both=kwargs["infest_both"], zm_suit=kwargs["zm_suit"], ss_suit=kwargs["ss_suit"])

	llk = np.sum(stats.norm.logpdf(sim_target, loc=target, scale=target_sd)) 
	lpost_X = lprior + llk

	trace[0] = k_X
	trace2[0] = np.append(sim_target, lpost_X)


	for i in range(1, iters+1): 
		# aaa = timeit.default_timer()
		k_Y = k_X + np.random.multivariate_normal(means, cov) # sample the parameters
		lprior = prior_bounds(prop = k_Y, lb = pars_lb, ub = pars_ub)

		if lprior > 0: 

			sim_target = function(factor_ss = k_Y[0], 
			e_violate_zm = k_Y[1], e_violate_ss = k_Y[2], river_inf_zm = k_Y[3], 
			river_inf_ss = k_Y[4], back_suit_zm = k_Y[5], back_suit_ss = k_Y[6], 
			boat_net = kwargs["boat_net"], # small_prob=kwargs['small_prob'], 
			river_o=kwargs["river_o"], river_d=kwargs["river_d"], river_w=kwargs["river_w"], 
			lake_id=kwargs["lake_id"], infest_zm=kwargs["infest_zm"], infest_ss=kwargs["infest_ss"], 
			infest_both=kwargs["infest_both"], zm_suit=kwargs["zm_suit"], ss_suit=kwargs["ss_suit"])
			# print(sim_target)
			
			'''
			sim_target = function(factor_ss = k_Y[0], 
			e_violate_zm = k_Y[1], e_violate_ss = k_Y[2], river_inf_zm = k_Y[3], 
			river_inf_ss = k_Y[4], back_suit_zm = k_Y[5], back_suit_ss = k_Y[6], 
			boat_net = boat_net, 
			river_o=river_o, river_d=river_d, river_w=river_w, 
			lake_id=lake_id, infest_zm=infest_zm, infest_ss=infest_ss, 
			infest_both=infest_both, zm_suit=zm_suit, ss_suit=ss_suit)
			'''

			llk = np.sum(stats.norm.logpdf(sim_target, loc=target, scale=target_sd)) 
			lpost_Y = lprior + llk

			diff_X_Y = lpost_Y - lpost_X

			if(np.log(stats.uniform.rvs(0, 1)) < diff_X_Y): 
				k_X = k_Y
				lpost_X = lpost_Y
				if verbose == 1: 
					print("Step:", i , "; Accepted: ", k_X)
					print("log-posterior", lpost_X)
					print("results:", sim_target)
		
		# update trace
		trace[i] = k_X
		trace2[i] = np.append(sim_target, lpost_X)

		if i in adapt_step:  # update covariance
			cov_X = cov
			l = np.arange(i - adapt_par[1] + 1, i+1, 1)
			N = l.shape[0]
			cov = (N-1) * np.cov(trace[(i+1-adapt_par[1]):(i+1)].T)/N
			try: 
				np.linalg.cholesky(cov)
			except np.linalg.LinAlgError: 
				cov = cov_X
				print("not positive-definite")

		if i in [int(iters*x) for x in np.arange(0, 1.1, 0.1)]: 
			if verbose > 0: 
				print("========================================")
				print(str(int(i/iters*100))+ "% updated")
				print("parameter: ", k_X, "; log-post: ", lpost_X)
		
		# print(timeit.default_timer()-aaa)

	acc_rate = np.unique(trace[:, 1]).shape[0]/iters
	trace = trace[burn_in:]
	trace2 = trace2[burn_in:]

	output = dict()

	for k in range(len(pars_name)): 
		output.update({pars_name[k]: trace[:, k].tolist()})
	for k in range(len(target)): 
		output.update({'target'+str(k): trace2[:, k].tolist()})
	output.update({'logP': trace2[:, -1].tolist()})
	output.update({'acceptance': acc_rate})
	return output


# import matplotlib as mpl
# print(mpl.rcParams['backend'])
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.get_backend()

# plt.plot(trace[:, 4])
# plt.show()

# plt.hist(trace[:, 4])
# plt.show()

# plt.plot(trace2[:, 5])
# plt.show()

# plt.hist(trace2[:, 5])
# plt.show()



