import numpy as np
import pandas as pd
import timeit
import copy
import json
import os

class MyError(Exception): 
	def __init__(self, arg): 
		self.arg = arg

class SetDirectory: 
	def __init__(self, env): 
		if env == "mylaptop": 
			self.wdpath = "/Users/zoekao/Documents/FishProject/virsim2"
		if env == "msi": 
			self.wdpath = "/home/ennse/kaoxx085/fish/virsim"
		os.chdir(self.wdpath)

class LoadData(SetDirectory): 
	def __init__(self, env): 
		SetDirectory.__init__(self, env)

	def getLakeAttributes(self):
		att = pd.read_csv('data/lake_attribute.csv')
		att['infest'].fillna(0, inplace=True)
		att['inspect'].fillna(0, inplace=True)
		att['zm_suit'].fillna(0, inplace=True)
		att['ss_suit'].fillna(0, inplace=True)
		self.lake_id = att['id'].values
		self.infest_zm = att['infest_zm'].values
		self.infest_ss = att['infest_ss'].values
		self.infest_both = self.infest_zm
		self.zm_suit = att['zm_suit'].values
		self.ss_suit = att['ss_suit'].values
		return self.lake_id, self.infest_zm, self.infest_ss, self.infest_both, self.zm_suit, self.ss_suit
	
	def getBoaterNet(self): 
		with open('data/boat_dict.txt') as json_file:  
			boat_dict = json.load(json_file)

		boat_net = dict()
		tmp_key = [k for k, v in boat_dict.items()]
		for k in tmp_key: 
			k2, v2 = [[key for key, val in boat_dict[k].items() if key != k], \
			[val for key, val in boat_dict[k].items() if key != k]]
			k2 = [int(x) for x in k2]
			v2 = [x for x in v2]
			if len(k2) > 0: 
				boat_net.update({int(k): dict(zip(k2, v2))})

		self.boat_net = boat_net
		return self.boat_net

	def getRiverNet(self): 
		river_net = pd.read_csv('data/river_net_sim.csv')
		self.river_o = river_net['origin'].values
		self.river_d = river_net['destination'].values
		self.river_w = river_net['weight'].values	
		return self.river_o, self.river_d, self.river_w

class DrawParams: 
	def __init__(self): 
		self.pars_name = ["factor_ss", "e_violate_zm", "e_violate_ss", "river_inf_zm", \
			"river_inf_ss", "back_suit_zm", "back_suit_ss"]
		k_Y = np.random.uniform(0, 0.5, len(self.pars_name))
		self.pars = k_Y.tolist()

class Target: 
	def __init__(self, lake_id): 
		zm_yr = np.arange(2012, 2020)
		zm_count = np.array([48, 63, 79, 94, 126, 160, 194, 218])
		ss_yr = np.arange(2016, 2020)
		ss_count = np.array([8, 10, 13, 14])

		target_zm = zm_count[1:]
		target_ss = ss_count[1:]
		self.target_all = np.append(target_zm, target_ss)

		sd_zm = np.sqrt((target_zm / lake_id.shape[0] * \
			(1 - target_zm / lake_id.shape[0])) / lake_id.shape[0]) * lake_id.shape[0]
		sd_ss = 1
		self.sd_all = np.append(sd_zm , np.repeat(sd_ss, 3))


class AllData(LoadData): 
	def __init__(self, env = "mylaptop"): 
		super().__init__(env)

	def get_all_data(self):
		# Load lake attributes
		lake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit = super().getLakeAttributes()
		boat_net = super().getBoaterNet()
		river_o, river_d, river_w = super().getRiverNet()

		# Loading targets
		load_target = Target(lake_id)
		target_all = load_target.target_all
		sd_all = load_target.sd_all 

		return lake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit, boat_net, \
			river_o, river_d, river_w, target_all, sd_all

