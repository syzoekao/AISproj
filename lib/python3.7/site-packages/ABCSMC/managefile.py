import os
import json
import re
import glob
import ABCSMC.data as data

def append_bayes_out(env, path, file_pattern):
	data.SetDirectory(env)

	tmp = os.listdir(path)
	tmp = [val for sublist in [re.findall(file_pattern + '(.*).txt', x) for x in tmp] \
		for val in sublist]

	outfile = []
	for i in tmp: 
		with open(path + '/' + file_pattern + str(i) + '.txt') as json_file:
			outfile += [json.load(json_file)]
	return outfile


def remove_files(env, path, file_pattern): 
	data.SetDirectory(env)

	files = glob.glob(path + '/' + file_pattern + '*')
	for f in files:
		os.remove(f)


