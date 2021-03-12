import re
import ABCSMC.data as data

def gen_init_files(i_file): 
	file = open("msi_files/abc" + str(i_file) + ".py", "w")
	file.write("import numpy as np\n\
import pandas as pd\n\
import timeit\n\
import copy\n\
import json\n\
import os\n\
import ABCSMC.ABCSMC as abc\n\
import ABCSMC.data as abcdata\n\
\n\
myenv = \"msi\"\n\
lake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit, boat_net, \\\n\
\triver_o, river_d, river_w, target_all, sd_all = abcdata.AllData(env = myenv).get_all_data()\n\
\n\
i_file = " + str(i_file) + "\n\
\n\
result_l = []\n\
\n\
n_samp = 50\n\
\n\
for i in range((n_samp * (i_file - 1) + 1), (n_samp * i_file + 1)):\n\
\tprint('i = ', str(i))\n\
\taa = timeit.default_timer()\n\
\toutput = abc.init_particle(1, lake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit, boat_net, \\\n\
\t\triver_o, river_d, river_w, target_all, sd_all)\n\
\n\
\twith open(\'genout/genout0_\' + str(i) + \'.txt\', \'w\') as fout:\n\
\t\tjson.dump(output, fout)\n\
\tprint(timeit.default_timer()-aa)\n\
")
	file.close()

def gen_pbs(filename, simfile):
	file = open("msi_files/" + filename + ".pbs", "w")
	file.write("#!/bin/bash\n\
#PBS -l nodes=1:ppn=1,pmem=2500mb,walltime=48:00:00\n\
#PBS -m abe\n\
#PBS -M kaoxx085@umn.edu\n\
cd /home/ennse/kaoxx085/fish/virsim\n\
module load python\n\
source bin/activate\n\
python msi_files/" + simfile + "$PBS_ARRAYID.py\n\
")
	file.close()


def gen_bayes_files(i_file, gen_t): 
	file = open("msi_files/abc" + str(i_file) + ".py", "w")
	file.write("import numpy as np\n\
import pandas as pd\n\
import timeit\n\
import copy\n\
import json\n\
import os\n\
import ABCSMC.ABCSMC as abc\n\
import ABCSMC.data as abcdata\n\
import ABCSMC.SampleTasks as task\n\
\n\
myenv = \"msi\"\n\
\n\
lake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit, boat_net, \\\n\
\triver_o, river_d, river_w, target_all, sd_all = abcdata.AllData(env = myenv).get_all_data()\n\
\n\
t = " + str(gen_t) + "\n\
t_minus1 = " + str(gen_t - 1) + "\n\
n_samp = 5\n\
i_file = " + str(i_file) + "\n\
\n\
with open(\'genout/gen\' + str(t_minus1) + \'.txt\') as json_file:\n\
\tlast_gen = json.load(json_file)\n\
\n\
for i in range((n_samp * (i_file - 1) + 1), (n_samp * i_file + 1)):\n\
\tprint('i = ', str(i))\n\
\taa = timeit.default_timer()\n\
\toutput = task.particle_sample(last_gen, t, \\\n\
\t\tlake_id, infest_zm, infest_ss, infest_both, zm_suit, ss_suit, boat_net, \\\n\
\t\triver_o, river_d, river_w, target_all, sd_all)\n\
\n\
\twith open(\'genout/genout\' + str(t) + \'_\' + str(i) + \'.txt\', \'w\') as fout:\n\
\t\tjson.dump(output, fout)\n\
\tprint(timeit.default_timer()-aa)\n\
")
	file.close()



def gen_pbs(filename, simfile):
	file = open("msi_files/" + filename + ".pbs", "w")
	file.write("#!/bin/bash\n\
#PBS -l nodes=1:ppn=1,pmem=2500mb,walltime=24:00:00\n\
#PBS -m abe\n\
#PBS -M kaoxx085@umn.edu\n\
cd /home/ennse/kaoxx085/fish/virsim\n\
module load python\n\
source bin/activate\n\
python msi_files/" + simfile + "$PBS_ARRAYID.py\n\
")
	file.close()


