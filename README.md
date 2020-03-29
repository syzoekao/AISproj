# AISproj
Predictive boater movements and simulation of lake-level risk of infestation in MN

### Post-calibration simulation

The file [simulation1.py](https://github.com/syzoekao/AISproj/blob/master/simulation1.py) is an example file used to simulate the spread of AIS across lakes in MN. There were 100 simulation files to run 100 simulations per file. The data used in this file are provided in [data](https://github.com/syzoekao/AISproj/tree/master/data). 

Data include: 

1. [Lake attributes](https://github.com/syzoekao/AISproj/blob/master/data/lake_attribute.csv)

It is important to note that all the lakes followed the id column instead of the dow column in the simulation. I converted the lakes from the id to dow in the analysis after simulations. 

2. [Zebra mussel infested lakes](https://github.com/syzoekao/AISproj/blob/master/data/zm_dow.csv) and [starry stonewort infested lakes](https://github.com/syzoekao/AISproj/blob/master/data/ss_dow.csv)

This is based on the infestation status in the end of 2018. 

3. Dictionary for boater movement

There are 20 samples of the boater movement dictionary. The boater movements are organizaed as nested dictionary. An example is 

{"0": {"1": 0.4615, "2": 0.1923, "3": 0.1211}} 

The first key "0" is the origin lake according to the id assigned in the `lake_attribute.csv` file. The second keys "1", "2", and "3" are the destination lakes (following the id assigned in the `lake_attribute.csv` too). The values 0.4615, 0.1923, and 0.1211 are the weekly number of boaters traveling from origin lake ("0") to the destination lakes ("1", "2", "3"). 

4. [River network](https://github.com/syzoekao/AISproj/blob/master/data/river_net_sim.csv)

This is organized as an edge list. The primary columns used are the `origin`, `destination`, and `weight`. Similarly, the ids used in the columns `origin` and `destination` are based on the id column in the `lake_attribute.csv`. 

5. Calibration parameters

The parameters used in the simulation model. The data were originally organized in [numpy array](https://github.com/syzoekao/AISproj/blob/master/data/param_sample.npy), but there is a csv version [here](https://github.com/syzoekao/AISproj/blob/master/data/param_sample.csv). The parameters were calibrated using Metropolis-Hastings Markov chain Monte Carlo so there are repeated values. 





