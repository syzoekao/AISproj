# File description

This provides file description for: [UtilFunction.py](https://github.com/syzoekao/AISproj/blob/master/lib/python3.7/site-packages/AISsim/UtilFunction.py) and [PostAnalysisClean.py](https://github.com/syzoekao/AISproj/blob/master/lib/python3.7/site-packages/AISsim/PostAnalysisClean.py). 

## [UtilFunction.py](https://github.com/syzoekao/AISproj/blob/master/lib/python3.7/site-packages/AISsim/UtilFunction.py)

* This file contains three primary simulation functions: [infest_outcome_func](https://github.com/syzoekao/AISproj/blob/8c6de5bdb1666b7318623f228fb63ad91d65683a/lib/python3.7/site-packages/AISsim/UtilFunction.py#L11-L253), [pre_infest_outcome_func](https://github.com/syzoekao/AISproj/blob/master/lib/python3.7/site-packages/AISsim/UtilFunction.py#L279-L482), and [Scenario](https://github.com/syzoekao/AISproj/blob/master/lib/python3.7/site-packages/AISsim/UtilFunction.py#L498-L762). 
* `infest_outcome_func()`: performed Bayesian calibration process to obtain the posterior parameter sets. This function is **NOT** used for risk simulation for each lake.
* `pre_infest_outcome_func()`: simulated AIS invasion across lakes via boater movements and water connectivity from the begining of 2013 to the end of 2018. This function took the posterior parameters from the Bayesian calibration process to do the simulations. The results were used to validate that the posterior parameters reproduced the acceptable trends of number of infested lakes over the observed years, 2013-2018. This is not needed for predicting lake risk after 2019. 
* `Scenario()`: simulated AIS invasion across lakes under different scenarios, including: StatusQuo, Education, Penalty, MendDecon (mendatory decontamination), and ReduceTraffic (reduction in boater traffic). 
* We use 100 `Python` scripts to run post-calibration simulations. The example python simulation script is this file, [simulation1.py](https://github.com/syzoekao/AISproj/blob/master/simulation1.py). Each simulation python script would generate 100 simulation runs. Therefore, there are 10,000 simulation runs in total. 
* The example results files are in [`results/`](https://github.com/syzoekao/AISproj/tree/master/results) from one simulation file, [simulation1.py](https://github.com/syzoekao/AISproj/blob/master/simulation1.py). 


Here are detailed descriptions of `pre_infest_outcome_func()` and `Scenario()`. 

### `pre_infest_outcome_func()`

* This function aims to reproduce the number of infested lakes in 2013-2018. The seeded zebra mussel infested lakes were the infested lakes at the end of 2012 (53 lakes). However, the seeded starry stonewort infested lakes were the infested lakes at the end of 2016 (8 lakes). 
* In the file `simulation1.py`, we ran the function first [here](https://github.com/syzoekao/AISproj/blob/c8ffe0b01ddc6b996c4e713d7ae4a2386e37a3f6/simulation1.py#L88-L95). 
* The results that got saved are these three lists: 

    + `pre_ann_out`: This list has 8 elements for each simulation run. The first 6 elements are the number of zebra mussel infested lakes at the year end from 2013 to 2018. The last two elements are the number of starry stonewort infested lakes at the end of 2017 and 2018. 
    + `pre_res_zm`: Each simulation outputed a python dictionary recording *zebra mussel* infested lakes over the 6 years (2013-2018) of simulation. The dictionary has 4 keys (`id`, `time`, `boat`, and `river`) and each key has a corresponding list of values. The length of each list across the four keys should be the same. The list under the key `id` inclucdes the ids of infested lakes. The list under the key `time` represents the year of infestation corresponding to the lake id at the same location in the list under the key `id`. The list under the key `boat` represents whether the lake was infested via boater movements (= 1) or river connection (= 0). The list under the key `river` represents whether the lake was infested via river connection (= 1) or boater connection (= 0). 
    + `pre_res_ss`: Each simulation outputed a python dictionary recording *starry stonewort* infested lakes over the 2 years (2017-2018) of simulation. The dictionary has 4 keys (`id`, `time`, `boat`, and `river`) and each key has a corresponding list of values. The length of each list across the four keys should be the same. The list under the key `id` inclucdes the ids of infested lakes. The list under the key `time` represents the year of infestation corresponding to the lake id at the same location in the list under the key `id`. The list under the key `boat` represents whether the lake was infested via boater movements (= 1) or river connection (= 0). The list under the key `river` represents whether the lake was infested via river connection (= 1) or boater connection (= 0). 


### `Scenario()`

* This function simulated AIS invasion across lakes from 2019 to 2025 under different scenarios: StatusQuo, Education, Penalty, MendDecon (mendatory decontamination), and ReduceTraffic (reduction in boater traffic). 
* The results of this function are used to calculate the risk of AIS invasion for each lake. 
* We seeded the infested lakes using the reported infested lakes at the end of 2018 by zebra mussel and starry stonewort. 
* The simulations were performed in this chunk of [code](https://github.com/syzoekao/AISproj/blob/c8ffe0b01ddc6b996c4e713d7ae4a2386e37a3f6/simulation1.py#L104-L162). 
* We appended results under different scenarios by zebra mussel infested lakes and starry stonewort infested lakes in this part of [code](https://github.com/syzoekao/AISproj/blob/c8ffe0b01ddc6b996c4e713d7ae4a2386e37a3f6/simulation1.py#L164-L171). 
* We save the following results for each simulation: 

    + `scenario_ann_xx`: A dictionary stored lists of the number of infested lakes by year under different scenario (key): StatusQuo (S), Education (E), Penalty (P), MendDecon (E), and ReduceTraffic (T). `xx` could be `zm` (zebra mussel) or `ss` (starry stonewort). 
    + `scenario_res_xx`: A nested dictionary with two levels. The first level of keys are the scenarios: `"S"`, `"E"`, `"P"`, `"E"`, and `"T"`. Under each scenario key, there is a dictionary about infested lakes over years, including `id` with a list of lake ids, `time` with a list of infested year of the corresponding lake id, `boat` with a list of 0s and 1s indicating whether the corresponding infested lake was infested via boater movement (= 1) or river connection (= 0), and `river` with a list of 0s and 1s indicating whether the corresponding infested lake was infested via boater movement (= 0) or river connection (= 1). `xx` could be `zm` (zebra mussel) or `ss` (starry stonewort). 


## [PostAnalysisClean.py](https://github.com/syzoekao/AISproj/blob/master/lib/python3.7/site-packages/AISsim/PostAnalysisClean.py)

* The lake specific risk is calculated as the number of times a lake was infested by a species divided by the number of simulations (10,000). 
* The final results are stored in these two csv files: `'risk table (zm)(new).csv'` and `'risk table (ss)(new).csv'`





