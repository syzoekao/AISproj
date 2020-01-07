# Simulated boater movements

This folder contains 20 simulated boater movement networks predicted using XGBoost models. There are two types of .csv files: `lake_data.csv` and `boatsx.csv`

* `lake_data.csv`: has the information of 9182 lakes in Minnesota. The columns are dow number (`dow`), lake_name (`lake_name`), lake size in acres (`acre`), the lake coordinator (`utm_x` and `utm_y`), county of the lake (`county`), the name of the county (`county_name`), an indicator whether the lake is an inspected lake (`inspect`), an indicator whether the lake is infested with zebra mussels (`zm2019`), starry stonewort (`ss2019`), and eurasian watermilfoil (`ew2019`), respectively. The infestation status was determined by the Minnesota Department of Natural Resources (DNR) list of infested waters as of Nov 1, 2019. 

* `boatsx.csv`: there are 20 `.csv` files starting with "boats". Each file is a simulated boater movement network from the XGBoost models. Each file includes 3 columns: the origin lake (`dow_origin`), the destination lake (`dow_destination`), and the predicted number of boaters in a year (`weight`).  

