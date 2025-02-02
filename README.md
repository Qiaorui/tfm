# Final Master Thesis (Analysing and Predicting User Demand in Vehicle-Sharing System)

The repository for the Final Master Thesis. Which I want to analyze and predict user demand in vehicle sharing system, particularly the case of New York Citi Bike.

**Shared transport** is an economical and sustainable mode of urban mobility. It occupies very important position in the mobility market nowadays. Given the growing market size and importance of vehicle-sharing system, we want to further analyse the urban mobility patterns based on large datasets containing users' mobility information. Thus it gives us a better understanding about urban mobility and the possibility to improve some mobility issues.

One of the issues we want to discuss in this thesis is the 
user demand-driven rebalancing problem. Given the demand-driven nature of shared transport systems, availability of its infrastructure heavily depends on users’ mobility patterns. An unbalanced situation where the number of available vehicles cannot meet user demand will reduce the efficiency of vehicle-sharing system. We want to design a machine learning approach to predict the user demand using historical trip data, weather data and geographic data. 

Finally we will build a smart management system by tracking every shared vehicle in real-time and and propose a user demand prediction model based on our learning model and future weather forecast. The resulting outcomes should also provide insights which assist Shared Transport providers to make more appropriate decisions.


## Getting Started

```bash
# clone the project
git https://github.com/Qiaorui/tfm.git

cd tfm

# install package
bash install.sh

# run script with all test case
bash run.sh test all
```
In the **run.sh** you have two arguments. The first you have select the operation which be one of [model, analyze, test]. The second you have to choose which database you want to perform the operation, it can be one of [JC, NYC, all].

I provide two main scripts:
```
usage: prepare.py [-h] [-rw RW] [-cw CW] [-rt RT] [-ct CT] [-cs CS] [-ot OT]
                  [-s]

optional arguments:
  -h, --help  show this help message and exit
  -rw RW      input raw weather data path
  -cw CW      input cleaned weather data path
  -rt RT      input raw trip data path
  -ct CT      input cleaned trip data path
  -cs CS      input cleaned trip data path
  -ot OT      Outlier threshold
  -s          plot statistical report
```

```
usage: generate_model.py [-h] [-cw CW] [-ct CT] [-ot OT] [-ts TS]
                         [-start START] [-th TH] [-s]

optional arguments:
  -h, --help    show this help message and exit
  -cw CW        input cleaned weather data path
  -ct CT        input cleaned trip data path
  -ot OT        Outlier threshold
  -ts TS        Time slot for the aggregation, units in minute
  -start START  Input start date
  -th TH        Threshold datetime to split train and test dataset
  -s            plot statistical report
```
> Note: if any error happens please check prerequisites first.

### Prerequisites

You need python 3.7 in your system.

The package are listed in **requirements.txt**, you can easily install by

```bash
bash install.sh
```

## Architecture

```
.
├── README.md                       
├── cleaned_data                    Directory for cleaned data
├── generate_model.py               Second step: Data preprocessing, Predictive Modelling, Evaluation
├── install.sh                      Prepared one-key script for install all required python packages
├── prepare.py                      First step: Data Collection, Data Cleaning, Data Visualisation
├── raw_data                        Directory for raw data
├── related_references              Directory for related references which are not so useful
├── requirements.txt
├── results                         Directory for temporal saved results
├── run.sh                          Prepared one-key script to run First and Second step
├── scripts
│   ├── judge.py
│   ├── models.py
│   ├── mySSA.py
│   ├── preprocess.R
│   ├── preprocess.py
│   ├── statistics.py
│   ├── utils.py
│   └── weather_scrapper.rb         Script to scrap weahter data
├── test.ipynb                      Notebook for the SSA dimension selection
└── useful_references               Directory for useful references, important!
```

## Development Methodology

The methodology we are going to follow is Agile-like method. We publish **bugs**, **tasks** and **issues** as cards in [project panel](https://github.com/Qiaorui/zooli/projects/1). Then each member has to pick his task and drag into corresponding state column.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [repository tags](https://github.com/Qiaorui/zooli/tags).
