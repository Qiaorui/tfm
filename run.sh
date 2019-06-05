#!/bin/bash

TS_LIST=(30 60 120 180)
OUTLIER_THRESHOLD=720

start_analyze () {
    echo Start Analyze

    # JC analysis
    echo "  Start analyzing JC"
    python3 prepare.py -ot ${OUTLIER_THRESHOLD} -rt "raw_data/JC*tripdata.csv" -ct "cleaned_data/JC_trip_data.csv" > ./results/log.txt
    mkdir analysis_JC
    mv ./results/* analysis_JC

    # NYC analysis
    echo "  Start analyzing NYC"
    python3 prepare.py -ot ${OUTLIER_THRESHOLD} -rt "raw_data/201*tripdata.csv" -ct "cleaned_data/NYC_trip_data.csv" > ./results/log.txt
    mkdir analysis_NYC
    mv ./results/* analysis_NYC

    echo Finish Analyze
}

start_test () {
    python3 generate_model.py -ot 720 -ts 60 -ct cleaned_data/JC_trip_data.csv -start 2018-09-01 -th 2018-12-01
    python3 generate_model.py -ot 720 -ts 60 -ct cleaned_data/NYC_trip_data.csv -start 2018-09-01 -th 2018-12-01
}

if [[ $1 = "test" ]]; then
    start_test
fi
if [[ $1 = "analyze" ]]; then
    start_analyze
fi