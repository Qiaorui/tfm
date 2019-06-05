#!/bin/bash

TS_LIST=(30 60 120 180)
OUTLIER_THRESHOLD=720

ANALYZE_FLAG=0
MODEL_FLAG=0
JC_FLAG=0
NYC_FLAG=0

start_analyze () {
    echo Start Analyze

    if [[ $JC_FLAG = 1 ]]; then
        # JC analysis
        echo "  Start analyzing JC"
        python3 prepare.py -ot ${OUTLIER_THRESHOLD} -rt "raw_data/JC*tripdata.csv" -ct "cleaned_data/JC_trip_data.csv" > ./results/log.txt
        mkdir analysis_JC
        mv ./results/* analysis_JC
    fi

    if [[ $NYC_FLAG = 1 ]]; then
        # NYC analysis
        echo "  Start analyzing NYC"
        python3 prepare.py -ot ${OUTLIER_THRESHOLD} -rt "raw_data/201*tripdata.csv" -ct "cleaned_data/NYC_trip_data.csv" > ./results/log.txt
        mkdir analysis_NYC
        mv ./results/* analysis_NYC
    fi

    echo Finish Analyze
}

start_model () {
    python3 generate_model.py -ot ${OUTLIER_THRESHOLD} -ts 60 -ct "cleaned_data/JC_trip_data.csv" -start 2018-09-01 -th 2018-12-01
    python3 generate_model.py -ot ${OUTLIER_THRESHOLD} -ts 60 -ct "cleaned_data/NYC_trip_data.csv" -start 2018-09-01 -th 2018-12-01
}

if [ $# -ne 2 ]; then
    echo "Illegal number of parameters"
    exit
fi

if [[ $1 = "model" ]]; then
    MODEL_FLAG=1
fi
if [[ $1 = "analyze" ]]; then
    ANALYZE_FLAG=1
fi
if [[ $1 = "test" ]]; then
    ANALYZE_FLAG=1
    MODEL_FLAG=1
fi
if [[ $2 = "JC" ]]; then
    JC_FLAG=1
fi
if [[ $2 = "NYC" ]]; then
    NYC_FLAG=1
fi
if [[ $2 = "all" ]]; then
    JC_FLAG=1
    NYC_FLAG=1
fi


if [[ $ANALYZE_FLAG = 1 ]]; then
    start_analyze
fi
if [[ $MODEL_FLAG= 1 ]]; then
    start_model
fi

