#!/bin/bash

TS_LIST=(180 120 60 30)
START_LIST=(2018-09-01 2018-06-01 2017-12-01 2018-08-01 2018-05-01 2017-11-01)
TH_LIST=(2018-12-01 2018-11-01)


OUTLIER_THRESHOLD=720

ANALYZE_FLAG=0
MODEL_FLAG=0
JC_FLAG=0
NYC_FLAG=0

start_analyze () {
    echo "Start Analyze"

    if [[ $JC_FLAG = 1 ]]; then
        # JC analysis
        echo "----Start analyzing JC"
        python3 prepare.py -ot ${OUTLIER_THRESHOLD} -rt "raw_data/JC*tripdata.csv" -ct "cleaned_data/JC_trip_data.csv" > ./results/log.txt
        mkdir -p analysis_JC
        mv ./results/* analysis_JC
    fi

    if [[ $NYC_FLAG = 1 ]]; then
        # NYC analysis
        echo "----Start analyzing NYC"
        python3 prepare.py -ot ${OUTLIER_THRESHOLD} -rt "raw_data/201*tripdata.csv" -ct "cleaned_data/NYC_trip_data.csv" > ./results/log.txt
        mkdir -p analysis_NYC
        mv ./results/* analysis_NYC
    fi

    echo "Finish Analyze"
}

start_model () {
    echo "Start Modelling"

    if [[ $JC_FLAG = 1 ]]; then
        echo "----Start modelling JC"
        for ts in ${TS_LIST[@]}
        do
            for ((i=0; i<${#TH_LIST[@]}; i++));
            do
                for ((j=3*i; j<${#START_LIST[@]} + ((i-1)*3); j++));
                do
                    echo "--------Start the case -ot ${OUTLIER_THRESHOLD} -ts $ts -ct "cleaned_data/JC_trip_data.csv" -start ${START_LIST[$j]} -th ${TH_LIST[$i]}"
                    python3 -u generate_model.py -ot ${OUTLIER_THRESHOLD} -ts $ts -ct "cleaned_data/JC_trip_data.csv" -start ${START_LIST[$j]} -th ${TH_LIST[$i]} | tee ./results/log.txt
                    mkdir -p "JC_${ts}_${START_LIST[$j]}_${TH_LIST[$i]}"
                    mv ./results/* "JC_${ts}_${START_LIST[$j]}_${TH_LIST[$i]}"
                done
            done
        done
    fi

    if [[ $NYC_FLAG = 1 ]]; then
        echo "----Start modelling NYC"
        for ts in ${TS_LIST[@]}
        do
            for ((i=0; i<${#TH_LIST[@]}; i++));
            do
                for ((j=3*i; j<${#START_LIST[@]} + ((i-1)*3); j++));
                do
                    echo "--------Start the case -ot ${OUTLIER_THRESHOLD} -ts $ts -ct "cleaned_data/JC_trip_data.csv" -start ${START_LIST[$j]} -th ${TH_LIST[$i]}"
                    python3 -u generate_model.py -ot ${OUTLIER_THRESHOLD} -ts $ts -ct "cleaned_data/NYC_trip_data.csv" -start ${START_LIST[$j]} -th ${TH_LIST[$i]} | tee ./results/log.txt
                    mkdir -p "NYC_${ts}_${START_LIST[$j]}_${TH_LIST[$i]}"
                    mv ./results/* "NYC_${ts}_${START_LIST[$j]}_${TH_LIST[$i]}"
                done
            done
        done
    fi

    echo "Finish Modelling"
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

mkdir -p results
if [[ $ANALYZE_FLAG = 1 ]]; then
    start_analyze
fi
if [[ $MODEL_FLAG = 1 ]]; then
    start_model
fi

