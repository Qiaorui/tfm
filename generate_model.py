from scripts import preprocess
from scripts import utils
import pandas as pd
import gc
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cw", default="cleaned_data/weather.csv", help="input cleaned weather data path")
    parser.add_argument("-ct", default="cleaned_data/JC_trip_data.csv", help="input cleaned trip data path")
    parser.add_argument("-ot", type=int, help="Outlier threshold")
    parser.add_argument("-ts", type=int, default=30, help="Time slot for the aggregation, units in minute")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")

    args = parser.parse_args()

    weather_data_path = args.cw
    trip_data_path = args.ct

    pd.set_option('display.precision', 3)
    pd.set_option('display.max_columns', 500)

    weather_data = utils.read_cleaned_weather_data(weather_data_path)
    assert weather_data is not None

    trip_data = utils.read_cleaned_trip_data(trip_data_path)
    assert trip_data is not None

    if args.ot is not None:
        print("Removing outlier with threshold", args.ot)
        preprocess.remove_trip_outlier(trip_data, args.ot)

    print("{0:*^80}".format(" Prepare training data "))
    # Remove trips which contains sink station
    start_stations_ids = list(utils.get_start_station_dict(trip_data).keys())
    trip_data = trip_data[trip_data.End_Station_ID.isin(start_stations_ids)]

    print("Breaking trip data to pick-up data and drop-off data")
    # pick_ups, drop_offs = utils.break_up(trip_data)

    pick_ups = trip_data['Start_Station_ID', 'Start_Time'].copy()
    drop_offs = trip_data['End_Station_ID', 'Stop_Time'].copy()

    del trip_data
    gc.collect()

    pick_ups = utils.aggregate_by_time_slot(pick_ups, args.ts)
    pick_ups = utils.fill_weather_data(pick_ups, weather_data)


    # PCA

    # Training modules, train data by different techniques
    print("{0:*^80}".format(" Training "))

    # Save model per each techniques

    # Evaluate the prediction
    print("{0:*^80}".format(" Evaluation "))


if __name__ == '__main__':
    main()