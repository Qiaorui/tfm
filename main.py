from scripts import preprocess
from scripts import utils
from scripts import statistics
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-rw", default="raw_data/weather.csv", help="input raw weather data path")
    parser.add_argument("-cw", default="cleaned_data/weather.csv", help="input cleaned weather data path")
    parser.add_argument("-rt", default="raw_data/JC*tripdata.csv", help="input raw trip data path")
    parser.add_argument("-ct", default="cleaned_data/JC_trip_data.csv", help="input cleaned trip data path")

    parser.add_argument("-ot", type=int, help="Outlier threshold")
    parser.add_argument("-ts", type=int, default=30, help="Time slot for the aggregation, units in minute")

    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-s", action="store_true", help="print statistical report")

    args = parser.parse_args()

    weather_data_path = args.cw
    trip_data_path = args.ct
    raw_weather_data_path = args.rw
    raw_trip_data_path = args.rt

    pd.set_option('display.precision', 2)
    pd.set_option('display.max_columns', 500)

    # Read raw data and clean it
    # First check if cleaned data exists, if not, read from raw and save the cleaned version of it
    print("{0:*^80}".format(" Preprocess "))

    weather_data = utils.read_data(weather_data_path)
    if weather_data is None:
        print("No weather data found. Building from the raw data set...")
        preprocess.preprocess_weathers(raw_weather_data_path, weather_data_path)
        weather_data = utils.read_data(weather_data_path)
        assert weather_data is not None

    trip_data = utils.read_data(trip_data_path)
    if trip_data is None:
        print("No trip data found. Building from the raw data set...")
        if not preprocess.preprocess_trips(raw_trip_data_path, trip_data_path):
            print("No raw data found in the path, beginning downloading...")
            utils.download_trip_data(raw_trip_data_path)
            preprocess.preprocess_trips(raw_trip_data_path, trip_data_path)

        trip_data = utils.read_data(trip_data_path)
        assert trip_data is not None

    if args.ot is not None:
        print("Removing outlier with threshold", args.ot)
        preprocess.remove_trip_outlier(trip_data, args.ot)

    print("Breaking trip data to pick-up data and drop-off data")
    pick_ups, drop_offs = utils.break_up(trip_data)
    print("PICK UPS:")
    print(pick_ups.head(5))
    print("DROP OFF")
    print(drop_offs.head(5))

    if args.s:
        # Statistical analysis
        print("{0:*^80}".format(" Statistic Analysis "))

        print("{0:-^80}".format(" Weather Analysis "))
        #statistics.analyse_weather(weather_data, 2017)

        print("{0:-^80}".format(" Trip Analysis "))
        #statistics.analyse_trip_duration(trip_data)

        print("{0:-^80}".format(" Time Analysis "))
        #statistics.analyse_date_pattern(trip_data)

        print("{0:-^80}".format(" Geographic Analysis "))
        statistics.analyse_geo_pattern(trip_data)

    # Training modules, train data by different techniques
    print("{0:*^80}".format(" Training "))

        # Save model per each techniques

    # Evaluate the prediction
    print("{0:*^80}".format(" Evaluation "))


if __name__ == '__main__':
    main()
