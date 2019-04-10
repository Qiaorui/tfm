from scripts import preprocess
from scripts import utils
from scripts import statistics
import pandas as pd
import argparse
import gc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-rw", default="raw_data/weather.csv", help="input raw weather data path")
    parser.add_argument("-cw", default="cleaned_data/weather.csv", help="input cleaned weather data path")
    parser.add_argument("-rt", default="raw_data/JC*tripdata.csv", help="input raw trip data path")
    parser.add_argument("-ct", default="cleaned_data/JC_trip_data.csv", help="input cleaned trip data path")
    parser.add_argument("-cs", default="cleaned_data/station_data.csv", help="input cleaned trip data path")

    parser.add_argument("-ot", type=int, help="Outlier threshold")

    parser.add_argument("-s", action="store_true", help="print statistical report")

    args = parser.parse_args()

    location_data_path = args.cs
    weather_data_path = args.cw
    trip_data_path = args.ct
    raw_weather_data_path = args.rw
    raw_trip_data_path = args.rt

    pd.set_option('display.precision', 4)
    pd.set_option('display.max_columns', 500)

    # Read raw data and clean it
    # First check if cleaned data exists, if not, read from raw and save the cleaned version of it
    print("{0:*^80}".format(" Preprocess "))

    weather_data = utils.read_cleaned_weather_data(weather_data_path)
    if weather_data is None:
        print("No weather data found. Building from  raw data set...")
        preprocess.preprocess_weathers(raw_weather_data_path, weather_data_path)
        weather_data = utils.read_cleaned_weather_data(weather_data_path)
        assert weather_data is not None

    station_data = utils.read_station_data(location_data_path)
    if station_data is None:
        print("No location data found. Beginning downloading...")
        utils.download_station_data(location_data_path)
        station_data = utils.read_station_data(location_data_path)
        assert station_data is not None

    trip_data = utils.read_cleaned_trip_data(trip_data_path)
    if trip_data is None:
        print("No trip data found. Building from raw data set...")
        if not preprocess.preprocess_trips(raw_trip_data_path, trip_data_path, station_data):
            print("No raw data found in the path, beginning downloading...")
            utils.download_trip_data(raw_trip_data_path)
            preprocess.preprocess_trips(raw_trip_data_path, trip_data_path, station_data)

        trip_data = utils.read_cleaned_trip_data(trip_data_path)
        assert trip_data is not None

    if args.ot is not None:
        print("Removing outlier with threshold", args.ot)
        preprocess.remove_trip_outlier(trip_data, args.ot)

    if args.s:
        # Statistical analysis
        print("{0:*^80}".format(" Statistic Analysis "))

        print("{0:-^80}".format(" Weather Analysis "))
        statistics.analyse_weather(weather_data, 2017)

        pick_ups = trip_data[['Start_Time']].copy()
        pick_ups.rename(columns={"Start_Time": "Timestamp"}, inplace=True)
        start = pick_ups["Timestamp"].min().replace(hour=0, minute=0, second=0)
        end = pick_ups["Timestamp"].max().replace(hour=23, minute=59)

        index = pd.date_range(start=start, end=end, freq='30Min')

        pick_ups = utils.aggregate_by_time_slot(pick_ups, 30, index)
        pick_ups = utils.fill_weather_data(pick_ups, weather_data)

        statistics.analyse_weather_trip(pick_ups)
        exit(1)
        print("{0:-^80}".format(" Demographic Analysis "))
        statistics.analyse_demographic_pattern(raw_trip_data_path)

        print("{0:-^80}".format(" Trip Analysis "))
        statistics.analyse_trip_duration(trip_data)

        print("{0:-^80}".format(" Time Analysis "))
        statistics.analyse_date_pattern(trip_data)

        print("{0:-^80}".format(" Geographic Analysis "))
        statistics.plot_stations(station_data)
        statistics.show_station_change(raw_trip_data_path, station_data, trip_data)
        statistics.analyse_geo_pattern(trip_data)
        statistics.plot_unbalance_network(trip_data)


if __name__ == '__main__':
    main()
