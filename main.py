from scripts import preprocess
from scripts import utils
from scripts import  statistics


def main():

    weather_data_path = "cleaned_data/weather.csv"
    trip_data_path = "cleaned_data/JC_trip_data.csv"
    raw_weather_data_path = "raw_data/weather.csv"
    raw_trip_data_path = "raw_data/JC*tripdata.csv"

    # Read raw data and clean it
    # First check if cleaned data exists, if not, read from raw and save the cleaned version of it
    print("{0:*^60}".format(" Preprocess "))

    weather_data = utils.read_data(weather_data_path)
    if weather_data is None:
        print("No weather data found. Building from the raw data set...")
        preprocess.preprocess_weathers(raw_weather_data_path, weather_data_path)
        weather_data = utils.read_data(weather_data_path)
        assert weather_data is not None

    trip_data = utils.read_data(trip_data_path)
    if trip_data is None:
        print("No trip data found. Building from the raw data set...")
        preprocess.preprocess_trips(raw_trip_data_path, trip_data_path)
        trip_data = utils.read_data(trip_data_path)
        assert trip_data is not None

    # Statistical analysis
    print("{0:*^60}".format(" Statistic Analysis "))
    statistics.analyse_weather(weather_data, 2017)

    print("{0:-^60}".format(" Weather Analysis "))
    statistics.analyse_trip(trip_data)

    print("{0:-^60}".format(" Trip Analysis "))

    # Training modules, train data by different techniques
    print("{0:*^60}".format(" Training "))

        # Save model per each techniques

    # Evaluate the prediction
    print("{0:*^60}".format(" Evaluation "))


if __name__ == '__main__':
    main()
