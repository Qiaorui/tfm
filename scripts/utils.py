import glob
import pandas as pd


def read_raw_trip_data(path):
    all_files = glob.glob(path)
    frame_list = []
    for f in all_files:
        df = pd.read_csv(f, index_col=None, header=0)
        df.columns = ["Trip_Duration", "Start_Time", "Stop_Time", "Start_Station_ID", "Start_Station_Name",
                      "Start_Latitude", "Start_Longitude", "End_Station_ID", "End_Station_Name",
                      "End_Latitude", "End_Longitude", "Bike_ID", "User_Type", "Birth_Year", "Gender"]
        frame_list.append(df)
    if not frame_list:
        return None
    df = pd.concat(frame_list, sort=True)

    print(len(df), "rows from trip data have been read")

    return df


def read_raw_weather_data(path):
    all_files = glob.glob(path)
    frame_list = []
    for f in all_files:
        df = pd.read_csv(f, index_col=None, header=None)
        # Weather condition in String
        # Temperature in C
        # Wind Speed in kph
        # Humidity in %
        # Visibility in km
        df.columns = ["Datetime", "Condition", "Temperature", "Wind", "Humidity", "Visibility"]
        frame_list.append(df)
    if not frame_list:
        return None
    df = pd.concat(frame_list, sort=True)

    print(len(df), "rows from weather data have been read")

    return df


def read_data(path):
    all_files = glob.glob(path)
    frame_list = []
    for f in all_files:
        df = pd.read_csv(f, index_col=None, header=0)
        frame_list.append(df)
    if not frame_list:
        return None
    df = pd.concat(frame_list, sort=True)

    print(len(df), "rows from", path, "have been read")
    print("Columns:", list(df))

    return df


def break_up(df):
    pickups = df.loc[:, ['Start_Holiday', 'Start_Hour', 'Start_Latitude', 'Start_Longitude', 'Start_Month',
                         'Start_Season', 'Start_Station_ID', 'Start_Time', 'Start_Weekday', 'Start_Year']]
    dropoffs = df.loc[:, ['End_Latitude', 'End_Longitude', 'End_Station_ID', 'Stop_Holiday', 'Stop_Hour', 'Stop_Month',
                          'Stop_Season', 'Stop_Time', 'Stop_Weekday', 'Stop_Year']]

    return pickups, dropoffs
