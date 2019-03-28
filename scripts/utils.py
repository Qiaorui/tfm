import glob
import pandas as pd
import os
from tqdm import tqdm
import requests
import math
import zipfile
import numpy as np


BASE_URL = "https://s3.amazonaws.com/tripdata/"
BASE_PATTERN_NYC ="{}-citibike-tripdata.csv.zip"
BASE_PATTERN_JC = "JC-{}-citibike-tripdata.csv.zip"

DATE_RANGE = [y * 100 + m + 1 for y in range(2017, 2019) for m in range(12)]


def download_trip_data(dest_path):
    save_path = "raw_data/"
    if os.path.isdir(os.path.dirname(dest_path)):
        save_path = os.path.dirname(dest_path) + "/"

    for date in DATE_RANGE:
        file_path = save_path + BASE_PATTERN_JC.format(date)
        print(file_path.split(".zip")[0], end="")
        if os.path.isfile(file_path.split(".zip")[0]):
            print(" : FOUND")
        else:
            print()
            download(BASE_URL+BASE_PATTERN_JC.format(date), save_path=file_path)
            unzip(file_path)
            os.remove(file_path)

    for date in DATE_RANGE:
        file_path = save_path + BASE_PATTERN_NYC.format(date)
        print(file_path.split(".zip")[0], end="")
        if os.path.isfile(file_path.split(".zip")[0]):
            print(" : FOUND")
        else:
            print()
            download(BASE_URL+BASE_PATTERN_NYC.format(date), save_path=file_path)
            unzip(file_path)
            os.remove(file_path)


def download(url, save_path):
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    with open(save_path, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB',
                         unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")


def unzip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path))


def read_raw_trip_data(path):
    all_files = glob.glob(path)
    frame_list = []
    for f in tqdm(all_files, leave=False, unit="file", desc="Loading data"):
        df = pd.read_csv(f,
                         index_col=None,
                         header=0,
                         names=["Trip_Duration", "Start_Time", "Stop_Time", "Start_Station_ID", "Start_Station_Name",
                      "Start_Latitude", "Start_Longitude", "End_Station_ID", "End_Station_Name",
                      "End_Latitude", "End_Longitude", "Bike_ID", "User_Type", "Birth_Year", "Gender"],
                         usecols=["Trip_Duration", "Start_Time", "Stop_Time", "Start_Station_ID","Start_Latitude",
                                  "Start_Longitude", "End_Station_ID", "End_Latitude", "End_Longitude"],
                         na_values={"Start_Latitude":0,"Start_Longitude":0, "End_Latitude":0,"End_Longitude":0},
                         dtype={'End_Latitude': np.float32, 'End_Longitude': np.float32, 'End_Station_ID': np.float32,
                                'Start_Latitude': np.float32, 'Start_Longitude': np.float32,
                                'Start_Station_ID': np.float32, 'Trip_Duration': np.int32},
                         parse_dates=["Start_Time", "Stop_Time"]
                         )
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
        df = pd.read_csv(f, index_col=None, header=None,
                         names=["Datetime", "Condition", "Temperature", "Wind", "Humidity", "Visibility"])
        # Weather condition in String
        # Temperature in C
        # Wind Speed in kph
        # Humidity in %
        # Visibility in km
        frame_list.append(df)
    if not frame_list:
        return None
    df = pd.concat(frame_list, sort=True)

    print(len(df), "rows from weather data have been read")

    return df


def read_raw_location_data(path):
    all_files = glob.glob(path)
    frame_list = []
    for f in tqdm(all_files, leave=False, unit="file", desc="Loading data"):
        df = pd.read_csv(f,
                         index_col=None,
                         header=0,
                         names=["Trip_Duration", "Start_Time", "Stop_Time", "Start_Station_ID", "Start_Station_Name",
                                "Start_Latitude", "Start_Longitude", "End_Station_ID", "End_Station_Name",
                                "End_Latitude", "End_Longitude", "Bike_ID", "User_Type", "Birth_Year", "Gender"],
                         usecols=["Start_Time", "Start_Station_ID","Start_Latitude", "Start_Station_Name",
                                  "Start_Longitude"],
                         na_values={"Start_Latitude":0,"Start_Longitude":0},
                         dtype={'Start_Latitude': np.float32, 'Start_Longitude': np.float32,
                                'Start_Station_ID': np.float32},
                         parse_dates=["Start_Time"]
                         )
        frame_list.append(df)
    if not frame_list:
        return None
    df = pd.concat(frame_list, sort=True)
    print(len(df), "rows from trip data have been read")
    return df


def read_cleaned_location_data(path):
    all_files = glob.glob(path)
    frame_list = []
    for f in tqdm(all_files, leave=False, unit="file", desc="Loading data"):
        df = pd.read_csv(f,
                         index_col=None,
                         header=0,
                         #names=["Station_ID", "Station_Name", "Latitude", "Longitude", "First_Time", "Last_Time"],
                         dtype={'Latitude': np.float32, 'Longitude': np.float32, 'Station_ID': np.int16,},
                         parse_dates=["First_Time", "Last_Time"]
                         )
        frame_list.append(df)
    if not frame_list:
        return None
    df = pd.concat(frame_list, sort=True)
    print(len(df), "rows from trip data have been read")
    return df


def read_cleaned_trip_data(path):
    all_files = glob.glob(path)
    frame_list = []
    for f in tqdm(all_files, leave=False, unit="file", desc="Loading trip data"):
        df = pd.read_csv(f, index_col=None, header=0,
                         dtype={'End_Latitude': np.float32, 'End_Longitude': np.float32, 'End_Station_ID': np.int16,
                                'Start_Holiday': bool, 'Start_Hour': np.int8, 'Start_Latitude': np.float32,
                                'Start_Longitude': np.float32, 'Start_Month': np.int8, 'Start_Season': np.int8,
                                'Start_Station_ID': np.int16, 'Start_Weekday': np.int8, 'Start_Year': np.int16,
                                'Stop_Holiday': bool, 'Stop_Hour': np.int8, 'Stop_Month': np.int8, 'Stop_Season': np.int8,
                                'Stop_Weekday': np.int8, 'Stop_Year': np.int16, 'Trip_Duration': np.int32},
                         parse_dates=["Start_Time", "Stop_Time"]
                         )
        frame_list.append(df)
    if not frame_list:
        return None
    df = pd.concat(frame_list, sort=True)

    print(len(df), "rows from", path, "have been read")
    print("Columns:", list(df))

    return df


def read_cleaned_weather_data(path):
    all_files = glob.glob(path)
    frame_list = []
    for f in tqdm(all_files, leave=False, unit="file", desc="Loading weather data"):
        df = pd.read_csv(f, index_col=None, header=0,
                         dtype={"Condition": str, "Temperature": np.float32, "Wind": np.float32, "Humidity": np.int8,
                                "Visibility": np.float32},
                         parse_dates=["Datetime"]
                         )
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


def get_pickups(df):
    return df.loc[:, ['Start_Holiday', 'Start_Hour', 'Start_Latitude', 'Start_Longitude', 'Start_Month',
                         'Start_Season', 'Start_Station_ID', 'Start_Time', 'Start_Weekday', 'Start_Year']]


def get_dropoffs(df):
    return df.loc[:, ['End_Latitude', 'End_Longitude', 'End_Station_ID', 'Stop_Holiday', 'Stop_Hour', 'Stop_Month',
                          'Stop_Season', 'Stop_Time', 'Stop_Weekday', 'Stop_Year']]


def aggregate_by_time_slot(df, ts):
    print(df)

    print(ts)
    return None


def fill_weather_data(df, weather_df):

    return None


def get_station_list(df):
    stations = df[["Start_Station_ID", "Start_Latitude", "Start_Longitude"]].copy()
    stations.rename(
        columns={'Start_Station_ID': 'Station_ID', "Start_Latitude": "Latitude", "Start_Longitude": "Longitude"},
        inplace=True
    )
    df2 = df[["End_Station_ID", "End_Latitude", "End_Longitude"]].copy()
    df2.rename(
        columns={'End_Station_ID': 'Station_ID', "End_Latitude": "Latitude", "End_Longitude": "Longitude"},
        inplace=True
    )

    df2.drop_duplicates(inplace=True)

    stations = pd.concat([stations, df2], ignore_index=True)
    stations.drop_duplicates(inplace=True)
    stations.reset_index(inplace=True, drop=True)

    return stations


def get_start_station_dict(df):
    stations = df[["Start_Station_ID", "Start_Latitude", "Start_Longitude"]].copy()
    stations.rename(
        columns={'Start_Station_ID': 'Station_ID', "Start_Latitude": "Latitude", "Start_Longitude": "Longitude"},
        inplace=True
    )
    stations.drop_duplicates(inplace=True)
    stations.reset_index(inplace=True, drop=True)
    return stations.set_index('Station_ID').T.to_dict('list')
