import glob
import pandas as pd
import os
from tqdm import tqdm
import requests
import math


BASE_URL = "https://s3.amazonaws.com/tripdata/"
BASE_PATTERN_NYC ="{}-citibike-tripdata.csv.zip"
BASE_PATTERN_JC = "JC-{}-citibike-tripdata.csv.zip"

DATE_RANGE = [y * 100 + m + 1 for y in range(2017, 2019) for m in range(12)]


def download_trip_data(dest_path):
    save_path = "raw_data"
    if os.path.isdir(os.path.dirname(dest_path)):
        save_path = dest_path

    for date in DATE_RANGE:
        file_path = save_path + BASE_PATTERN_JC.format(date)
        print(file_path, end="")
        if os.path.isfile(file_path.split(".zip")[0]):
            print(" : FOUND")
        else:
            download(BASE_URL+BASE_PATTERN_JC.format(date), save_path=file_path)


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
