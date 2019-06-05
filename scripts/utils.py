import glob
import pandas as pd
import os
from tqdm import tqdm
import requests
import math
import zipfile
import numpy as np
import urllib.request
import json
import certifi
import glob
import re

BASE_URL = "https://s3.amazonaws.com/tripdata/"
BASE_PATTERN_NYC = "{}-citibike-tripdata.csv.zip"
BASE_PATTERN_JC = "JC-{}-citibike-tripdata.csv.zip"
STATION_URL = "https://feeds.citibikenyc.com/stations/stations.json"

DATE_RANGE = [y * 100 + m + 1 for y in range(2017, 2019) for m in range(12)]
#DATE_RANGE += [201901, 201902]


def download_station_data(dest_path):
    with urllib.request.urlopen(STATION_URL, cafile=certifi.where()) as url:
        data = json.loads(url.read().decode())
        df = pd.DataFrame(data.get("stationBeanList"))
        df = df.replace('', np.nan)
        df.dropna(axis='columns', inplace=True)
        df.drop(
            df.columns.difference(['id', 'latitude', 'longitude', 'stationName', 'totalDocks']),
            axis=1,
            inplace=True
        )
        df.rename(
            columns={'id': 'Station_ID', "latitude": "Latitude", "longitude": "Longitude",
                     "stationName": "Station_Name", "totalDocks": "Docks"},
            inplace=True
        )
        df.to_csv(dest_path, index=False)
        print(dest_path, "created")


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
            download(BASE_URL + BASE_PATTERN_JC.format(date), save_path=file_path)
            unzip(file_path)
            os.remove(file_path)

    for date in DATE_RANGE:
        file_path = save_path + BASE_PATTERN_NYC.format(date)
        print(file_path.split(".zip")[0], end="")
        if os.path.isfile(file_path.split(".zip")[0]):
            print(" : FOUND")
        else:
            print()
            download(BASE_URL + BASE_PATTERN_NYC.format(date), save_path=file_path)
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
                         usecols=["Trip_Duration", "Start_Time", "Stop_Time", "Start_Station_ID", "Start_Latitude",
                                  "Start_Longitude", "End_Station_ID", "End_Latitude", "End_Longitude"],
                         na_values={"Start_Latitude": 0, "Start_Longitude": 0, "End_Latitude": 0, "End_Longitude": 0},
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
                         usecols=["Start_Station_ID", "Start_Latitude", "Start_Station_Name", "Start_Longitude"],
                         na_values={"Start_Latitude": 0, "Start_Longitude": 0},
                         dtype={'Start_Latitude': np.float32, 'Start_Longitude': np.float32,
                                'Start_Station_ID': np.float32}
                         )
        frame_list.append(df)
    if not frame_list:
        return None
    df = pd.concat(frame_list, sort=True)
    print(len(df), "rows from trip data have been read")
    return df


def read_raw_demographic_data(path):
    all_files = glob.glob(path)
    frame_list = []
    for f in tqdm(all_files, leave=False, unit="file", desc="Loading data"):
        df = pd.read_csv(f,
                         index_col=None,
                         header=0,
                         names=["Trip_Duration", "Start_Time", "Stop_Time", "Start_Station_ID", "Start_Station_Name",
                                "Start_Latitude", "Start_Longitude", "End_Station_ID", "End_Station_Name",
                                "End_Latitude", "End_Longitude", "Bike_ID", "User_Type", "Birth_Year", "Gender"],
                         usecols=["Start_Time", "Trip_Duration", "User_Type", "Birth_Year", "Gender"],
                         na_values={"Gender": 0},
                         parse_dates=["Start_Time"]
                         )
        frame_list.append(df)
    if not frame_list:
        return None
    df = pd.concat(frame_list, sort=True)

    print(len(df), "rows from trip data have been read")

    return df


def read_station_data(path):
    all_files = glob.glob(path)
    frame_list = []
    for f in all_files:
        df = pd.read_csv(f,
                         index_col=None,
                         header=0,
                         dtype={'Latitude': np.float32, 'Longitude': np.float32, 'Station_ID': np.int16,
                                'Docks': np.int8}
                         )
        frame_list.append(df)
    if not frame_list:
        return None
    df = pd.concat(frame_list, sort=True)
    print(len(df), "rows from station data have been read")
    print("Columns:", list(df))
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
                                'Stop_Holiday': bool, 'Stop_Hour': np.int8, 'Stop_Month': np.int8,
                                'Stop_Season': np.int8,
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
    for f in all_files:
        df = pd.read_csv(f, index_col=None, header=0,
                         dtype={"Condition": str, "Temperature": np.float16, "Wind": np.float16, "Humidity": np.int8,
                                "Visibility": np.float16},
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


def aggregate_by_time_slot(df, ts, index):
    df = df.groupby([pd.Grouper(key="Timestamp", freq=str(ts)+'Min')]).size().reset_index(name="Count")
    df = df.set_index("Timestamp")
    df = df.reindex(index, fill_value=0)
    return df


def fill_weather_data(df, weather_df):
    idx = np.searchsorted(weather_df.Datetime, df.index) - 1
    df["Temperature"] = weather_df.iloc[idx].Temperature.values
    df["Wind"] = weather_df.iloc[idx].Wind.values
    df["Humidity"] = weather_df.iloc[idx].Humidity.values
    df["Visibility"] = weather_df.iloc[idx].Visibility.values
    df["Condition"] = weather_df.iloc[idx].Condition.values
    return df


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


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (
                1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))


def closest(v, data):
    if np.isnan(v['Latitude']):
        res = next((item for item in data if item["Station_ID"] == v["Station_ID"]), {"Station_ID": np.nan})
        v["Distance"] = np.nan
    else:
        res = min(data, key=lambda p: distance(v['Latitude'], v['Longitude'], p['Latitude'], p['Longitude']))
        v["Distance"] = distance(res["Latitude"], res["Longitude"], v['Latitude'], v['Longitude'])
    v["Closest_Station_ID"] = res["Station_ID"]
    return v


def get_next_filename(prefix):
    all_files = glob.glob("results/" + prefix + "[0-9]*.pdf")
    all_numbers = [int(re.search(r'\d+', name).group()) for name in all_files]

    return prefix + str(max(all_numbers)+1) if all_numbers else prefix + "1"