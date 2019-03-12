from . import utils
import subprocess
import os
import pandas as pd
import numpy as np
from geopy import distance
from tqdm import tqdm

from pandas.tseries.holiday import USFederalHolidayCalendar


def preprocess_trips(raw_path, dest_path):
    df = utils.read_raw_trip_data(raw_path)

    if df is None:
        return False

    print("Dropping columns :", ["Bike_ID", "Birth_Year", "User_Type", "Gender", "Start_Station_Name", "End_Station_Name"])
    df.drop(["Bike_ID", "Birth_Year", "User_Type", "Gender", "Start_Station_Name", "End_Station_Name"], 1, inplace=True)
    df["Start_Longitude"] = df.Start_Longitude.replace({0 : np.nan})
    df["Start_Latitude"] = df.Start_Latitude.replace({0: np.nan})
    df["End_Longitude"] = df.End_Longitude.replace({0: np.nan})
    df["End_Latitude"] = df.End_Latitude.replace({0: np.nan})

    print(df.describe())
    print(df.isnull().sum())
    print('Contains ', sum(df.isnull().sum()), ' NULL values.')
    print('Dropping null or incorrect rows')
    complete_size = len(df.index)
    df.dropna(subset=["Start_Time", "Stop_Time"], inplace=True)
    df.dropna(how='all', subset=["Start_Station_ID", "Start_Latitude", "Start_Longitude"], inplace=True)
    df.dropna(how='all', subset=["End_Station_ID", "End_Latitude", "End_Longitude"], inplace=True)

    stations = get_station_list(df)
    complete_station(df, stations)

    df.dropna(inplace=True)

    print("Removed", complete_size - len(df.index), "rows")

    df.loc[:, 'Trip_Duration'] = (df.loc[:, 'Trip_Duration'] / 60).apply(np.ceil)

    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['Stop_Time'] = pd.to_datetime(df['Stop_Time'])

    df['Start_Hour'] = df['Start_Time'].dt.hour
    df['Start_Weekday'] = df['Start_Time'].dt.weekday + 1
    df['Start_Month'] = df['Start_Time'].dt.month
    df['Start_Season'] = df['Start_Time'].dt.quarter
    df['Start_Year'] = df['Start_Time'].dt.year

    df['Stop_Hour'] = df['Stop_Time'].dt.hour
    df['Stop_Weekday'] = df['Stop_Time'].dt.weekday + 1
    df['Stop_Month'] = df['Stop_Time'].dt.month
    df['Stop_Season'] = df['Stop_Time'].dt.quarter
    df['Stop_Year'] = df['Stop_Time'].dt.year

    tqdm.pandas(desc="calculating distances")
    df['Distance'] = df.apply(calculate_distance, axis=1)

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df['Start_Time'].min(), end=df['Start_Time'].max())
    print('From {} until {}'.format(df['Start_Time'].min(), df['Start_Time'].max()))

    df['Start_Holiday'] = df['Start_Time'].dt.normalize().isin(holidays) #| (df['Start_Weekday'] == 7)
    df['Stop_Holiday'] = df['Stop_Time'].dt.normalize().isin(holidays)

    print(len(df[df['Start_Holiday'] == True]), 'trips done during holidays')

    print(df.describe())

    df.to_csv(dest_path, index=False)

    return True


def preprocess_weathers(raw_path, dest_path):
    df = utils.read_raw_weather_data(raw_path)

    print(df.isnull().sum())
    print('Contains ', sum(df.isnull().sum()), ' NULL values. Need to impute :', sum(df.isnull().sum()) > 0)

    target_script = os.path.join(os.path.dirname(__file__), 'preprocess.R')
    subprocess.call(["Rscript", target_script, raw_path, dest_path], shell=False)


def calculate_distance(row):
    return distance.distance((row['Start_Latitude'], row['Start_Longitude']),
                             (row['End_Latitude'], row['End_Longitude'])).km


def remove_trip_outlier(df, th):
    # delete all rows with column 'Trip_Duration' has value more than defined threshold
    indexNames = df[df['Trip_Duration'] >= th].index
    df.drop(indexNames, inplace=True)


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


def complete_station(df, stations):

    complete_cases = stations[stations.Station_ID.isin(stations[stations.isnull().any(1)]["Station_ID"])].dropna()
    for _, row in complete_cases.iterrows():
        id, lat, lng = row["Station_ID"], row["Latitude"], row["Longitude"]
        df.loc[df['Start_Station_ID'] == id, "Start_Latitude"] = lat
        df.loc[df['Start_Station_ID'] == id, "Start_Longitude"] = lng

        df.loc[df['End_Station_ID'] == id, "End_Latitude"] = lat
        df.loc[df['End_Station_ID'] == id, "End_Longitude"] = lng

        df.loc[(df['Start_Latitude'] == lat) & (df['Start_Longitude'] == lng), "Start_Station_ID"] = id
        df.loc[(df['End_Latitude'] == lat) & (df['End_Longitude'] == lng), "End_Station_ID"] = id