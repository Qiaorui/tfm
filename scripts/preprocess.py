from . import utils
import subprocess
import os
import pandas as pd
import numpy as np
from geopy import distance
from pandas.tseries.holiday import USFederalHolidayCalendar


def preprocess_trips(raw_path, dest_path):
    df = utils.read_raw_trip_data(raw_path)

    print(df.describe())
    print(df.isnull().sum())
    print('Contains ', sum(df.isnull().sum()), ' NULL values.')
    print('Dropping null or incorrect rows')
    complete_size = len(df.index)
    df.dropna(subset=["Trip_Duration", "Start_Time", "Stop_Time", "Start_Station_ID",
                      "Start_Latitude", "Start_Longitude", "End_Station_ID",
                      "End_Latitude", "End_Longitude"], inplace=True)
    df = df[(df['End_Latitude'] != 0) & (df['End_Latitude'] != 0)].copy()
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

    df['Distance'] = df.apply(calculate_distance, axis=1)

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df['Start_Time'].min(), end=df['Start_Time'].max())
    print('From {} until {}'.format(df['Start_Time'].min(), df['Start_Time'].max()))

    df['Start_Holiday'] = df['Start_Time'].dt.normalize().isin(holidays) #| (df['Start_Weekday'] == 7)
    df['Stop_Holiday'] = df['Stop_Time'].dt.normalize().isin(holidays)

    print(len(df[df['Start_Holiday'] == True]), 'trips done during holidays')

    print("Dropping columns :", ["Bike_ID", "Birth_Year", "User_Type", "Gender", "Start_Station_Name", "End_Station_Name"])
    df.drop(["Bike_ID", "Birth_Year", "User_Type", "Gender", "Start_Station_Name", "End_Station_Name"], 1, inplace=True)

    print(df.describe())

    df.to_csv(dest_path, index=False)


def preprocess_weathers(raw_path, dest_path):
    df = utils.read_raw_weather_data(raw_path)

    print(df.isnull().sum())
    print('Contains ', sum(df.isnull().sum()), ' NULL values. Need to impute :', sum(df.isnull().sum()) > 0)

    target_script = os.path.join(os.path.dirname(__file__), 'preprocess.R')
    subprocess.call(["Rscript", target_script, raw_path, dest_path], shell=False)


def calculate_distance(row):
    return distance.distance((row['Start_Latitude'], row['Start_Longitude']),
                             (row['End_Latitude'], row['End_Longitude'])).km
