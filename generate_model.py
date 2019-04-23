from scripts import preprocess
from scripts import utils
import pandas as pd
import gc
import argparse

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.decomposition import PCA


def pca(df, tv):

    y = df[tv]
    x = df.drop(tv, axis=1)

    # Using Pearson Correlation
    plt.figure(figsize=(12, 10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.RdBu)
    plt.show()

    pca = PCA(svd_solver='full')
    fit = pca.fit(x)
    # summarize components
    print("Explained Variance: ", fit.explained_variance_ratio_)
    plt.semilogy(fit.explained_variance_ratio_, '--o', label="Explained Variance Ration")
    plt.semilogy(fit.explained_variance_ratio_.cumsum(), '--o', label="Cumulative Explained Variance Ratio")
    plt.legend()
    plt.show()

    cols = list(x.columns)
    model = LinearRegression()
    # Initializing RFE model
    rfe = RFE(model, 7)
    # Transforming data using RFE
    X_rfe = rfe.fit_transform(x, y)
    # Fitting the data to model
    model.fit(X_rfe, y)
    temp = pd.Series(rfe.support_, index=cols)
    selected_features_rfe = temp[temp == True].index
    print(selected_features_rfe)


def prepare_data(df, weather_data, time_slot):
    data = pd.DataFrame()

    start = df["Timestamp"].min().replace(hour=0, minute=0, second=0)
    end = df["Timestamp"].max().replace(hour=23, minute=59)
    index = pd.date_range(start=start, end=end, freq=str(time_slot) + 'Min')

    station_groups = df.groupby("Station_ID")
    for sid, sdf in station_groups:
        sdf = utils.aggregate_by_time_slot(sdf, time_slot, index)
        sdf = utils.fill_weather_data(sdf, weather_data)
        sdf["Station_ID"] = sid
        data = data.append(sdf)

    cloudy_conds = ["Clear", "Partly Cloudy", "Scattered Clouds", "Mostly Cloudy", "Haze", "Overcast"]
    data.loc[-data.Condition.isin(cloudy_conds), 'Condition'] = 0
    data.loc[data.Condition.isin(cloudy_conds), 'Condition'] = 1
    data.rename(index=str, columns={"Condition": "Condition_Good"}, inplace=True)
    data.index = pd.to_datetime(data.index)

    data['Time_Fragment'] = np.ceil((data.index.hour*60 + data.index.minute)/time_slot)

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=data.index.min(), end=data.index.max())
    data['Holiday'] = 0
    data.loc[data.index.normalize().isin(holidays), 'Holiday'] = 1

    data['Weekday'] = data.index.dayofweek
    data['Month'] = data.index.month

    data['Weekend'] = 0
    data.loc[data['Weekday'] > 4, 'Weekend'] = 1

    data.loc[data['Weekday'] == 5, 'Weekday'] += 1
    data.loc[data['Weekday'] == 6, 'Weekday'] += 2

    data['Weekday_Cos'] = np.cos(2*np.pi/10 * (data['Weekday']-2))
    data['Month_Cos'] = np.cos(2*np.pi/12 * (data['Month']))
    data['Time_Fragment_Cos'] = np.cos(2*np.pi/(data['Time_Fragment'].max()+1) * (data['Time_Fragment']))
    data['Time_Fragment_Sin'] = np.sin(2 * np.pi / (data['Time_Fragment'].max() + 1) * (data['Time_Fragment']))

    data.drop(['Weekday', 'Month', 'Time_Fragment'], axis='columns', inplace=True)

    data['Holiday'] = data['Holiday'].astype(np.int8)
    data['Condition_Good'] = data['Condition_Good'].astype(np.int8)
    data['Station_ID'] = data['Station_ID'].astype(np.int16)
    data['Weekend'] = data['Weekend'].astype(np.int8)
    data['Weekday_Cos'] = data['Weekday_Cos'].astype(np.float16)
    data['Month_Cos'] = data['Month_Cos'].astype(np.float16)
    data['Time_Fragment_Cos'] = data['Time_Fragment_Cos'].astype(np.float16)
    data['Time_Fragment_Sin'] = data['Time_Fragment_Sin'].astype(np.float16)

    max_count = data['Count'].max()
    if max_count < (1 << 7):
        data['Count'] = data['Count'].astype(np.int8)
    elif max_count < (1 << 15):
        data['Count'] = data['Count'].astype(np.int16)
    elif max_count < (1 << 31):
        data['Count'] = data['Count'].astype(np.int32)
    data.info()

    return data


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
    time_slot = args.ts

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

    pick_ups = trip_data[['Start_Station_ID', 'Start_Time']].copy()
    drop_offs = trip_data[['End_Station_ID', 'Stop_Time']].copy()

    del trip_data
    gc.collect()

    pick_ups.rename(columns={"Start_Station_ID": "Station_ID", "Start_Time": "Timestamp"}, inplace=True)
    drop_offs.rename(columns={"Stop_Station_ID": "Station_ID", "Stop_Time": "Timestamp"}, inplace=True)

    busiest_station = pick_ups["Station_ID"].value_counts().idxmax()

    data = prepare_data(pick_ups, weather_data, time_slot)

    pca_data = data.loc[data["Station_ID"]==busiest_station].drop("Station_ID", axis=1)

    pca(pca_data, 'Count')
    # PCA

    # Training modules, train data by different techniques
    print("{0:*^80}".format(" Training "))

    # Save model per each techniques

    # Evaluate the prediction
    print("{0:*^80}".format(" Evaluation "))


if __name__ == '__main__':
    main()