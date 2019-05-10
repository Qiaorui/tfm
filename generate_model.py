from scripts import preprocess
from scripts import utils
from scripts import models
import pandas as pd
import gc
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.decomposition import PCA
import statsmodels.api as sm

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
warnings.simplefilter("ignore")


def pca(df, tv):

    y = df[tv]
    x = df.drop(tv, axis=1)

    N = 1000
    plt.plot(y.tail(N).index, y.tail(N))
    plt.gcf().autofmt_xdate()
    plt.show()

    decomposition = sm.tsa.seasonal_decompose(y.tail(N), model='additive')
    fig = decomposition.plot()
    plt.show()

    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(6, 1, 1)
    plt.plot(x.tail(N).index, x.tail(N)['Temperature'])
    plt.ylabel("Temp(Cº)")
    # make these tick labels invisible
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = plt.subplot(6, 1, 2, sharex=ax1)
    plt.plot(x.tail(N).index, x.tail(N)['Humidity'])
    plt.ylabel("Hum(%)")
    # make these tick labels invisible
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3 = plt.subplot(6, 1, 3, sharex=ax1)
    plt.plot(x.tail(N).index, x.tail(N)['Time_Fragment_Cos'])
    plt.ylabel("TF(Cos)")
    # make these tick labels invisible
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax4 = plt.subplot(6, 1, 4, sharex=ax1)
    plt.plot(x.tail(N).index, x.tail(N)['Time_Fragment_Sin'])
    plt.ylabel("TF(Sin)")
    plt.setp(ax4.get_xticklabels(), visible=False)

    ax5 = plt.subplot(6, 1, 5, sharex=ax1)
    plt.plot(x.tail(N).index, x.tail(N)['Weekday_Cos'])
    plt.ylabel("Weekday(Cos)")
    plt.setp(ax5.get_xticklabels(), visible=False)

    plt.subplot(6, 1, 6, sharex=ax1)
    plt.plot(x.tail(N).index, x.tail(N)['Condition_Good'])
    plt.ylabel("Condition")
    plt.xticks(x.tail(N).index.normalize().unique(), x.tail(N).index.normalize().unique().day)

    plt.subplots_adjust(hspace = .001)
    plt.tight_layout()
    plt.show()

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


def prepare_data(df, weather_data, time_slot):
    data = pd.DataFrame()

    start = df["Timestamp"].min().replace(hour=0, minute=0, second=0)
    end = df["Timestamp"].max().replace(hour=23, minute=59)
    index = pd.date_range(start=start, end=end, freq=str(time_slot) + 'Min', normalize=True)

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
    data['Weekday_Sin'] = np.sin(2*np.pi/10 * (data['Weekday']-2))
    data['Month_Cos'] = np.cos(2*np.pi/12 * (data['Month']))
    data['Month_Sin'] = np.sin(2 * np.pi / 12 * (data['Month']))
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
    data['Weekday_Sin'] = data['Weekday_Sin'].astype(np.float16)
    data['Month_Sin'] = data['Month_Sin'].astype(np.float16)

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
    parser.add_argument("-tp", type=float, default=0.2, help="Test size percentage for split the data")
    parser.add_argument("-start", default="2017-01-01", help="Input start date")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")

    args = parser.parse_args()

    weather_data_path = args.cw
    trip_data_path = args.ct
    time_slot = args.ts
    test_pct = args.tp
    start = pd.to_datetime(args.start).normalize()

    pd.set_option('display.precision', 3)
    pd.set_option('display.max_columns', 500)

    weather_data = utils.read_cleaned_weather_data(weather_data_path)
    assert weather_data is not None

    trip_data = utils.read_cleaned_trip_data(trip_data_path)
    assert trip_data is not None

    assert 24*60 % time_slot == 0

    trip_data = trip_data[trip_data['Start_Time'] >= start]

    if args.ot is not None:
        print("Removing outlier with threshold", args.ot)
        preprocess.remove_trip_outlier(trip_data, args.ot)

    print("{0:*^80}".format(" Prepare training data "))
    # Remove trips which contains sink station
    start_stations_ids = list(utils.get_start_station_dict(trip_data).keys())
    trip_data = trip_data[trip_data.End_Station_ID.isin(start_stations_ids)]

    print("Breaking trip data to pick-up data and drop-off data")
    pick_ups = trip_data[['Start_Station_ID', 'Start_Time']].copy()
    #drop_offs = trip_data[['End_Station_ID', 'Stop_Time']].copy()

    del trip_data
    gc.collect()

    pick_ups.rename(columns={"Start_Station_ID": "Station_ID", "Start_Time": "Timestamp"}, inplace=True)
    #drop_offs.rename(columns={"Stop_Station_ID": "Station_ID", "Stop_Time": "Timestamp"}, inplace=True)

    th_day = pick_ups['Timestamp'].max().value - (pick_ups['Timestamp'].max().value - pick_ups['Timestamp'].min().value) * test_pct
    th_day = pd.to_datetime(th_day).normalize()

    data = prepare_data(pick_ups, weather_data, time_slot)

    busiest_station = pick_ups["Station_ID"].value_counts().idxmax()
    print("{0:*^80}".format(" PCA "))
    # PCA
    pca_data = data.loc[data["Station_ID"]==busiest_station]
    #pca(pca_data, 'Count')

    # Training modules, train data by different techniques
    print("{0:*^80}".format(" Training "))
    x_train = data[data.index < th_day]
    x_test = data[data.index >= th_day]
    y_train = x_train['Count']
    y_test = x_test['Count']
    x_train.drop('Count', axis=1, inplace=True)
    x_test.drop('Count', axis=1, inplace=True)

    """
    arima = models.ARIMA()
    #arima.test(x_train, y_train, 1440//time_slot, busiest_station)
    #arima.fit(x_train, y_train, param[0], param[1])
    arima.fit(x_train, y_train, (1, 0, 1), (1, 0, 1, 24))
    y = arima.predict(x_test)
    models.score(y_test.tolist(), y)

    ha = models.HA()
    ha.fit(x_train, y_train)
    y = ha.predict(x_test)
    models.score(y_test.tolist(), y)
    """

    ssa = models.SSA()
    ssa.test(x_train, y_train, 1440//time_slot)
    exit(1)

    #dg = data.groupby("Station_ID")
    #for id, df in dg:
    #    sequence = convert_to_sequence(df.drop(columns=['Holiday', 'Station_ID']), ['Count'], 3, 4)
    #    print(sequence)
    #    exit(1)

    # Save model per each techniques

    # Evaluate the prediction
    print("{0:*^80}".format(" Evaluation "))
    y_test = pca_data['Count']
    week_sample = y_test.loc[th_day - pd.DateOffset(7):th_day + pd.DateOffset(7)]
    day_sample = y_test.loc[th_day - pd.DateOffset(1):th_day + pd.DateOffset(1)]
    base_week_df = pd.DataFrame(index=week_sample.index)
    base_week_df = base_week_df[base_week_df.index >= th_day]
    base_day_df = pd.DataFrame(index=day_sample.index)
    base_day_df = base_day_df[base_day_df.index >= th_day]

    ha_sample = ha.predict(pca_data.loc[th_day : th_day + pd.DateOffset(7)])
    arima_sample = arima.predict(pca_data.loc[th_day : th_day + pd.DateOffset(7)])

    base_week_df['ha'] = ha_sample
    base_week_df['arima'] = arima_sample

    plt.figure(figsize=(15, 7))
    plt.plot(week_sample, label="Observed")
    plt.plot(base_week_df['ha'], label="HA")
    plt.plot(base_week_df['arima'], label="ARIMA")
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()

    ha_sample = ha.predict(pca_data.loc[th_day : th_day + pd.DateOffset(1)])
    arima_sample = arima.predict(pca_data.loc[th_day : th_day + pd.DateOffset(1)])
    base_day_df['ha'] = ha_sample
    base_day_df['arima'] = arima_sample

    plt.figure(figsize=(15, 7))
    plt.plot(day_sample, label="Observed")
    plt.plot(base_day_df['ha'], label="HA")
    plt.plot(base_week_df['arima'], label="ARIMA")
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
