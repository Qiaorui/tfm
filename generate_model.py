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


def lstrip_data(data, th):
    to_remove = []

    groups = data.groupby("Station_ID")
    for sid, df in groups:
        start_day = df.index.min()
        th_day = start_day + pd.DateOffset(th)
        cumsum = df['Count'].cumsum()
        index = np.argmax(cumsum > 0)
        if index > th_day:
            to_remove.append((sid, index.normalize()))

    for sid, idx in to_remove:
        data = data[~((data["Station_ID"] == sid) & (data.index < idx))]

    return data


def evaluate_ha(data, th_day, n_days, ha):
    x_test = data[data.index >= th_day]
    y_test = x_test['Count']
    x_test.drop('Count', axis=1, inplace=True)

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_test = x_test.loc[x_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        y = ha.predict(x_test)
        mae, rmse = models.score(y_test.tolist(), y)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['HA'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['HA'])

    return mae_df, rmse_df


def evaluate_ssa(data, th_day, n_days, ssa, ssa_param):
    x_test = data[data.index >= th_day]
    y_test = x_test['Count']
    x_test.drop('Count', axis=1, inplace=True)

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_test = x_test.loc[x_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        y = ssa.predict(x_test, ssa_param)
        mae, rmse = models.score(y_test.tolist(), y)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['SSA'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['SSA'])

    return mae_df, rmse_df


def evaluate_arima(data, th_day, n_days, arima):
    x_test = data[data.index >= th_day]
    y_test = x_test['Count']
    x_test.drop('Count', axis=1, inplace=True)

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_test = x_test.loc[x_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        y = arima.predict(x_test)
        mae, rmse = models.score(y_test.tolist(), y)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['ARIMA'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['ARIMA'])

    return mae_df, rmse_df


def plot_sample_station_prediction(df, th_day, n_days, ha, arima, ssa):
    y = df['Count']
    x = df.drop('Count', axis=1)
    x_test = x.loc[th_day : th_day + pd.DateOffset(n_days)]

    sample = y.loc[th_day - pd.DateOffset(n_days):th_day + pd.DateOffset(n_days)]
    base_df = pd.DataFrame(index=sample.index)
    base_df = base_df[base_df.index >= th_day]

    ha_sample = ha.predict(x_test)
    arima_sample = arima.predict(x_test)
    ssa_sample = ssa.predict(x_test, 5)

    base_df['HA'] = ha_sample
    base_df['ARIMA'] = arima_sample
    base_df['SSA'] = ssa_sample

    plt.figure(figsize=(15, 7))
    plt.plot(sample, label="Observed")
    plt.plot(base_df['HA'], label="HA")
    plt.plot(base_df['ARIMA'], label="ARIMA")
    plt.plot(base_df['SSA'], label="SSA")
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cw", default="cleaned_data/weather.csv", help="input cleaned weather data path")
    parser.add_argument("-ct", default="cleaned_data/JC_trip_data.csv", help="input cleaned trip data path")
    parser.add_argument("-ot", type=int, help="Outlier threshold")
    parser.add_argument("-ts", type=int, default=30, help="Time slot for the aggregation, units in minute")
    #parser.add_argument("-tp", type=float, default=0.2, help="Test size percentage for split the data")
    parser.add_argument("-start", default="2017-01-01", help="Input start date")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")

    args = parser.parse_args()

    weather_data_path = args.cw
    trip_data_path = args.ct
    time_slot = args.ts
    #test_pct = args.tp
    start = pd.to_datetime(args.start).normalize()
    seasonality = 1440//time_slot if 1440//time_slot > 1 else 7

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

    #th_day = pick_ups['Timestamp'].max().value - (pick_ups['Timestamp'].max().value - pick_ups['Timestamp'].min().value) * test_pct
    #th_day = pd.to_datetime(th_day).normalize()
    th_day = pd.to_datetime("2018-11-01").normalize()

    data = prepare_data(pick_ups, weather_data, time_slot)

    # Left Strip the data in case some station are new and hasn't historical data
    data = lstrip_data(data, 7)

    station_freq_counts = pick_ups["Station_ID"].value_counts()
    busiest_station = station_freq_counts.idxmax()
    idle_station = station_freq_counts.idxmin()
    median_station = station_freq_counts.index[len(station_freq_counts)//2]
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

    arima = models.ARIMA()
    #param, param2 = arima.test(x_train, y_train, seasonality, station_freq_counts.index)
    #arima.fit(x_train, y_train, param, param2)
    arima.fit(x_train, y_train, (1, 0, 1), (1, 0, 1, 24))

    ssa = models.SSA()
    ssa_param = ssa.test(x_train, y_train, seasonality, busiest_station)
    ssa.fit(x_train, y_train, ssa_param)

    ha = models.HA()
    ha.fit(x_train, y_train)

    #dg = data.groupby("Station_ID")
    #for id, df in dg:
    #    sequence = convert_to_sequence(df.drop(columns=['Holiday', 'Station_ID']), ['Count'], 3, 4)
    #    print(sequence)
    #    exit(1)

    # Save model per each techniques

    # Evaluate the prediction
    print("{0:*^80}".format(" Evaluation "))
    days_to_evaluate = [30, 14, 7, 1]
    mae_df, rmse_df = evaluate_ha(data, th_day, days_to_evaluate, ha)

    mae, rmse = evaluate_ssa(data, th_day, days_to_evaluate, ssa, ssa_param)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    mae, rmse = evaluate_arima(data, th_day, days_to_evaluate, arima)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    for n in days_to_evaluate:
        plot_sample_station_prediction(pca_data, th_day, n, ha, arima, ssa)

    mae_df.sort_index(inplace=True)
    rmse_df.sort_index(inplace=True)
    print(mae_df)
    print(rmse_df)
    xs_label = [str(i) + "days" for i in days_to_evaluate]

    plt.plot(mae_df['HA'], linestyle='-', marker='o', label="HA")
    plt.plot(mae_df['ARIMA'], linestyle='-', marker='o', label="ARIMA")
    plt.plot(mae_df['SSA'], linestyle='-', marker='o', label="SSA")
    plt.ylabel("MAE")
    plt.xticks(days_to_evaluate, xs_label)
    plt.legend()
    plt.show()

    plt.plot(rmse_df['HA'], linestyle='-', marker='o', label="HA")
    plt.plot(rmse_df['ARIMA'], linestyle='-', marker='o', label="ARIMA")
    plt.plot(rmse_df['SSA'], linestyle='-', marker='o', label="SSA")
    plt.ylabel("RMSE")
    plt.xticks(days_to_evaluate, xs_label)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
