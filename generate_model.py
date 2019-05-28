from scripts import preprocess
from scripts import utils
from scripts import judge
import pandas as pd
import gc
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
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
    plt.ylabel("Temp(%)")
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

    # Normalize the data
    scaler = MinMaxScaler()
    data[['Temperature', 'Wind', 'Humidity', 'Visibility']] = scaler.fit_transform(
        data[['Temperature', 'Wind', 'Humidity', 'Visibility']])

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
    data['Temperature'] = data['Temperature'].astype(np.float16)
    data['Wind'] = data['Wind'].astype(np.float16)
    data['Humidity'] = data['Wind'].astype(np.float16)
    data['Visibility'] = data['Visibility'].astype(np.float16)

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


def plot_sample_station_prediction(df, th_day, n_days, ha=None, arima=None, ssa=None, lr=None, mlp=None,
                                   lstm1=None, lstm2=None, lstm3=None, lstm4=None, lstm5=None, n_pre=2, n_post=2):
    y = df['Count']
    x = df.drop('Count', axis=1)
    x_test = x.loc[th_day : th_day + pd.DateOffset(n_days)]

    sample = y.loc[th_day - pd.DateOffset(n_days):th_day + pd.DateOffset(n_days)]
    base_df = pd.DataFrame(index=sample.index)
    base_df = base_df[base_df.index >= th_day]

    if ha is not None:
        ha_sample = ha.predict(x_test)
        base_df['HA'] = ha_sample
    if arima is not None:
        arima_sample = arima.predict(x_test)
        base_df['ARIMA'] = arima_sample
    if ssa is not None:
        ssa_sample = ssa.predict(x_test, 5)
        base_df['SSA'] = ssa_sample
    if lr is not None:
        lr_sample = lr.predict(x_test)
        base_df['LR'] = lr_sample
    if mlp is not None:
        mlp_sample = mlp.predict(x_test)
        base_df['MLP'] = mlp_sample

    sample.drop(sample.tail(1).index, inplace=True)
    base_df.drop(base_df.tail(1).index, inplace=True)

    non_sequential_columns = ['Station_ID', 'Condition_Good', 'Holiday', 'Weekend']
    x_non_sec_df = df[non_sequential_columns].loc[th_day: th_day + pd.DateOffset(n_days - 1)]
    x_non_sec_df = x_non_sec_df[(x_non_sec_df.index.hour == 0) & (x_non_sec_df.index.minute == 0)]
    if lstm1 is not None:
        x_sec_df, ydf, _ = judge.convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                                  target_as_feature=False, use_future=False, use_past=True)
        x_sec_df = x_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]
        x_sec_df = x_sec_df[(x_sec_df.index.hour == 0) & (x_sec_df.index.minute == 0)]

        lstm1_sample = lstm1.predict(x_sec_df, x_non_sec_df)
        base_df['LSTM_1'] = lstm1_sample.flatten()

    if lstm2 is not None:
        x_sec_df, ydf, _ = judge.convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                                  target_as_feature=False, use_future=True, use_past=False)
        x_sec_df = x_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]
        x_sec_df = x_sec_df[(x_sec_df.index.hour == 0) & (x_sec_df.index.minute == 0)]

        lstm2_sample = lstm2.predict(x_sec_df, x_non_sec_df)
        base_df['LSTM_2'] = lstm2_sample.flatten()
    if lstm3 is not None:
        x_sec_df, ydf, x_future_sec_df = judge.convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                                  target_as_feature=False, use_future=True, use_past=True)
        x_sec_df = x_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]
        x_future_sec_df = x_future_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]

        x_sec_df = x_sec_df[(x_sec_df.index.hour == 0) & (x_sec_df.index.minute == 0)]
        x_future_sec_df = x_future_sec_df[(x_future_sec_df.index.hour == 0) & (x_future_sec_df.index.minute == 0)]

        lstm3_sample = lstm3.predict(x_sec_df, x_non_sec_df, x_future_sec_df)
        base_df['LSTM_3'] = lstm3_sample.flatten()
    if lstm4 is not None:
        x_sec_df, ydf, _ = judge.convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                                  target_as_feature=True, use_future=False, use_past=True)
        x_sec_df = x_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]
        x_sec_df = x_sec_df[(x_sec_df.index.hour == 0) & (x_sec_df.index.minute == 0)]

        lstm4_sample = []
        for i in range(len(x_sec_df.index)):
            x_sec_row = x_sec_df.iloc[[i]]
            x_non_sec_row = x_non_sec_df.iloc[[i]]
            if lstm4_sample:
                for idx, j in enumerate(y[-n_pre:]):
                    x_sec_row['Count-' + str(n_pre-idx)] = j
            y_row = lstm4.predict(x_sec_row, x_non_sec_row)
            lstm4_sample.extend(y_row.flatten())

        base_df['LSTM_4'] = lstm4_sample
    if lstm5 is not None:
        x_sec_df, ydf, x_future_sec_df = judge.convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                                  target_as_feature=True, use_future=True, use_past=True)
        x_sec_df = x_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]
        x_future_sec_df = x_future_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]

        x_sec_df = x_sec_df[(x_sec_df.index.hour == 0) & (x_sec_df.index.minute == 0)]
        x_future_sec_df = x_future_sec_df[(x_future_sec_df.index.hour == 0) & (x_future_sec_df.index.minute == 0)]

        lstm5_sample = []
        for i in range(len(x_sec_df.index)):
            x_sec_row = x_sec_df.iloc[[i]]
            x_non_sec_row = x_non_sec_df.iloc[[i]]
            x_sec_future_row = x_future_sec_df.iloc[[i]]
            if lstm5_sample:
                for idx, j in enumerate(y[-n_pre:]):
                    x_sec_row['Count-' + str(n_pre-idx)] = j
            y_row = lstm5.predict(x_sec_row, x_non_sec_row, x_sec_future_row)
            lstm5_sample.extend(y_row.flatten())

        base_df['LSTM_5'] = lstm5_sample

    plt.figure(figsize=(15, 7))
    plt.plot(sample, label="Observed")
    for col in base_df:
        plt.plot(base_df[col], label=col)
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
    #pca(pca_data.drop('Station_ID', axis=1), 'Count')

    # Training modules, train data by different techniques
    print("{0:*^80}".format(" Training "))

    ha, ssa, arima, lr, mlp, lstm1, lstm2, lstm3, lstm4, lstm5 = None, None, None, None, None, None, None, None, None, None

    days_to_evaluate = [30, 14, 7, 1]

    mae_df, rmse_df, ha = judge.evaluate_ha(data, th_day, days_to_evaluate)

    mae, rmse, ssa = judge.evaluate_ssa(data, th_day, days_to_evaluate, seasonality, busiest_station)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    mae, rmse, arima = judge.evaluate_arima(data, th_day, days_to_evaluate)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')
    
    mae, rmse, lr = judge.evaluate_lr(data, th_day, days_to_evaluate)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    mae, rmse, mlp = judge.evaluate_mlp(data, th_day, days_to_evaluate)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    mae, rmse, lstm1 = judge.evaluate_lstm_1(data, th_day, days_to_evaluate, n_pre=seasonality, n_post=seasonality)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    mae, rmse, lstm2 = judge.evaluate_lstm_2(data, th_day, days_to_evaluate, n_pre=seasonality, n_post=seasonality)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    mae, rmse, lstm3 = judge.evaluate_lstm_3(data, th_day, days_to_evaluate, n_pre=seasonality, n_post=seasonality)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    mae, rmse, lstm4 = judge.evaluate_lstm_4(data, th_day, days_to_evaluate, n_pre=seasonality, n_post=seasonality)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    mae, rmse, lstm5 = judge.evaluate_lstm_5(data, th_day, days_to_evaluate, n_pre=seasonality, n_post=seasonality)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')


    # Evaluate the prediction
    print("{0:*^80}".format(" Evaluation "))
    for n in days_to_evaluate:
        plot_sample_station_prediction(pca_data, th_day, n, ha=ha, arima=arima, ssa=ssa, lr=lr, mlp=mlp,
                                       lstm1=lstm1, lstm2=lstm2, lstm3=lstm3, lstm4=lstm4, lstm5=lstm5, n_pre=seasonality,
                                       n_post=seasonality)

    mae_df.sort_index(inplace=True)
    rmse_df.sort_index(inplace=True)
    print("MAE:")
    print(mae_df)
    print("\nRMSE:")
    print(rmse_df)
    xs_label = [str(i) + "days" for i in days_to_evaluate]

    for col in mae_df:
        plt.plot(mae_df[col], linestyle='-', marker='o', label=col)
    plt.ylabel("MAE")
    plt.xticks(days_to_evaluate, xs_label)
    plt.legend()
    plt.show()

    for col in rmse_df:
        plt.plot(rmse_df[col], linestyle='-', marker='o', label=col)
    plt.ylabel("RMSE")
    plt.xticks(days_to_evaluate, xs_label)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
