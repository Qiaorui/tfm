import warnings
warnings.filterwarnings("ignore")
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from scripts import preprocess
from scripts import utils
from scripts import judge
from scripts import models
sys.stderr = stderr
import pandas as pd
import gc
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def pca(df, tv, seasonality, show):

    y = df[tv]
    x = df.drop(tv, axis=1)

    N = 1000
    plt.plot(y.tail(N).index, y.tail(N))
    plt.gcf().autofmt_xdate()
    plt.savefig('results/p1.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    decomposition = sm.tsa.seasonal_decompose(y.tail(N), model='additive', freq=seasonality)
    fig = decomposition.plot()
    plt.savefig('results/p2.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

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
    plt.savefig('results/p3.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.cla()
        plt.close()

    # Using Pearson Correlation
    plt.figure(figsize=(12, 10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.RdBu)
    plt.savefig('results/p4.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    pca = PCA(svd_solver='full')
    fit = pca.fit(x)
    # summarize components
    print("Explained Variance: ", fit.explained_variance_ratio_)
    plt.semilogy(fit.explained_variance_ratio_, '--o', label="Explained Variance Ration")
    plt.semilogy(fit.explained_variance_ratio_.cumsum(), '--o', label="Cumulative Explained Variance Ratio")
    plt.legend()
    plt.savefig('results/p5.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


def prepare_data(df, weather_data, time_slot):
    data = pd.DataFrame()

    start = df["Timestamp"].min().replace(hour=0, minute=0, second=0, microsecond=0)
    end = df["Timestamp"].max().replace(hour=23, minute=59)
    index = pd.date_range(start=start, end=end, freq=str(time_slot) + 'Min')

    station_groups = df.groupby("Station_ID")
    for sid, sdf in station_groups:
        lat, lng = sdf.iloc[-1]['Latitude'], sdf.iloc[-1]['Longitude']
        sdf = utils.aggregate_by_time_slot(sdf, time_slot, index)
        sdf = utils.fill_weather_data(sdf, weather_data)
        sdf["Station_ID"] = sid

        # More station features
        if utils.ENCODER == "statistics":
            sdf['Latitude'] = lat
            sdf['Longitude'] = lng
            sdf['Mean_Count'] = sdf['Count'].mean()
            sdf['AM_Ratio'] = sdf[(sdf.index.dayofweek < 6) & (sdf.index.hour >= 7) & (sdf.index.hour <= 9)]['Count'].mean()
            sdf['PM_Ratio'] = sdf[(sdf.index.dayofweek < 6) & (sdf.index.hour >= 17) & (sdf.index.hour <= 19)]['Count'].mean()

        data = data.append(sdf)

    # Normalize the data
    scaler = MinMaxScaler()
    if utils.ENCODER == "statistics":
        normalize_features = ['Temperature', 'Wind', 'Humidity', 'Visibility', 'Latitude', 'Longitude', 'Mean_Count', 'AM_Ratio', 'PM_Ratio']
    else:
        normalize_features = ['Temperature', 'Wind', 'Humidity', 'Visibility']

    data[normalize_features] = scaler.fit_transform(data[normalize_features])

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

    data['Weekday_Cos'] = np.cos(2 * np.pi / 7 * data['Weekday'])
    data['Weekday_Sin'] = np.sin(2 * np.pi / 7 * data['Weekday'])
    data['Month_Cos'] = np.cos(2*np.pi/12 * (data['Month']))
    data['Month_Sin'] = np.sin(2 * np.pi / 12 * (data['Month']))
    data['Time_Fragment_Cos'] = np.cos(2 * np.pi / (data['Time_Fragment'].max() + 1) * data['Time_Fragment'])
    data['Time_Fragment_Sin'] = np.sin(2 * np.pi / (data['Time_Fragment'].max() + 1) * data['Time_Fragment'])

    data.drop(['Month'], axis='columns', inplace=True)

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
    data['Humidity'] = data['Humidity'].astype(np.float16)
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


def plot_sample_station_prediction(df, th_day, n_days, ha=None, arima=None, ssa=None, lr=None, mlp=None,
                                   lstm1=None, lstm2=None, lstm3=None, lstm4=None, lstm5=None, n_pre=2, n_post=2, show=False):
    print("Visualising the prediction for", n_days, "days")
    last_day = df.index[df.index < (th_day + pd.DateOffset(n_days))].max()

    y = df['Count']
    x = df.drop('Count', axis=1)
    x_test = x.loc[th_day : last_day]

    sample = y.loc[th_day - pd.DateOffset(n_days):last_day]
    base_df = pd.DataFrame(index=sample.index)
    base_df = base_df[base_df.index >= th_day]

    if ha is not None:
        ha_sample = ha.predict(x_test)
        base_df['HA'] = ha_sample
    x_test = x_test.drop(['Weekday', 'Time_Fragment'], axis=1)
    if arima is not None:
        arima_sample = arima.predict(x_test)
        base_df['ARIMA'] = arima_sample
    if ssa is not None:
        ssa_sample = ssa.predict(x_test, 5)
        base_df['SSA'] = ssa_sample
    if lr is not None:
        if utils.ENCODER == "statistics":
            lr_sample = lr.predict(x_test.drop('Station_ID', axis=1))
        else:
            lr_sample = lr.predict(x_test)
        base_df['LR'] = lr_sample
    if mlp is not None:
        if utils.ENCODER == "statistics":
            mlp_sample = mlp.predict(x_test.drop('Station_ID', axis=1))
        else:
            mlp_sample = mlp.predict(x_test)
        base_df['MLP'] = mlp_sample

    df = df.drop(['Weekday', 'Time_Fragment'], axis=1)

    if utils.scaler is not None:
        df[['Count']] = utils.scaler.transform(df[['Count']])

    non_sequential_columns = judge.non_sequential_columns
    x_non_sec_df = df[non_sequential_columns].loc[th_day: th_day + pd.DateOffset(n_days - 1)]
    x_non_sec_df = x_non_sec_df[(x_non_sec_df.index.hour == 0) & (x_non_sec_df.index.minute == 0)]
    if lstm1 is not None:
        x_sec_df, ydf, _ = judge.convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                                  target_as_feature=False, use_future=False, use_past=True)
        x_sec_df = x_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]
        x_sec_df = x_sec_df[(x_sec_df.index.hour == 0) & (x_sec_df.index.minute == 0)]

        lstm1_sample = lstm1.predict(x_sec_df, x_non_sec_df)

        if utils.scaler is not None:
            lstm1_sample = utils.scaler.inverse_transform(np.array(lstm1_sample).reshape(-1, 1))
        base_df['LSTM_1'] = lstm1_sample

    if lstm2 is not None:
        x_sec_df, ydf, _ = judge.convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                                  target_as_feature=False, use_future=True, use_past=False)
        x_sec_df = x_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]
        x_sec_df = x_sec_df[(x_sec_df.index.hour == 0) & (x_sec_df.index.minute == 0)]

        lstm2_sample = lstm2.predict(x_sec_df, x_non_sec_df)

        if utils.scaler is not None:
            lstm2_sample = utils.scaler.inverse_transform(np.array(lstm2_sample).reshape(-1, 1))
        base_df['LSTM_2'] = lstm2_sample

    if lstm3 is not None:
        x_sec_df, ydf, x_future_sec_df = judge.convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                                  target_as_feature=False, use_future=True, use_past=True)
        x_sec_df = x_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]
        x_future_sec_df = x_future_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]

        x_sec_df = x_sec_df[(x_sec_df.index.hour == 0) & (x_sec_df.index.minute == 0)]
        x_future_sec_df = x_future_sec_df[(x_future_sec_df.index.hour == 0) & (x_future_sec_df.index.minute == 0)]

        lstm3_sample = lstm3.predict(x_sec_df, x_non_sec_df, x_future_sec_df)

        if utils.scaler is not None:
            lstm3_sample = utils.scaler.inverse_transform(np.array(lstm3_sample).reshape(-1, 1))
        base_df['LSTM_3'] = lstm3_sample

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
                for idx, j in enumerate(lstm4_sample[-n_pre:]):
                    x_sec_row['Count-' + str(n_pre-idx)] = j
            y_row = lstm4.predict(x_sec_row, x_non_sec_row)
            lstm4_sample.extend(y_row)

        if utils.scaler is not None:
            lstm4_sample = utils.scaler.inverse_transform(np.array(lstm4_sample).reshape(-1, 1))
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
                for idx, j in enumerate(lstm5_sample[-n_pre:]):
                    x_sec_row['Count-' + str(n_pre-idx)] = j
            y_row = lstm5.predict(x_sec_row, x_non_sec_row, x_sec_future_row)
            lstm5_sample.extend(y_row)

        if utils.scaler is not None:
            lstm5_sample = utils.scaler.inverse_transform(np.array(lstm5_sample).reshape(-1, 1))
        base_df['LSTM_5'] = lstm5_sample

    plt.figure(figsize=(15, 7))
    plt.plot(sample, label="Observed")
    for col in base_df:
        plt.plot(base_df[col], label=col)
    plt.gcf().autofmt_xdate()
    plt.legend()
    filename = utils.get_next_filename("p")
    plt.savefig('results/' + filename + '.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


def evaluate_model_by_station(df, th_day, n_days, ha=None, arima=None, ssa=None, lr=None, mlp=None,
                                   lstm1=None, lstm2=None, lstm3=None, lstm4=None, lstm5=None, n_pre=2, n_post=2):

    last_day = df.index[df.index < (th_day + pd.DateOffset(n_days))].max()

    y = df['Count']
    x = df.drop('Count', axis=1)
    x_test = x.loc[th_day : last_day]

    sample = y.loc[th_day :last_day]

    mae_dict, rmse_dict = {}, {}

    if ha is not None:
        ha_sample = ha.predict(x_test)
        mae, rmse = judge.score(sample, ha_sample)
        mae_dict['HA'] = mae
        rmse_dict['HA'] = rmse
    x_test = x_test.drop(['Weekday', 'Time_Fragment'], axis=1)
    if arima is not None:
        arima_sample = arima.predict(x_test)

        mae, rmse = judge.score(sample, arima_sample)
        mae_dict['ARIMA'] = mae
        rmse_dict['ARIMA'] = rmse
    if ssa is not None:
        ssa_sample = ssa.predict(x_test, 5)

        mae, rmse = judge.score(sample, ssa_sample)
        mae_dict['SSA'] = mae
        rmse_dict['SSA'] = rmse
    if lr is not None:
        if utils.ENCODER == "statistics":
            lr_sample = lr.predict(x_test.drop('Station_ID', axis=1))
        else:
            lr_sample = lr.predict(x_test)

        mae, rmse = judge.score(sample, lr_sample)
        mae_dict['LR'] = mae
        rmse_dict['LR'] = rmse

    if utils.scaler is not None:
        df[['Count']] = utils.scaler.transform(df[['Count']])

    if mlp is not None:
        if utils.ENCODER == "statistics":
            mlp_sample = mlp.predict(x_test.drop('Station_ID', axis=1))
        else:
            mlp_sample = mlp.predict(x_test)

        if utils.scaler is not None:
            mlp_sample = utils.scaler.inverse_transform(np.array(mlp_sample).reshape(-1, 1))

        mae, rmse = judge.score(sample, mlp_sample)
        mae_dict['MLP'] = mae
        rmse_dict['MLP'] = rmse

    df = df.drop(['Weekday', 'Time_Fragment'], axis=1)

    non_sequential_columns = judge.non_sequential_columns
    x_non_sec_df = df[non_sequential_columns].loc[th_day: th_day + pd.DateOffset(n_days - 1)]
    x_non_sec_df = x_non_sec_df[(x_non_sec_df.index.hour == 0) & (x_non_sec_df.index.minute == 0)]
    if lstm1 is not None:
        x_sec_df, ydf, _ = judge.convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                                  target_as_feature=False, use_future=False, use_past=True)
        x_sec_df = x_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]
        x_sec_df = x_sec_df[(x_sec_df.index.hour == 0) & (x_sec_df.index.minute == 0)]

        lstm1_sample = lstm1.predict(x_sec_df, x_non_sec_df)

        if utils.scaler is not None:
            lstm1_sample = utils.scaler.inverse_transform(np.array(lstm1_sample).reshape(-1, 1))

        mae, rmse = judge.score(sample, lstm1_sample)
        mae_dict['LSTM_1'] = mae
        rmse_dict['LSTM_1'] = rmse

    if lstm2 is not None:
        x_sec_df, ydf, _ = judge.convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                                  target_as_feature=False, use_future=True, use_past=False)
        x_sec_df = x_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]
        x_sec_df = x_sec_df[(x_sec_df.index.hour == 0) & (x_sec_df.index.minute == 0)]

        lstm2_sample = lstm2.predict(x_sec_df, x_non_sec_df)

        if utils.scaler is not None:
            lstm2_sample = utils.scaler.inverse_transform(np.array(lstm2_sample).reshape(-1, 1))

        mae, rmse = judge.score(sample, lstm2_sample)
        mae_dict['LSTM_2'] = mae
        rmse_dict['LSTM_2'] = rmse

    if lstm3 is not None:
        x_sec_df, ydf, x_future_sec_df = judge.convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                                  target_as_feature=False, use_future=True, use_past=True)
        x_sec_df = x_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]
        x_future_sec_df = x_future_sec_df.loc[th_day : th_day + pd.DateOffset(n_days-1)]

        x_sec_df = x_sec_df[(x_sec_df.index.hour == 0) & (x_sec_df.index.minute == 0)]
        x_future_sec_df = x_future_sec_df[(x_future_sec_df.index.hour == 0) & (x_future_sec_df.index.minute == 0)]

        lstm3_sample = lstm3.predict(x_sec_df, x_non_sec_df, x_future_sec_df)

        if utils.scaler is not None:
            lstm3_sample = utils.scaler.inverse_transform(np.array(lstm3_sample).reshape(-1, 1))
        mae, rmse = judge.score(sample, lstm3_sample)
        mae_dict['LSTM_3'] = mae
        rmse_dict['LSTM_3'] = rmse

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
                for idx, j in enumerate(lstm4_sample[-n_pre:]):
                    x_sec_row['Count-' + str(n_pre-idx)] = j
            y_row = lstm4.predict(x_sec_row, x_non_sec_row)
            lstm4_sample.extend(y_row)

        if utils.scaler is not None:
            lstm4_sample = utils.scaler.inverse_transform(np.array(lstm4_sample).reshape(-1, 1))
        mae, rmse = judge.score(sample, lstm4_sample)
        mae_dict['LSTM_4'] = mae
        rmse_dict['LSTM_4'] = rmse

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
                for idx, j in enumerate(lstm5_sample[-n_pre:]):
                    x_sec_row['Count-' + str(n_pre-idx)] = j
            y_row = lstm5.predict(x_sec_row, x_non_sec_row, x_sec_future_row)
            lstm5_sample.extend(y_row)

        if utils.scaler is not None:
            lstm5_sample = utils.scaler.inverse_transform(np.array(lstm5_sample).reshape(-1, 1))
        mae, rmse = judge.score(sample, lstm5_sample)
        mae_dict['LSTM_5'] = mae
        rmse_dict['LSTM_5'] = rmse

    return mae_dict, rmse_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cw", default="cleaned_data/weather.csv", help="input cleaned weather data path")
    parser.add_argument("-ct", default="cleaned_data/JC_trip_data.csv", help="input cleaned trip data path")
    parser.add_argument("-ot", type=int, help="Outlier threshold")
    parser.add_argument("-ts", type=int, default=30, help="Time slot for the aggregation, units in minute")
    parser.add_argument("-start", default="2017-01-01", help="Input start date")
    parser.add_argument("-th", default="2018-12-01", help="Threshold datetime to split train and test dataset")
    parser.add_argument("-e", default="dummy", help="Choose encoding strategy to encode station ID" )
    parser.add_argument("-n", action="store_true", help="normalize the target variable")
    parser.add_argument("-s", action="store_true", help="plot statistical report")

    args = parser.parse_args()
    weather_data_path = args.cw
    trip_data_path = args.ct
    time_slot = args.ts
    utils.ENCODER = args.e
    normalise = args.n
    th_day = pd.to_datetime(args.th).normalize()
    start = pd.to_datetime(args.start).normalize()
    show = args.s
    seasonality = 1440//time_slot if 1440//time_slot > 1 else 7

    if utils.ENCODER == "statistics":
        judge.non_sequential_columns = ['Condition_Good', 'Holiday', 'Weekend', 'Latitude', 'Longitude', 'Mean_Count', 'AM_Ratio', 'PM_Ratio']

    pd.set_option('display.precision', 3)
    pd.set_option('display.max_columns', 500)

    weather_data = utils.read_cleaned_weather_data(weather_data_path)
    assert weather_data is not None

    trip_data = utils.read_cleaned_trip_data(trip_data_path)
    assert trip_data is not None

    assert 24*60 % time_slot == 0

    if args.ot is not None:
        print("Removing outlier with threshold", args.ot)
        preprocess.remove_trip_outlier(trip_data, args.ot)

    print("{0:*^80}".format(" Prepare training data "))
    # Remove trips which contains sink station
    start_stations_ids = list(utils.get_start_station_dict(trip_data).keys())
    trip_data = trip_data[trip_data.End_Station_ID.isin(start_stations_ids)]

    print("Breaking trip data to pick-up data and drop-off data")
    pick_ups = trip_data[['Start_Station_ID', 'Start_Time', 'Start_Latitude', 'Start_Longitude']].copy()
    #drop_offs = trip_data[['End_Station_ID', 'Stop_Time']].copy()
    del trip_data
    gc.collect()

    pick_ups.rename(columns={"Start_Station_ID": "Station_ID", "Start_Time": "Timestamp", 'Start_Latitude': 'Latitude',
                             'Start_Longitude': 'Longitude'}, inplace=True)
    #drop_offs.rename(columns={"Stop_Station_ID": "Station_ID", "Stop_Time": "Timestamp"}, inplace=True)

    # Left Strip the data in case some station are new and hasn't historical data
    pick_ups = utils.slice_data_by_time(pick_ups, 'Timestamp', start, th_day + pd.DateOffset(30), 3)
    data = prepare_data(pick_ups, weather_data, time_slot)

    print(data.describe())
    station_freq_counts = pick_ups["Station_ID"].value_counts() // ((th_day - start)/np.timedelta64(1,'D') + 30)
    plt.hist(station_freq_counts.values, bins=min(len(station_freq_counts.index)//5, 200))
    plt.title("Daily Frequency Distribution")
    plt.xlabel("Daily Frequency")
    plt.ylabel("Quantity of station")
    plt.savefig('results/p0.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    busiest_station = station_freq_counts.idxmax()
    idle_station = station_freq_counts.idxmin()
    median_station = station_freq_counts.index[len(station_freq_counts)//2]
    print("{0:*^80}".format(" PCA "))
    # PCA
    pca_data = data.loc[data["Station_ID"]==busiest_station]
    pca(pca_data.drop(['Station_ID'], axis=1), 'Count', seasonality, show)

    # Training modules, train data by different techniques
    print("{0:*^80}".format(" Training "))
    # Fit the normalisation range
    if normalise:
        utils.scaler.fit(np.array(data['Count']).reshape(-1, 1))
        print(utils.scaler)
    else:
        utils.scaler = None

    ha, ssa, arima, lr, mlp, lstm1, lstm2, lstm3, lstm4, lstm5 = None, None, None, None, None, None, None, None, None, None

    days_to_evaluate = [30, 14, 7]

    mae_df, rmse_df, ha = judge.evaluate_ha(data, th_day, days_to_evaluate)
    data = data.drop(['Weekday', 'Time_Fragment'], axis=1)

    if utils.ENCODER == "statistics":
        mae, rmse, lr = judge.evaluate_lr(data.drop('Station_ID', axis=1), th_day, days_to_evaluate)
    else:
        mae, rmse, lr = judge.evaluate_lr(data, th_day, days_to_evaluate)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    days_to_evaluate = [30, 14, 7, 1]
    mae, rmse, arima = judge.evaluate_arima(data, th_day, days_to_evaluate, seasonality, station_freq_counts.index, show)
    if arima is not None:
        mae_df = mae_df.join(mae, how='outer')
        rmse_df = rmse_df.join(rmse, how='outer')

    if arima is not None:
        mae, rmse, ssa = judge.evaluate_ssa(data, th_day, days_to_evaluate, seasonality, busiest_station, show)
        mae_df = mae_df.join(mae, how='outer')
        rmse_df = rmse_df.join(rmse, how='outer')

    if utils.scaler is not None:
        data[['Count']] = utils.scaler.transform(data[['Count']])

    days_to_evaluate = [30, 14, 7]

    if utils.ENCODER == "statistics":
        mae, rmse, mlp = judge.evaluate_mlp(data.drop('Station_ID', axis=1), th_day, days_to_evaluate, show)
    else:
        mae, rmse, mlp = judge.evaluate_mlp(data, th_day, days_to_evaluate, show)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    mae, rmse, lstm1 = judge.evaluate_lstm_1(data, th_day, days_to_evaluate, seasonality, seasonality, show)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    mae, rmse, lstm2 = judge.evaluate_lstm_2(data, th_day, days_to_evaluate, seasonality, seasonality, show)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    mae, rmse, lstm3 = judge.evaluate_lstm_3(data, th_day, days_to_evaluate, seasonality, seasonality, show)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    days_to_evaluate = [30, 14, 7, 1]
    mae, rmse, lstm4 = judge.evaluate_lstm_4(data, th_day, days_to_evaluate, seasonality, seasonality, show)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    mae, rmse, lstm5 = judge.evaluate_lstm_5(data, th_day, days_to_evaluate, seasonality, seasonality, show)
    mae_df = mae_df.join(mae, how='outer')
    rmse_df = rmse_df.join(rmse, how='outer')

    # Evaluate the prediction
    print("{0:*^80}".format(" Evaluation "))
    for n in days_to_evaluate:
        plot_sample_station_prediction(pca_data, th_day, n, ha=ha, arima=arima, ssa=ssa, lr=lr, mlp=mlp,
                                       lstm1=lstm1, lstm2=lstm2, lstm3=lstm3, lstm4=lstm4, lstm5=lstm5, n_pre=seasonality,
                                       n_post=seasonality, show=show)

    mae_df.sort_index(inplace=True)
    rmse_df.sort_index(inplace=True)
    print("MAE:")
    print(mae_df)
    print("\nRMSE:")
    print(rmse_df)
    xs_label = [str(i) + "days" for i in days_to_evaluate]

    for col in mae_df:
        plt.plot(mae_df[col].dropna(), linestyle='-', marker='o', label=col)
    plt.ylabel("MAE")
    plt.xticks(days_to_evaluate, xs_label)
    plt.legend()
    filename = utils.get_next_filename("p")
    plt.savefig('results/' + filename + '.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    for col in rmse_df:
        plt.plot(rmse_df[col].dropna(), linestyle='-', marker='o', label=col)
    plt.ylabel("RMSE")
    plt.xticks(days_to_evaluate, xs_label)
    plt.legend()
    filename = utils.get_next_filename("p")
    plt.savefig('results/' + filename + '.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    # Evaluate per each station
    data = prepare_data(pick_ups, weather_data, time_slot)

    station_ids = station_freq_counts.index.values.tolist()

    mae_df = pd.DataFrame()
    rmse_df = pd.DataFrame()
    for sid in station_ids:
        print("Inspecting Station,", sid)
        station_data = data.loc[data["Station_ID"] == sid]
        mae, rmse = evaluate_model_by_station(station_data, th_day, 30, ha=ha, arima=arima, ssa=ssa, lr=lr, mlp=mlp,
                                       lstm1=lstm1, lstm2=lstm2, lstm3=lstm3, lstm4=lstm4, lstm5=lstm5, n_pre=seasonality,
                                       n_post=seasonality)
        mae_df = mae_df.append(mae, ignore_index=True)
        rmse_df = rmse_df.append(rmse, ignore_index=True)

    # Drop types we don't use
    if 'LR' in mae_df.columns:
        mae_df.drop('LR', axis=1, inplace=True)
        rmse_df.drop('LR', axis=1, inplace=True)
    if 'SSA' in mae_df.columns:
        mae_df.drop('SSA', axis=1, inplace=True)
        rmse_df.drop('SSA', axis=1, inplace=True)

    mae_df = mae_df.sub(mae_df['HA'], axis=0)
    rmse_df = rmse_df.sub(rmse_df['HA'], axis=0)

    mae_df['freq'] = station_freq_counts.values.tolist()
    mae_df = mae_df.groupby('freq').mean().reset_index(drop=True)

    rmse_df['freq'] = station_freq_counts.values.tolist()
    rmse_df = rmse_df.groupby('freq').mean().reset_index(drop=True)
    print(mae_df)
    print(rmse_df)

    n_ticks = 5
    max_ticks_size = len(mae_df.index)
    x_ticks_index = [mae_df.index.values.tolist()[x * (max_ticks_size - 1) // (n_ticks - 1)] for x in range(n_ticks)]
    freq_index = station_freq_counts.drop_duplicates().values.tolist()
    x_ticks_label = [freq_index[x * (max_ticks_size - 1) // (n_ticks - 1)] for x in range(n_ticks)]

    for col in mae_df:
        plt.plot(mae_df[col].dropna(), linestyle='-', label=col)
    plt.ylabel("MAE")
    plt.xlabel("Daily Frequency")
    plt.xticks(x_ticks_index, x_ticks_label)

    plt.legend()
    filename = utils.get_next_filename("p")
    plt.savefig('results/' + filename + '.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    for col in rmse_df:
        plt.plot(rmse_df[col].dropna(), linestyle='-', label=col)
    plt.ylabel("RMSE")
    plt.xlabel("Daily Frequency")

    plt.xticks(x_ticks_index, x_ticks_label)

    plt.legend()
    filename = utils.get_next_filename("p")
    plt.savefig('results/' + filename + '.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


if __name__ == '__main__':
    main()
