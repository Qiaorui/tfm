from scripts import models
import pandas as pd
import numpy as np
import math
import sklearn.metrics


def score(y_true, y_pred):
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print("MAE:", mae, "  RMSE:", rmse)
    return mae, rmse


# 1: Many to Many don't use target variable as feature and don't use future feature
# 2: Only using future feature
# 3: 1+2, using past and future feature but not using target variable as feature
# 4: 1 + target variable as feature
# 5: 3 using tartget variable as feature
def convert_to_sequence(df, output_columns, n_pre=0, n_post=2, target_as_feature=False, use_future=False, use_past=True):
    assert n_post > 1

    x_df = pd.DataFrame(index=df.index)
    y_df = pd.DataFrame(index=df.index)

    feature_columns = [col for col in df.columns.tolist() if col not in output_columns]

    # Add past variables
    if use_past:
        for n_pre in range(n_pre, 0, -1):
            for col in feature_columns:
                new_column_name = col + "-" + str(n_pre)
                x_df[new_column_name] = df[col].shift(n_pre).values
            if target_as_feature:
                for col in output_columns:
                    new_column_name = col + "-" + str(n_pre)
                    x_df[new_column_name] = df[col].shift(n_pre).values

    # Add future variables
    if use_future:
        for n_post in range(n_post):
            for col in feature_columns:
                new_column_name = col + "+" + str(n_post)
                x_df[new_column_name] = df[col].shift(-n_post).values

    # Add Y target variables
    for n_post in range(n_post):
        for col in output_columns:
            new_column_name = col + "+" + str(n_post)
            y_df[new_column_name] = df[col].shift(-n_post).values

    x_df.dropna(inplace=True)
    y_df.dropna(inplace=True)
    return x_df, y_df


def evaluate_ha(data, th_day, n_days):
    x_train = data[data.index < th_day]
    y_train = x_train['Count']
    x_train.drop('Count', axis=1, inplace=True)

    x_test = data[data.index >= th_day]
    y_test = x_test['Count']
    x_test.drop('Count', axis=1, inplace=True)

    ha = models.HA()
    ha.fit(x_train, y_train)

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_test = x_test.loc[x_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        y = ha.predict(x_test)
        mae, rmse = score(y_test.tolist(), y)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['HA'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['HA'])

    return mae_df, rmse_df, ha


def evaluate_ssa(data, th_day, n_days, seasonality, busiest_station):
    x_train = data[data.index < th_day]
    y_train = x_train['Count']
    x_train.drop('Count', axis=1, inplace=True)

    x_test = data[data.index >= th_day]
    y_test = x_test['Count']
    x_test.drop('Count', axis=1, inplace=True)

    ssa = models.SSA()
    ssa_param = ssa.test(x_train, y_train, seasonality, busiest_station)
    ssa.fit(x_train, y_train, ssa_param)

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_test = x_test.loc[x_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        y = ssa.predict(x_test, ssa_param)
        mae, rmse = score(y_test.tolist(), y)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['SSA'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['SSA'])

    return mae_df, rmse_df, ssa


def evaluate_arima(data, th_day, n_days):
    x_train = data[data.index < th_day]
    y_train = x_train['Count']
    x_train.drop('Count', axis=1, inplace=True)

    x_test = data[data.index >= th_day]
    y_test = x_test['Count']
    x_test.drop('Count', axis=1, inplace=True)

    arima = models.ARIMA()
    #param, param2 = arima.test(x_train, y_train, seasonality, station_freq_counts.index)
    #arima.fit(x_train, y_train, param, param2)
    arima.fit(x_train, y_train, (1, 0, 1), (1, 0, 1, 24))

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_test = x_test.loc[x_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        y = arima.predict(x_test)
        mae, rmse = score(y_test.tolist(), y)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['ARIMA'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['ARIMA'])

    return mae_df, rmse_df, arima


def evaluate_lr(data, th_day, n_days):
    x_train = data[data.index < th_day]
    y_train = x_train['Count']
    x_train.drop('Count', axis=1, inplace=True)

    x_test = data[data.index >= th_day]
    y_test = x_test['Count']
    x_test.drop('Count', axis=1, inplace=True)

    lr = models.LR()
    lr.fit(x_train, y_train)

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_test = x_test.loc[x_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        y = lr.predict(x_test)
        mae, rmse = score(y_test.tolist(), y)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['LR'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['LR'])

    return mae_df, rmse_df, lr


def evaluate_mlp(data, th_day, n_days):
    x_train = data[data.index < th_day]
    y_train = x_train['Count']
    x_train.drop('Count', axis=1, inplace=True)

    x_test = data[data.index >= th_day]
    y_test = x_test['Count']
    x_test.drop('Count', axis=1, inplace=True)

    mlp = models.MLP()
    #mlp.test(x_train, y_train)
    mlp.fit(x_train, y_train)

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_test = x_test.loc[x_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        y = mlp.predict(x_test)
        mae, rmse = score(y_test.tolist(), y)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['MLP'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['MLP'])

    return mae_df, rmse_df, mlp


def evaluate_lstm_1(data, th_day, n_days, n_pre=2, n_post=2):
    non_sequential_columns = ['Station_ID', 'Condition_Good', 'Holiday', 'Weekend']
    sequential_columns = [col for col in data.columns.tolist() if col not in non_sequential_columns]

    x_sec = pd.DataFrame()
    x_non_sec = pd.DataFrame()
    y = pd.DataFrame()

    groups = data.groupby('Station_ID')
    # Frame data as a sequence
    for station_id, df in groups:
        # Sequential features
        x_sec_df, ydf = convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                       target_as_feature=False, use_future=False, use_past=True)

        start_time = x_sec_df.index.min()
        end_time = ydf.index.max()

        x_sec_df = x_sec_df[x_sec_df.index <= end_time]
        ydf = ydf[start_time:]
        y = y.append(ydf)

        x_sec = x_sec.append(x_sec_df)

        # Non-sequential
        x_non_sec_df = df[non_sequential_columns][(df.index >= start_time) & (df.index <= end_time)]
        x_non_sec = x_non_sec.append(x_non_sec_df)

    x_sec_train = x_sec[x_sec.index < th_day]
    x_non_sec_train = x_non_sec[x_non_sec.index < th_day]
    y_train = y[y.index < th_day]

    x_sec_test = x_sec[x_sec.index >= th_day]
    x_non_sec_test = x_non_sec[x_non_sec.index >= th_day]
    y_test = y[y.index >= th_day]

    x_sec_test = x_sec_test[(x_sec_test.index.minute == 0) & (x_sec_test.index.hour == 0)]
    x_non_sec_test = x_non_sec_test[(x_non_sec_test.index.minute == 0) & (x_non_sec_test.index.hour == 0) ]
    y_test = y_test[(y_test.index.minute == 0) & (y_test.index.hour == 0)]

    lstm = models.LSTM(n_pre, n_post)
    #lstm.test(x_train, y_train)
    lstm.fit(x_sec_train, x_non_sec_train, y_train, x_sec_test, x_non_sec_test, y_test)

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_sec_test = x_sec_test.loc[x_sec_test.index < (th_day + pd.DateOffset(n))]
        x_non_sec_test = x_non_sec_test.loc[x_non_sec_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        y = lstm.predict(x_sec_test, x_non_sec_test)
        mae, rmse = score(y_test.values.flatten(), y.flatten())
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['LSTM'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['LSTM'])

    return mae_df, rmse_df, lstm


def evaluate_lstm_2(data, th_day, n_days, n_pre=2, n_post=2):
    non_sequential_columns = ['Station_ID', 'Condition_Good', 'Holiday', 'Weekend']
    sequential_columns = [col for col in data.columns.tolist() if col not in non_sequential_columns]

    x_sec = pd.DataFrame()
    x_non_sec = pd.DataFrame()
    y = pd.DataFrame()

    groups = data.groupby('Station_ID')
    # Frame data as a sequence
    for station_id, df in groups:
        # Sequential features
        x_sec_df, ydf = convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                       target_as_feature=False, use_future=False, use_past=True)

        start_time = x_sec_df.index.min()
        end_time = ydf.index.max()

        x_sec_df = x_sec_df[x_sec_df.index <= end_time]
        ydf = ydf[start_time:]
        y = y.append(ydf)

        x_sec = x_sec.append(x_sec_df)

        # Non-sequential
        x_non_sec_df = df[non_sequential_columns][(df.index >= start_time) & (df.index <= end_time)]
        x_non_sec = x_non_sec.append(x_non_sec_df)

    x_sec_train = x_sec[x_sec.index < th_day]
    x_non_sec_train = x_non_sec[x_non_sec.index < th_day]
    y_train = y[y.index < th_day]

    x_sec_test = x_sec[x_sec.index >= th_day]
    x_non_sec_test = x_non_sec[x_non_sec.index >= th_day]
    y_test = y[y.index >= th_day]

    lstm = models.LSTM(n_pre, n_post)
    #lstm.test(x_train, y_train)
    lstm.fit(x_sec_train, x_non_sec_train, y_train, x_sec_test, x_non_sec_test, y_test)

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_sec_test = x_sec_test.loc[x_sec_test.index < (th_day + pd.DateOffset(n))]
        x_non_sec_test = x_non_sec_test.loc[x_non_sec_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        x_non_sec = lstm.predict(x_sec_test, x_non_sec_test)
        mae, rmse = score(y_test.tolist(), x_non_sec)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['LSTM'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['LSTM'])

    return mae_df, rmse_df, lstm

def evaluate_lstm_3(data, th_day, n_days, n_pre=2, n_post=2):
    non_sequential_columns = ['Station_ID', 'Condition_Good', 'Holiday', 'Weekend']
    sequential_columns = [col for col in data.columns.tolist() if col not in non_sequential_columns]

    x_sec = pd.DataFrame()
    x_non_sec = pd.DataFrame()
    y = pd.DataFrame()

    groups = data.groupby('Station_ID')
    # Frame data as a sequence
    for station_id, df in groups:
        # Sequential features
        x_sec_df, ydf = convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                       target_as_feature=False, use_future=False, use_past=True)

        start_time = x_sec_df.index.min()
        end_time = ydf.index.max()

        x_sec_df = x_sec_df[x_sec_df.index <= end_time]
        ydf = ydf[start_time:]
        y = y.append(ydf)

        x_sec = x_sec.append(x_sec_df)

        # Non-sequential
        x_non_sec_df = df[non_sequential_columns][(df.index >= start_time) & (df.index <= end_time)]
        x_non_sec = x_non_sec.append(x_non_sec_df)

    x_sec_train = x_sec[x_sec.index < th_day]
    x_non_sec_train = x_non_sec[x_non_sec.index < th_day]
    y_train = y[y.index < th_day]

    x_sec_test = x_sec[x_sec.index >= th_day]
    x_non_sec_test = x_non_sec[x_non_sec.index >= th_day]
    y_test = y[y.index >= th_day]

    lstm = models.LSTM(n_pre, n_post)
    #lstm.test(x_train, y_train)
    lstm.fit(x_sec_train, x_non_sec_train, y_train, x_sec_test, x_non_sec_test, y_test)

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_sec_test = x_sec_test.loc[x_sec_test.index < (th_day + pd.DateOffset(n))]
        x_non_sec_test = x_non_sec_test.loc[x_non_sec_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        x_non_sec = lstm.predict(x_sec_test, x_non_sec_test)
        mae, rmse = score(y_test.tolist(), x_non_sec)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['LSTM'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['LSTM'])

    return mae_df, rmse_df, lstm

def evaluate_lstm_4(data, th_day, n_days, n_pre=2, n_post=2):
    non_sequential_columns = ['Station_ID', 'Condition_Good', 'Holiday', 'Weekend']
    sequential_columns = [col for col in data.columns.tolist() if col not in non_sequential_columns]

    x_sec = pd.DataFrame()
    x_non_sec = pd.DataFrame()
    y = pd.DataFrame()

    groups = data.groupby('Station_ID')
    # Frame data as a sequence
    for station_id, df in groups:
        # Sequential features
        x_sec_df, ydf = convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                       target_as_feature=False, use_future=False, use_past=True)

        start_time = x_sec_df.index.min()
        end_time = ydf.index.max()

        x_sec_df = x_sec_df[x_sec_df.index <= end_time]
        ydf = ydf[start_time:]
        y = y.append(ydf)

        x_sec = x_sec.append(x_sec_df)

        # Non-sequential
        x_non_sec_df = df[non_sequential_columns][(df.index >= start_time) & (df.index <= end_time)]
        x_non_sec = x_non_sec.append(x_non_sec_df)

    x_sec_train = x_sec[x_sec.index < th_day]
    x_non_sec_train = x_non_sec[x_non_sec.index < th_day]
    y_train = y[y.index < th_day]

    x_sec_test = x_sec[x_sec.index >= th_day]
    x_non_sec_test = x_non_sec[x_non_sec.index >= th_day]
    y_test = y[y.index >= th_day]

    lstm = models.LSTM(n_pre, n_post)
    #lstm.test(x_train, y_train)
    lstm.fit(x_sec_train, x_non_sec_train, y_train, x_sec_test, x_non_sec_test, y_test)

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_sec_test = x_sec_test.loc[x_sec_test.index < (th_day + pd.DateOffset(n))]
        x_non_sec_test = x_non_sec_test.loc[x_non_sec_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        x_non_sec = lstm.predict(x_sec_test, x_non_sec_test)
        mae, rmse = score(y_test.tolist(), x_non_sec)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['LSTM'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['LSTM'])

    return mae_df, rmse_df, lstm

def evaluate_lstm_5(data, th_day, n_days, n_pre=2, n_post=2):
    non_sequential_columns = ['Station_ID', 'Condition_Good', 'Holiday', 'Weekend']
    sequential_columns = [col for col in data.columns.tolist() if col not in non_sequential_columns]

    x_sec = pd.DataFrame()
    x_non_sec = pd.DataFrame()
    y = pd.DataFrame()

    groups = data.groupby('Station_ID')
    # Frame data as a sequence
    for station_id, df in groups:
        # Sequential features
        x_sec_df, ydf = convert_to_sequence(df.drop(columns=non_sequential_columns), ['Count'], n_pre, n_post,
                                       target_as_feature=False, use_future=False, use_past=True)

        start_time = x_sec_df.index.min()
        end_time = ydf.index.max()

        x_sec_df = x_sec_df[x_sec_df.index <= end_time]
        ydf = ydf[start_time:]
        y = y.append(ydf)

        x_sec = x_sec.append(x_sec_df)

        # Non-sequential
        x_non_sec_df = df[non_sequential_columns][(df.index >= start_time) & (df.index <= end_time)]
        x_non_sec = x_non_sec.append(x_non_sec_df)

    x_sec_train = x_sec[x_sec.index < th_day]
    x_non_sec_train = x_non_sec[x_non_sec.index < th_day]
    y_train = y[y.index < th_day]

    x_sec_test = x_sec[x_sec.index >= th_day]
    x_non_sec_test = x_non_sec[x_non_sec.index >= th_day]
    y_test = y[y.index >= th_day]

    lstm = models.LSTM(n_pre, n_post)
    #lstm.test(x_train, y_train)
    lstm.fit(x_sec_train, x_non_sec_train, y_train, x_sec_test, x_non_sec_test, y_test)

    mae_dict = {}
    rmse_dict = {}

    for n in n_days:
        x_sec_test = x_sec_test.loc[x_sec_test.index < (th_day + pd.DateOffset(n))]
        x_non_sec_test = x_non_sec_test.loc[x_non_sec_test.index < (th_day + pd.DateOffset(n))]
        y_test = y_test.loc[y_test.index < (th_day + pd.DateOffset(n))]
        x_non_sec = lstm.predict(x_sec_test, x_non_sec_test)
        mae, rmse = score(y_test.tolist(), x_non_sec)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['LSTM'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['LSTM'])

    return mae_df, rmse_df, lstm