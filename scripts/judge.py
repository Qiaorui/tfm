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


def convert_to_sequence(df, output_columns, lags=0, aheads=1, dropnan=True):
    new_df = pd.DataFrame()
    x_columns = []
    # Add lags (t-lag, t-lag+1, t-lag+2, ... , t-1)
    for lag in range(lags, 0, -1):
        for column in df.columns:
            new_column_name = column + "_lag_" + str(lag)
            new_df[new_column_name] = df[column].shift(lag).values
            x_columns.append(new_column_name)
    # Add current observation (t)
    for column in df.columns:
        new_df[column] = df[column].values
        x_columns.append(column)
    # Add ste aheads (t+1, t+2, ... , t+aheads)
    y_columns = []
    for ahead in range(1, aheads + 1, 1):
        for output_column in output_columns:
            new_column_name = output_column + "_ahead_" + str(ahead)
            new_df[new_column_name] = df[output_column].shift(-ahead).values
            y_columns.append(new_column_name)
    if dropnan:
        new_df.dropna(inplace=True)
    return new_df


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