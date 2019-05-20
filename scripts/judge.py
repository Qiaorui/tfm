from scripts import models
import pandas as pd
import numpy as np


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
        mae, rmse = models.score(y_test.tolist(), y)
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
        mae, rmse = models.score(y_test.tolist(), y)
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
        mae, rmse = models.score(y_test.tolist(), y)
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
        mae, rmse = models.score(y_test.tolist(), y)
        mae_dict[n] = mae
        rmse_dict[n] = rmse

    mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=['LR'])
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['LR'])

    return mae_df, rmse_df, lr