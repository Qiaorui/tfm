from . import utils
from .mySSA import mySSA
import pandas as pd
from tqdm import tqdm
import math
import sklearn.metrics
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib
import itertools
import numpy as np
import gc
import os

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

pd.options.mode.chained_assignment = None


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


def score(y_true, y_pred):
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print("MAE:", mae, "  RMSE:", rmse)
    return mae, rmse


class BaseModel:
    def __init__(self):
        self.data = None
        self.model = None


class HA(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating HA model")

    def fit(self, x, y):
        x['Count'] = y
        self.model = x.groupby(["Station_ID", "Weekday_Cos", "Time_Fragment_Cos"])['Count'].mean()

    def predict(self, x):
        y = []
        for idx, row in tqdm(x.iterrows(), leave=False, total=len(x.index), unit="row", desc="Predicting"):
            value = self.model.loc[(row["Station_ID"], row["Weekday_Cos"], row["Time_Fragment_Cos"])]
            y.append(value)
        return y


class ARIMA(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating ARIMA model")

    def test(self, x, y, s, sids):
        y = pd.DataFrame(y)
        y['Station_ID'] = x['Station_ID']
        groups = y.groupby(["Station_ID"])

        # ADF stationary test
        adf_results = []
        for Station_ID, df in tqdm(groups, leave=False, total=len(groups), unit="group", desc="ADF test"):
            res = adfuller(df['Count'].tolist())
            adf_results.append((Station_ID, res[0], res[1]))

        for adf in adf_results:
            print('Station ', adf[0], ' ADF Statistic:', adf[1], 'p-value:', adf[2])

        non_stationary = [(sid, adf, p) for sid, adf, p in adf_results if p > 0.01]
        if not non_stationary:
            print("\nAll stations followed stationary time series")
        else:
            print("\nSome stations may be non-stationary")
            for adf in non_stationary:
                print('Station ', adf[0], ' ADF Statistic:', adf[1], 'p-value:', adf[2])

        # Autocorrelation
        station = groups.get_group(sids[0])
        plot_acf(station['Count'], lags=np.arange(100))
        plt.show()
        # Partial Autocorrelation
        plot_pacf(station['Count'], lags=np.arange(100))
        plt.show()

        # Grid Search
        p = range(3)
        q = range(3)
        d = range(2)
        options = list(itertools.product(p, d, q, p, d, q, [s]))
        options.reverse()

        search_results = []
        for p, d, q, P, D, Q, S in tqdm(options, leave=False, total=len(options), unit="option", desc="ARIMA order"):
            sum_aic = 0
            param = (p, d, q)
            param_seasonal = (P, D, Q, S)

            gc.collect()
            for sid in sids:
                df = groups.get_group(sid)
                df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)

                diff_days = (df.index.max() - df.index.min())//np.timedelta64(1,'D')
                freq = df.index.freqstr

                file_name = "SARIMA_{}_{}_{}_{}{}{}_{}{}{}x{}.pkl".format(diff_days, freq, sid, p,d,q,P,D,Q,S)
                exists = os.path.exists("model/" + file_name)
                try:
                    results = None
                    if exists:
                        results = SARIMAXResults.load("model/" + file_name)
                    else:
                        mod = sm.tsa.statespace.SARIMAX(df.drop("Station_ID", axis=1),
                                                    order=param,
                                                    seasonal_order=param_seasonal)
                        results = mod.fit(disp=0)
                        #results.save("model/" + file_name)
                    if np.isnan(results.aic):
                        sum_aic = np.nan
                        break
                    sum_aic += results.aic
                except Exception as e:
                    print(str(e))
                    sum_aic = np.nan
                    break
            if not np.isnan(sum_aic):
                search_results.append((param, param_seasonal, sum_aic/len(sids)))
        search_results = sorted(search_results, key=lambda x: x[2])
        print("\nTop search results :")
        for param, param_seasonal, aic in search_results:
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, aic))

        return search_results[0][0], search_results[0][1]

    def fit(self, x, y, param, param_seasonal):
        self.model = {}

        y = pd.DataFrame(y)
        y['Station_ID'] = x['Station_ID']
        groups = y.groupby(["Station_ID"])
        sum_aic = 0
        for Station_ID, df in tqdm(groups, leave=False, total=len(groups), unit="group"):
            df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
            try:
                mod = sm.tsa.statespace.SARIMAX(df.drop("Station_ID", axis=1),
                                                order=param,
                                                seasonal_order=param_seasonal)
                results = mod.fit(disp=0)
                if np.isnan(results.aic):
                    sum_aic = np.nan
                    break
                sum_aic += results.aic
                self.model[Station_ID] = results
            except Exception as e:
                print(str(e))
                sum_aic = np.nan
                break
        print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, sum_aic / len(groups)))

    def predict(self, x):
        y = []
        sid = -1
        size = 0
        for idx, row in tqdm(x.iterrows(), leave=False, total=len(x.index), unit="row", desc="Predicting"):
            if row['Station_ID'] != sid:
                if size > 0:
                    pred = self.model[sid].forecast(size)
                    y.extend(list(pred))
                size = 0
                sid = row['Station_ID']
            size += 1

        pred = self.model[sid].forecast(size)
        y.extend(list(pred))
        return y


class SSA(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating SSA model")

    def test(self, x, y, s, sid):
        y = pd.DataFrame(y)
        y['Station_ID'] = x['Station_ID']
        groups = y.groupby(["Station_ID"])

        # Find contribution
        all_contrib = []
        for Station_ID, df in groups:
            df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
            ssa = mySSA(df['Count'])
            ssa.embed(embedding_dimension=40, suspected_frequency=s)
            ssa.decompose()
            contrib = list(itertools.chain.from_iterable(ssa.s_contributions.values))
            if not all_contrib:
                all_contrib = [[] for _ in range(len(contrib))]
            for idx, value in enumerate(contrib):
                all_contrib[idx].append(value)
        mean_contrib = [np.mean(x) for x in all_contrib]

        plt.figure(figsize=(11, 4))
        plt.bar(range(len(mean_contrib)), mean_contrib)
        plt.xlabel("Singular i")
        plt.xticks(range(len(mean_contrib)), range(1, len(mean_contrib)+1))
        plt.title("contribution of Singular")
        plt.show()

        cumulative = np.cumsum(mean_contrib)
        plt.figure(figsize=(11, 4))
        plt.plot(range(len(mean_contrib)), cumulative)
        plt.xlabel("Singular i")
        plt.xticks(range(len(mean_contrib)), range(1, len(mean_contrib)+1))
        plt.axhline(y=0.8, color='r', linestyle='-')
        plt.title("Cumulative Contribution of Singular")
        plt.show()

        #signal_size = next(x[0] for x in enumerate(cumulative) if x[1] > 0.8) + 1
        signal_size = 5

        # Plot the reconstruction
        station = groups.get_group(sid)
        ssa = mySSA(station['Count'])
        ssa.embed(embedding_dimension=50, suspected_frequency=s)
        ssa.decompose()

        for i in range(signal_size):
            plt.figure(figsize=(11,2))
            ssa.view_reconstruction(ssa.Xs[i], names=i, symmetric_plots=i != 0)
            plt.show()

        ts_copy10 = ssa.ts.copy()
        reconstructed10 = ssa.view_reconstruction(*[ssa.Xs[i] for i in list(range(signal_size))], names=list(range(signal_size)), return_df=True, plot=False)
        ts_copy10['Reconstruction'] = reconstructed10.Reconstruction.values
        ts_copy10.plot(title='Original vs. Reconstructed Time Series')
        plt.show()

        return signal_size

    def fit(self, x, y, s):
        self.model = {}

        y = pd.DataFrame(y)
        y['Station_ID'] = x['Station_ID']
        groups = y.groupby(["Station_ID"])
        for Station_ID, df in tqdm(groups, leave=False, total=len(groups), unit="group"):
            df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
            self.model[Station_ID] = mySSA(df['Count'])
            self.model[Station_ID].embed(embedding_dimension=50, suspected_frequency=s)
            self.model[Station_ID].decompose()

    def predict(self, x, signal_size):
        y = []
        sid = -1
        size = 0
        for idx, row in tqdm(x.iterrows(), leave=False, total=len(x.index), unit="row", desc="Predicting"):
            if row['Station_ID'] != sid:
                if size > 0:
                    pred = self.model[sid].forecast_recurrent(steps_ahead=size, singular_values=list(range(signal_size)), return_df=True)
                    pred = pred.tail(size)['Forecast'].values
                    y.extend(list(pred))
                size = 0
                sid = row['Station_ID']
            size += 1

        pred = self.model[sid].forecast_recurrent(steps_ahead=size, singular_values=list(range(signal_size)),
                                                  return_df=True)
        pred = pred.tail(size)['Forecast'].values
        y.extend(list(pred))
        return y


class LR(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating LR model")

    def fit(self, x, y):
        self.model = {}

        y = pd.DataFrame(y)
        y['Station_ID'] = x['Station_ID']
        groups = y.groupby(["Station_ID"])
        sum_aic = 0
        for Station_ID, df in tqdm(groups, leave=False, total=len(groups), unit="group"):
            df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)


    def predict(self, x):
        return None


class MLP(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating MLP model")


class LTSM(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating LTSM model")
