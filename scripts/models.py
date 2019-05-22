from .mySSA import mySSA
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import numpy as np
import gc
import os
import sklearn.linear_model
import sklearn.neural_network
import sklearn.model_selection
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

pd.options.mode.chained_assignment = None


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
        y = [0 if i < 0 else i for i in y]
        return y


class LR(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating LR model")
        #self.normalizer = MinMaxScaler()

    def fit(self, x, y):
        if 'Station_ID' in x.columns:
            dum = pd.get_dummies(x['Station_ID'], prefix="Station")
            self.data = dum.columns.values
            x = np.hstack([x.drop('Station_ID', axis=1), dum])
        #y = np.array(y.values)
        #y = self.normalizer.fit_transform(y.reshape(-1, 1)).reshape(1,-1)[0]
        self.model = sklearn.linear_model.LinearRegression()
        self.model.fit(x, y)
        print("R squared:", self.model.score(x, y))

    def predict(self, x):
        if 'Station_ID' in x.columns:
            dum = pd.get_dummies(x['Station_ID'], prefix="Station")
            if len(dum.columns.values) == 1:
                sid_column = dum.columns.values[0]
                dum = pd.DataFrame(np.zeros((len(x.index), len(self.data)), dtype=np.int8), columns=self.data)
                dum.loc[:, sid_column] = 1
            x = np.hstack([x.drop('Station_ID', axis=1), dum])
        y = self.model.predict(x)
        #y = self.normalizer.inverse_transform(y.reshape(-1, 1))
        #return y.reshape(1, -1)[0]
        return y


class MLP(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating MLP model")
        #self.normalizer = MinMaxScaler()

    def test(self, x, y):
        if 'Station_ID' in x.columns:
            dum = pd.get_dummies(x['Station_ID'], prefix="Station")
            self.data = dum.columns.values
            x = np.hstack([x.drop('Station_ID', axis=1), dum])
        #y = np.array(y.values)
        #y = self.normalizer.fit_transform(y.reshape(-1, 1)).reshape(1,-1)[0]

        n = x.shape[1] # Number of features, number of neurons in input layer
        o = 1 # Number of neurons in output layer
        max_layer_number = 3 # Max 3 layers, 39 combinations

        layers = []
        for i in range(1, max_layer_number + 1):
            layers.extend(list(itertools.product([n // 3, n *2// 3, n + o], repeat=i)))
        parameter_space = {
            'hidden_layer_sizes': layers,
            'solver': ['sgd', 'adam'],
            'activation': ['tanh', 'relu']
        }
        mlp = sklearn.neural_network.MLPRegressor()
        ms = sklearn.model_selection.GridSearchCV(mlp, parameter_space, scoring='neg_mean_squared_error', cv=3)
        ms.fit(x, y)
        print("Best parameters found:\n", ms.best_params_)
        means = ms.cv_results_['mean_test_score']
        stds = ms.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, ms.cv_results_['params']):
            print("{:.2f} (+/- {:.2f}) for {}".format(mean, std, params))
        self.model = ms

    def fit(self, x, y):
        if 'Station_ID' in x.columns:
            dum = pd.get_dummies(x['Station_ID'], prefix="Station")
            self.data = dum.columns.values
            x = np.hstack([x.drop('Station_ID', axis=1), dum])
        #y = np.array(y.values)
        #y = self.normalizer.fit_transform(y.reshape(-1, 1)).reshape(1,-1)[0]
        self.model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(64, 64), verbose=True)
        self.model.fit(x, y)

    def predict(self, x):
        if 'Station_ID' in x.columns:
            dum = pd.get_dummies(x['Station_ID'], prefix="Station")
            if len(dum.columns.values) == 1:
                sid_column = dum.columns.values[0]
                dum = pd.DataFrame(np.zeros((len(x.index), len(self.data)), dtype=np.int8), columns=self.data)
                dum.loc[:, sid_column] = 1
            x = np.hstack([x.drop('Station_ID', axis=1), dum])
        y = self.model.predict(x)
        #y = self.normalizer.inverse_transform(y.reshape(-1, 1))
        #return y.reshape(1, -1)[0]
        return y


class LTSM(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating LTSM model")

    def test(self, x, y):
        return None

    def fit(self, x, y):
        return None

    def predict(self, x):
        return None