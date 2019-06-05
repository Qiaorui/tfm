from .mySSA import mySSA
from . import utils
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
import keras
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow as tf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

pd.options.mode.chained_assignment = None

dropout = 0.


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
        self.model = {}

    """ Old version
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
                if len(search_results) > 2:
                    break
        search_results = sorted(search_results, key=lambda x: x[2])
        print("\nTop search results :")
        for param, param_seasonal, aic in search_results:
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, aic))

        return search_results[0][0], search_results[0][1]
    """

    def test(self, x, y, s, sids, show):
        y = pd.DataFrame(y)
        y['Station_ID'] = x['Station_ID']
        groups = y.groupby(["Station_ID"])

        # ADF stationary test
        adf_results = []
        for Station_ID, df in groups:
            res = adfuller(df['Count'].tolist())
            adf_results.append((Station_ID, res[0], res[1]))

        #for adf in adf_results:
        #    print('Station ', adf[0], ' ADF Statistic:', adf[1], 'p-value:', adf[2])

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
        filename = utils.get_next_filename("p")
        plt.savefig('results/' + filename + '.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()
            plt.close()
        # Partial Autocorrelation
        plot_pacf(station['Count'], lags=np.arange(100))
        filename = utils.get_next_filename("p")
        plt.savefig('results/' + filename + '.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()
            plt.close()

        # Grid Search
        p = range(3)
        q = range(3)
        d = range(2)
        options = list(itertools.product(p, d, q, p, d, q, [s]))
        options.reverse()

        station_parameters = {}
        for sid, df in tqdm(groups, leave=False, total=len(groups), unit="group", desc="Finding best parameter"):
            df = groups.get_group(sid)
            df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)

            diff_days = (df.index.max() - df.index.min()) // np.timedelta64(1, 'D')
            freq = df.index.freqstr

            gc.collect()
            search_results = []
            for p, d, q, P, D, Q, S in options:
                param = (p, d, q)
                param_seasonal = (P, D, Q, S)

                file_name = "SARIMA_{}_{}_{}_{}{}{}_{}{}{}x{}.pkl".format(diff_days, freq, sid, p, d, q, P, D, Q, S)
                exists = os.path.exists("model/" + file_name)
                results = None
                try:
                    if exists:
                        results = SARIMAXResults.load("model/" + file_name)
                    else:
                        mod = sm.tsa.statespace.SARIMAX(df.drop("Station_ID", axis=1),
                                                        order=param,
                                                        seasonal_order=param_seasonal)
                        results = mod.fit(disp=0)
                        # results.save("model/" + file_name)
                except Exception as e:
                    continue
                if not np.isnan(results.aic):
                    search_results.append((param, param_seasonal, results.aic, results))
                    if len(search_results) > 4:
                        break
            search_results = sorted(search_results, key=lambda x: x[2])

            station_parameters[sid] = search_results[0][:3]
            self.model[sid] = search_results[0][3]

        sum_aic = 0
        print("\nSelected Station Results :")
        for sid, (param, param_seasonal, aic) in station_parameters.items():
            sum_aic += aic
            print('Station {} :  SARIMA{}x{} - AIC:{}'.format(sid, param, param_seasonal, aic))

        print("\n Average AIC: ", sum_aic//len(sids))

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
        print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, sum_aic / len(groups)))

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
        y = [0 if i < 0 else i for i in y]
        return y


class SSA(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating SSA model")

    def test(self, x, y, s, sid, show):
        y = pd.DataFrame(y)
        y['Station_ID'] = x['Station_ID']
        groups = y.groupby(["Station_ID"])

        # Find contribution
        all_contrib = []
        for Station_ID, df in groups:
            df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
            ssa = mySSA(df['Count'])
            ssa.embed(embedding_dimension=40 if s < 40 else s*2, suspected_frequency=s)
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
        filename = utils.get_next_filename("p")
        plt.savefig('results/' + filename + '.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()
            plt.close()

        cumulative = np.cumsum(mean_contrib)
        plt.figure(figsize=(11, 4))
        plt.plot(range(len(mean_contrib)), cumulative)
        plt.xlabel("Singular i")
        plt.xticks(range(len(mean_contrib)), range(1, len(mean_contrib)+1))
        plt.axhline(y=0.8, color='r', linestyle='-')
        plt.title("Cumulative Contribution of Singular")
        filename = utils.get_next_filename("p")
        plt.savefig('results/' + filename + '.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()
            plt.close()

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
            filename = utils.get_next_filename("p")
            plt.savefig('results/' + filename + '.pdf', bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.clf()
                plt.close()

        ts_copy10 = ssa.ts.copy()
        reconstructed10 = ssa.view_reconstruction(*[ssa.Xs[i] for i in list(range(signal_size))], names=list(range(signal_size)), return_df=True, plot=False)
        ts_copy10['Reconstruction'] = reconstructed10.Reconstruction.values
        ts_copy10.plot(title='Original vs. Reconstructed Time Series')
        filename = utils.get_next_filename("p")
        plt.savefig('results/' + filename + '.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()
            plt.close()

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
        n = x.shape[1] # Number of features, number of neurons in input layer
        o = 1 # Number of neurons in output layer

        self.model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(n*2//3, n//3, n+o), solver='sgd', activation='tanh', verbose=True)
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
        y = [0 if i < 0 else i for i in y]
        return y


class LSTM(BaseModel):



    def __init__(self, n_pre, n_post):
        super().__init__()
        print("Creating LSTM model")
        self.n_pre = n_pre
        self.n_post = n_post

    """
          o o o
          ↑ ↑ ↑
    o➝o➝o➝o➝o➝o
    ↑ ↑ ↑
    o o o
    """
    def create_model_1(self, n_pre, n_feature, n_post, n_non_sec, hidden_dim=128):
        sequential_input_layer = keras.layers.Input(shape=(n_pre, n_feature))

        lstm_1_layer = keras.layers.LSTM(hidden_dim, dropout=dropout)(sequential_input_layer)
        repeat_layer = keras.layers.RepeatVector(self.n_post)(lstm_1_layer)
        lstm_2_layer = keras.layers.LSTM(hidden_dim, return_sequences=True, dropout=dropout)(repeat_layer)
        time_dense_layer = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_2_layer)
        activation_layer = keras.layers.Activation('linear')(time_dense_layer)
        flatten_layer = keras.layers.Flatten()(activation_layer)

        non_sequential_input_layer = keras.layers.Input(shape=(n_non_sec,))

        # Merging the second LSTM layer and non-sequential input layer
        merged = keras.layers.merge.concatenate([flatten_layer, non_sequential_input_layer])
        #dense_1_layer = keras.layers.Dense(hidden_dim)(merged)
        #dense_2_layer = keras.layers.Dense(hidden_dim)(dense_1_layer)
        output_layer = keras.layers.Dense(n_post)(merged)

        # Create keras model
        return keras.Model(inputs=[sequential_input_layer, non_sequential_input_layer], outputs=output_layer)

    """
    o o o
    ↑ ↑ ↑
    o➝o➝o
    ↑ ↑ ↑
    o o o
    """
    def create_model_2(self, n_pre, n_feature, n_post, n_non_sec, hidden_dim=128):
        sequential_input_layer = keras.layers.Input(shape=(n_pre, n_feature))

        lstm_1_layer = keras.layers.LSTM(hidden_dim, return_sequences=True, dropout=dropout)(sequential_input_layer)
        time_dense_layer = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_1_layer)
        activation_layer = keras.layers.Activation('linear')(time_dense_layer)
        flatten_layer = keras.layers.Flatten()(activation_layer)

        non_sequential_input_layer = keras.layers.Input(shape=(n_non_sec,))

        # Merging the second LSTM layer and non-sequential input layer
        merged = keras.layers.merge.concatenate([flatten_layer, non_sequential_input_layer])
        #dense_1_layer = keras.layers.Dense(hidden_dim)(merged)
        #dense_2_layer = keras.layers.Dense(hidden_dim)(dense_1_layer)
        output_layer = keras.layers.Dense(n_post)(merged)

        # Create keras model
        return keras.Model(inputs=[sequential_input_layer, non_sequential_input_layer], outputs=output_layer)

    """
          o o o
          ↑ ↑ ↑
    o➝o➝o➝o➝o➝o
    ↑ ↑ ↑ ↑ ↑ ↑
    o o o x x x
    """
    def create_model_3(self, n_pre, n_pre_feature, n_post, n_non_sec, n_post_feature, hidden_dim=128):

        # Define an input sequence and process it.
        encoder_inputs = keras.layers.Input(shape=(n_pre, n_pre_feature))
        encoder = keras.layers.LSTM(hidden_dim, return_state=True, dropout=dropout)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = keras.layers.Input(shape=(n_post, n_post_feature))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = keras.layers.LSTM(hidden_dim, return_sequences=True, return_state=True, dropout=dropout)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(n_post_feature, activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        flatten_layer = keras.layers.Flatten()(decoder_outputs)

        non_sequential_input_layer = keras.layers.Input(shape=(n_non_sec,))

        # Merging the second LSTM layer and non-sequential input layer
        merged = keras.layers.merge.concatenate([flatten_layer, non_sequential_input_layer])
        #dense_1_layer = keras.layers.Dense(hidden_dim)(merged)
        #dense_2_layer = keras.layers.Dense(hidden_dim)(dense_1_layer)
        output_layer = keras.layers.Dense(n_post)(merged)

        # Create keras model
        return keras.Model(inputs=[encoder_inputs, decoder_inputs, non_sequential_input_layer], outputs=output_layer)

    def fit(self, x_sec_train, x_non_sec_train, y_train, x_sec_test, x_non_sec_test, y_test, type, x_future_sec_train=None, x_future_sec_test=None, show=False):
        if 'Station_ID' in x_non_sec_train.columns:
            dum = pd.get_dummies(x_non_sec_train['Station_ID'], prefix="Station")
            self.data = dum.columns.values
            x_non_sec_train = np.hstack([x_non_sec_train.drop('Station_ID', axis=1), dum])
            dum = pd.get_dummies(x_non_sec_test['Station_ID'], prefix="Station")
            x_non_sec_test = np.hstack([x_non_sec_test.drop('Station_ID', axis=1), dum])

        assert x_sec_train.shape[1] % self.n_pre == 0
        x_sec_train = x_sec_train.values.reshape(x_sec_train.shape[0], self.n_pre, x_sec_train.shape[1] // self.n_pre)
        x_sec_test = x_sec_test.values.reshape(x_sec_test.shape[0], self.n_pre, x_sec_test.shape[1]// self.n_pre)
        if type == 3:
            x_future_sec_test = x_future_sec_test.values.reshape(x_future_sec_test.shape[0], self.n_post, x_future_sec_test.shape[1]//self.n_post)
            x_future_sec_train = x_future_sec_train.values.reshape(x_future_sec_train.shape[0], self.n_post, x_future_sec_train.shape[1]//self.n_post)

        dim = (x_sec_train.shape[2] + x_non_sec_train.shape[1]) * 2// 3 # 42

        batch_size = None

        model = None
        if type == 1:
            model = self.create_model_1(self.n_pre, x_sec_train.shape[2], self.n_post, x_non_sec_train.shape[1], dim)
        if type == 2:
            model = self.create_model_2(self.n_pre, x_sec_train.shape[2], self.n_post, x_non_sec_train.shape[1], dim)
        if type == 3:
            assert x_future_sec_train is not None
            assert x_future_sec_test is not None
            model = self.create_model_3(self.n_pre, x_sec_train.shape[2], self.n_post, x_non_sec_train.shape[1], x_future_sec_train.shape[2], dim)

        # Print the model summary
        print(model.summary())

        #tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
        model.compile(optimizer='adam', loss='mse')

        self.model = model

        # Train the model
        history = None
        if type == 3:
            history = model.fit([x_sec_train, x_future_sec_train, x_non_sec_train], y_train, batch_size=batch_size,
                                validation_data=([x_sec_test, x_future_sec_test, x_non_sec_test], y_test), epochs=10, verbose=0)
        else:
            history = model.fit([x_sec_train, x_non_sec_train], y_train, batch_size=batch_size, validation_data=([x_sec_test, x_non_sec_test], y_test), epochs=10, verbose=0)

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        filename = utils.get_next_filename("p")
        plt.savefig('results/' + filename + '.pdf', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()
            plt.close()

    def predict(self, x_sec, x_non_sec, x_future_sec=None):
        if 'Station_ID' in x_non_sec.columns:
            dum = pd.get_dummies(x_non_sec['Station_ID'], prefix="Station")
            if len(dum.columns.values) == 1:
                sid_column = dum.columns.values[0]
                dum = pd.DataFrame(np.zeros((len(x_non_sec.index), len(self.data)), dtype=np.int8), columns=self.data)
                dum.loc[:, sid_column] = 1
            x_non_sec = np.hstack([x_non_sec.drop('Station_ID', axis=1), dum])
        x_sec = x_sec.values.reshape(x_sec.shape[0], self.n_pre, x_sec.shape[1] // self.n_pre)
        y = None
        if x_future_sec is None:
            y = self.model.predict([x_sec, x_non_sec])
        else:
            x_future_sec = x_future_sec.values.reshape(x_future_sec.shape[0], self.n_post, x_future_sec.shape[1] // self.n_post)
            y = self.model.predict([x_sec, x_future_sec, x_non_sec])

        return np.array(list(map(lambda yy: [0 if i < 0 else i for i in yy], y)))