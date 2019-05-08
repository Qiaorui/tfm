from . import utils
import pandas as pd
from tqdm import tqdm
import math
import sklearn.metrics
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib
import itertools
import numpy as np
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

    def fit(self, x, y):
        print("Training...")

    def predict(self, x):
        print("Predicting...")
        y = None
        return y


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

    def fit(self, x, y):
        y = pd.DataFrame(y)
        y['Station_ID'] = x['Station_ID']
        groups = y.groupby(["Station_ID"])

        p = [0, 1, 2, 4, 6, 8, 10]
        q = d = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(v[0], v[1], v[2], 12) for v in list(itertools.product(p, d, q))]
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                sum_aic = 0
                for Station_ID, df in groups:
                    df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
                    try:
                        mod = sm.tsa.statespace.SARIMAX(df.drop("Station_ID", axis=1),
                                                        order=param,
                                                        seasonal_order=param_seasonal,
                                                        enforce_stationarity=False,
                                                        enforce_invertibility=False)
                        results = mod.fit(disp=0)
                        if results.aic is not np.nan:
                            sum_aic += results.aic
                    except:
                        continue
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, sum_aic/len(groups)))



    def predict(self, x):
        self.model = None


class SSA(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating SSA model")
        self.data = None


class VAR(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating VAR model")


class LTSM(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating LTSM model")
        self.data = None


class MLP(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating MLP model")
        self.data = None


class LR(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating LR model")
        self.data = None
