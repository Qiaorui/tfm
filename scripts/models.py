from . import utils
import pandas as pd
from tqdm import tqdm
import math
import sklearn.metrics


def convert_to_sequence(df, output_columns, lags=0, aheads=1, dropnan=True):
    new_df = pd.DataFrame()
    x_columns = []
    # Add lags (t-lag, t-lag+1, t-lag+2, ... , t-1)
    for lag in range(lags, 0, -1):
        for column in df.columns:
            new_column_name = column+"_lag_"+str(lag)
            new_df[new_column_name] = df[column].shift(lag).values
            x_columns.append(new_column_name)
    # Add current observation (t)
    for column in df.columns:
        new_df[column] = df[column].values
        x_columns.append(column)
    # Add ste aheads (t+1, t+2, ... , t+aheads)
    y_columns = []
    for ahead in range(1, aheads+1, 1):
        for output_column in output_columns:
            new_column_name = output_column+"_ahead_"+str(ahead)
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


class ARIMA(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating ARIMA model")



class SSA(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating SSA model")
        self.data = None


class VAR(BaseModel):
    def __init__(self):
        super().__init__()
        print("Creating VAR model")


