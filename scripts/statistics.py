import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def analyse_weather(df, start_year=None):
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    if start_year is not None:
        df = df.loc[df['Datetime'].dt.year >= start_year]

    print(df.describe())
    print(df.info())

    plt.figure(figsize=(20, 10))
    plt.title("Temperature")
    plt.xlabel("Date")
    plt.ylabel("Celsius")
    plt.plot(df['Datetime'].dt.date.unique(), df.groupby(df['Datetime'].dt.date)['Temperature'].mean())
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.title("Visibility")
    plt.xlabel("Date")
    plt.ylabel("Mile")
    plt.plot(df['Datetime'].dt.date.unique(), df.groupby(df['Datetime'].dt.date)['Visibility'].mean())
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.title("Humidity")
    plt.xlabel("Date")
    plt.ylabel("Percentage")
    plt.plot(df['Datetime'].dt.date.unique(), df.groupby(df['Datetime'].dt.date)['Humidity'].mean())
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.title("Wind")
    plt.xlabel("Date")
    plt.ylabel("KPH")
    plt.plot(df['Datetime'].dt.date.unique(), df.groupby(df['Datetime'].dt.date)['Wind'].mean())
    plt.show()


    plt.figure(figsize=(20, 10))
    plt.title("Weather Condition Distribution")
    gp = df.groupby(['Condition'])
    plt.bar(gp.size().index, gp.size())
    plt.show()

    """
    cloudy_conds = ["Clear", "Partly Cloudy", "Scattered Clouds", "Mostly Cloudy", "Haze", "Overcast"]
    foggy_conds = ["Fog", "Mist", "Light Freezing Fog"]
    rainy_conds = ["Heavy Rain", "Heavy Snow", 'Light Freezing Rain', 'Light Rain', 'Light Snow', "Rain", "Snow"]

    fig, ax = plt.subplots(figsize=(5, 5))

    cond_dict = dict((k, 0) for k in cloudy_conds)
    cond_dict.update(dict((k, 1) for k in foggy_conds))
    cond_dict.update(dict((k, 2) for k in rainy_conds))

    vals = [[], [], []]
    for k, v in gp.size().iteritems():
        vals[cond_dict[k]].append(v)

    size = 0.3

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(3) * 4)
    inner_colors = cmap(np.array(range(1, len(list(itertools.chain(*vals))))))

    ax.pie([sum(x) for x in vals], radius=1, colors=outer_colors, labels=["Cloudy", "Foggy", "Rainy"],
           wedgeprops=dict(width=size, edgecolor='w'))

    ax.pie(list(itertools.chain(*vals)), radius=1 - size, colors=inner_colors,
           wedgeprops=dict(width=size, edgecolor='w'))

    ax.set(aspect="equal", title='Weather Classification')
    plt.show()
    """


def analyse_trip(df):

    return None