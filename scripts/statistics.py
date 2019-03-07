import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


DPI = 300

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


def analyse_trip_duration(df, start_year=None):
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['Stop_Time'] = pd.to_datetime(df['Stop_Time'])
    if start_year is not None:
        df = df.loc[df['Start_Time'].dt.year >= start_year]

    print(df.describe())
    print(df.info())

    # Plot Distribution of trip duration
    f = df['Trip_Duration'].value_counts()
    f.sort_index(inplace=True)
    plt.figure(dpi=DPI)
    plt.hist(df.loc[:, 'Trip_Duration'])
    plt.title('Distribution of Trip Durations')
    plt.xlabel('Duration (m)')
    plt.show()

    # Boxplot of trip duration
    plt.figure(dpi=DPI)
    plt.boxplot(list(df.loc[:, 'Trip_Duration']), 0, 'gD')
    plt.title('Boxplot')
    plt.show()

    # Boxplot without outlier of trip duration
    plt.figure(dpi=DPI)
    plt.boxplot(list(df.loc[:, 'Trip_Duration']), 0, '')
    plt.title('Boxplot without outlier')
    plt.show()

    # Plot of trip duration distribution for the trips within 60 minutes
    plt.figure(dpi=DPI)
    plt.title('Trip duration distribution within 60 minutes')
    plt.plot(f.index, f)
    plt.xlim(0, 60)
    plt.xlabel("Minute")
    plt.show()

    # Plot of trip duration distribution for same station pick-up and drop-off
    tmp = df[(df['Start_Station_ID']) == (df['End_Station_ID'])]
    f2 = tmp['Trip_Duration'].value_counts().sort_index()
    plt.figure(dpi=DPI)
    plt.title('Trip duration distribution which same station drop-off')
    plt.plot(f2.index, f2)
    plt.xlim(0, 60)
    plt.ylabel("Count")
    plt.xlabel("Minute")
    plt.show()

    # Plot of average distance between stations by trip duration
    avg_time = df.groupby('Trip_Duration')['Distance'].mean()
    plt.figure(dpi=DPI)
    plt.title('Average distance between station by trip duration')
    plt.plot(avg_time.index, avg_time)
    plt.xlim(0, 180)
    plt.xlabel("Duration (minute)")
    plt.ylabel("Distance (km)")
    plt.show()


def analyse_date_pattern(df):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Seasonly distribution
    bins = list(range(1, 5, 1))

    plt.bar(bins, df['Start_Season'].value_counts().sort_index())
    plt.xticks(bins, ['Spring', 'Summer', 'Autumn', 'Winter'])
    plt.xlabel('Seasons of the Year')
    plt.ylabel('Trip count')
    plt.title('Ridership by Season for NYC', fontsize=15)
    plt.show()

    # Monthly distribution

    bins = list(range(1, 13, 1))
    trip_duration = df.groupby('Start_Month')['Trip_Duration'].sum()

    fig, ax1 = plt.subplots(figsize=(15, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Month of the Year')
    ax1.set_ylabel('Trip time (hours)', color=color)
    ax1.bar(bins, trip_duration / 60, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.setp(ax1, xticks=bins,
             xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'may', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax1.set_title("Ridership by Month for NYC", fontsize=15)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Trip count', color=color)  # we already handled the x-label with ax1
    ax2.plot(bins, df['Start_Month'].value_counts().sort_index(), color=color, linewidth=3)
    ax2.set_ylim([0, max(df['Start_Month'].value_counts()) * 1.1])
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # Weekly distribution

    bins = list(range(1, 8, 1))
    trip_duration = df.groupby('Start_Weekday')['Trip_Duration'].sum()

    fig, ax1 = plt.subplots(figsize=(15, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Week days')
    ax1.set_ylabel('Trip time (hours)', color=color)
    ax1.bar(bins, trip_duration / 60, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.setp(ax1, xticks=bins, xticklabels=weekdays)
    ax1.set_title("Ridership by Weekday for NYC", fontsize=15)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Trip count', color=color)  # we already handled the x-label with ax1
    ax2.set_ylim([0, max(df['Start_Weekday'].value_counts()) * 1.1])
    ax2.plot(bins, df['Start_Weekday'].value_counts().sort_index(), color=color, linewidth=3)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # Hourly distribution

    bins = list(range(24))
    trip_duration = df.groupby('Start_Hour')['Trip_Duration'].sum()

    fig, ax1 = plt.subplots(figsize=(15, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Trip time (hours)', color=color)
    ax1.bar(bins, trip_duration / 60, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.setp(ax1, xticks=bins)
    ax1.set_title("Ridership by Hour for NYC", fontsize=15)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Trip count', color=color)  # we already handled the x-label with ax1
    ax2.plot(bins, df['Start_Hour'].value_counts().sort_index(), color=color, linewidth=3)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # Weekday x Hour distribution

    tmp = df.groupby(['Start_Weekday', 'Start_Hour'], sort=True).size().reset_index(name='counts')

    bins = list(range(24))
    plt.figure(figsize=(15, 7))
    plt.xlabel('Hourly distribution by weekday')
    plt.ylabel('Trip count')
    plt.title('Ridership by hour and weekday for NYC', fontsize=15)
    plt.xticks(bins)
    for i in range(1, 8, 1):
        plt.plot(bins, tmp[tmp['Start_Weekday'] == i]['counts'], linestyle='-', marker='o', label=weekdays[i - 1])
    plt.legend()
    plt.show()

