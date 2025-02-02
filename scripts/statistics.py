import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import utils
import folium
from folium.plugins import HeatMap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import branca.colormap as cm
import itertools
import math
import webbrowser
import networkx as nx
import os
from pandas.plotting import register_matplotlib_converters
import warnings
warnings.simplefilter("ignore")

register_matplotlib_converters()

DPI = 300


def analyse_weather(df, start_year, show):
    if start_year is not None:
        df = df.loc[df['Datetime'].dt.year >= start_year]

    print(df.describe())
    print(df.info(memory_usage='deep'))

    plt.figure(figsize=(20, 10))
    plt.title("Temperature")
    plt.xlabel("Date")
    plt.ylabel("Celsius")
    plt.plot(df['Datetime'].dt.date.unique(), df.groupby(df['Datetime'].dt.date)['Temperature'].mean())
    plt.savefig('results/p1.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    #plt.figure(figsize=(10, 10))
    plt.title("Histogram of Temperature")
    plt.xlabel("Celsius")
    plt.hist(df['Temperature'], range(math.floor(df['Temperature'].min()), math.ceil(df['Temperature'].max()+1), 1))
    plt.savefig('results/p2.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    plt.figure(figsize=(20, 10))
    plt.title("Visibility")
    plt.xlabel("Date")
    plt.ylabel("Mile")
    plt.plot(df['Datetime'].dt.date.unique(), df.groupby(df['Datetime'].dt.date)['Visibility'].mean())
    plt.savefig('results/p3.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    #plt.figure(figsize=(10, 10))
    plt.title("Histogram of Visibility")
    plt.xlabel("Mile")
    plt.hist(df['Visibility'], range(math.floor(df['Visibility'].min()), math.ceil(df['Visibility'].max()+1), 1))
    plt.savefig('results/p4.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    plt.figure(figsize=(20, 10))
    plt.title("Humidity")
    plt.xlabel("Date")
    plt.ylabel("Percentage")
    plt.plot(df['Datetime'].dt.date.unique(), df.groupby(df['Datetime'].dt.date)['Humidity'].mean())
    plt.savefig('results/p5.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    #plt.figure(figsize=(10, 10))
    plt.title("Histogram of Humidity")
    plt.xlabel("Percentage")
    plt.hist(df['Humidity'], bins=50)
    plt.savefig('results/p6.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    plt.figure(figsize=(20, 10))
    plt.title("Wind")
    plt.xlabel("Date")
    plt.ylabel("KPH")
    plt.plot(df['Datetime'].dt.date.unique(), df.groupby(df['Datetime'].dt.date)['Wind'].mean())
    plt.savefig('results/p7.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    #plt.figure(figsize=(10, 10))
    plt.title("Histogram of Wind")
    plt.xlabel("KPH")
    plt.hist(df['Wind'], range(math.floor(df['Wind'].min()), math.ceil(df['Wind'].max()+1), 1))
    plt.savefig('results/p8.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    plt.figure(figsize=(20, 10))
    plt.title("Weather Condition Distribution")
    gp = df.groupby(['Condition'])
    plt.bar(gp.size().index, gp.size())
    plt.savefig('results/p9.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    cloudy_conds = ["Clear", "Partly Cloudy", "Scattered Clouds", "Mostly Cloudy", "Haze", "Overcast"]
    foggy_conds = ["Fog", "Mist", "Light Freezing Fog"]
    rainy_conds = ["Heavy Rain", "Heavy Snow", 'Light Freezing Rain', 'Light Rain', 'Light Snow', "Rain", "Snow"]

    fig, ax = plt.subplots(figsize=(10, 10))

    cond_dict = dict((k, 0) for k in cloudy_conds)
    cond_dict.update(dict((k, 1) for k in foggy_conds))
    cond_dict.update(dict((k, 2) for k in rainy_conds))

    vals = [[], [], []]
    for k, v in gp.size().iteritems():
        vals[cond_dict[k]].append(v)
    size = 0.3

    outer_colors = []
    inner_colors = []
    cmap = plt.get_cmap("Greens")
    outer_colors.append(cmap(188))
    inner_colors.extend(cmap(np.arange(0, 256, 256 // (len(cloudy_conds) + 1))[1:]))

    cmap = plt.get_cmap("Reds")
    outer_colors.append(cmap(188))
    # inner_colors.extend(cmap(np.arange(0, 256, 256 // (len(foggy_conds)))[1:]))

    cmap = plt.get_cmap("Blues")
    outer_colors.append(cmap(188))
    inner_colors.extend(cmap(np.arange(0, 256, 256 // (len(rainy_conds) + 3))[1:]))

    ax.pie([sum(x) for x in vals], radius=1, colors=outer_colors, labels=["Cloudy", "Foggy", "Rainy"],
           autopct='%1.1f%%', pctdistance=0.83,
           wedgeprops=dict(width=size, edgecolor='w'), textprops={'fontsize': 24})

    ax.pie(list(itertools.chain(*vals)), radius=1 - size, colors=inner_colors,
           labels=cloudy_conds + foggy_conds + rainy_conds,
           labeldistance=0.6, rotatelabels=True, wedgeprops=dict(width=size, edgecolor='w'))
    fig.suptitle('Weather Classification', fontsize=24)
    ax.set(aspect="equal")
    plt.savefig('results/p10.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.cla()
        plt.close()

def analyse_trip_duration(df, start_year, show):
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['Stop_Time'] = pd.to_datetime(df['Stop_Time'])
    if start_year is not None:
        df = df.loc[df['Start_Time'].dt.year >= start_year]

    print(df.describe())
    print(df.info())

    max_triptime = df["Trip_Duration"].max()

    f = df['Trip_Duration'].value_counts()
    f.sort_index(inplace=True)

    # Plot Distribution of trip duration
    plt.figure(dpi=DPI)
    plt.hist(df.loc[:, 'Trip_Duration'], range(0, max_triptime, 5))
    plt.title('Distribution of Trip Duration')
    plt.xlabel('Duration (minutes)')
    plt.savefig('results/p21.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    # Plot Distribution of trip duration in log scale
    plt.figure(dpi=DPI)
    plt.yscale('log')
    plt.hist(df.loc[:, 'Trip_Duration'], range(0, max_triptime, 5))
    plt.title('Distribution of Trip Duration')
    plt.xlabel('Duration (minutes)')
    plt.savefig('results/p22.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    # Boxplot of trip duration
    plt.figure(dpi=DPI)
    plt.boxplot(list(df.loc[:, 'Trip_Duration']), 0, 'gD')
    plt.ylabel('Duration (minutes)')
    plt.title('Boxplot')
    plt.savefig('results/p23.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    # Boxplot without outlier of trip duration
    plt.figure(dpi=DPI)
    plt.boxplot(list(df.loc[:, 'Trip_Duration']), 0, '')
    plt.title('Boxplot without outliers')
    plt.ylabel('Duration (minutes)')
    plt.savefig('results/p24.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    # Plot of trip duration distribution for the trips within 60 minutes
    plt.figure(dpi=DPI)
    plt.title('Trip duration distribution within 60 minutes')
    plt.plot(f.index, f)
    plt.xlim(0, 60)
    plt.ylabel("Count")
    plt.xlabel("Minute")
    plt.savefig('results/p25.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    # Plot of trip duration distribution for same station pick-up and drop-off
    tmp = df[(df['Start_Station_ID']) == (df['End_Station_ID'])]
    f2 = tmp['Trip_Duration'].value_counts().sort_index()
    plt.figure(dpi=DPI)
    plt.title('Trip duration distribution of cyclic trip')
    plt.plot(f2.index, f2)
    plt.xlim(0, 60)
    plt.ylabel("Count")
    plt.xlabel("Minute")
    plt.savefig('results/p26.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    # Plot of stations which most user done the circulation trip
    tmp = tmp[["Start_Station_ID", "Start_Latitude", "Start_Longitude"]]
    round_trip_station = tmp.groupby(["Start_Station_ID", "Start_Latitude", "Start_Longitude"], as_index=False).size().reset_index(name="Round_Count").sort_values("Start_Station_ID")
    all_station_count = df.groupby("Start_Station_ID").size().to_frame("Total_Count").sort_values("Start_Station_ID")
    data = pd.merge(round_trip_station, all_station_count, on="Start_Station_ID")
    data["Pct"] = data["Round_Count"] / data["Total_Count"] * 100

    lat, lng = df["Start_Latitude"].mean(), df["Start_Longitude"].mean()
    m = generate_base_map([lat, lng], 14, tiles="OpenStreetMap")

    colormap = cm.linear.YlOrRd_04.scale(0, 20)
    colormap.caption = 'Percentage of Round Trip'

    for _, row in data.iterrows():
        folium.Circle(
            location=[row['Start_Latitude'], row['Start_Longitude']],
            radius=70,
            color="black",
            weight=2,
            # dash_array= '5,5',
            fill_opacity=1,
            popup=str(row['Start_Station_ID']),
            fill_color=colormap(row['Pct'])
        ).add_to(m)

    m.add_child(colormap)
    # folium.LayerControl(collapsed=False).add_to(m)
    m.save("results/map1.html")
    if show:
        webbrowser.open("file://" + os.path.realpath("results/map1.html"))


def analyse_date_pattern(df, show):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Yearly and seasonally distribution
    t17 = df.groupby("Start_Year").get_group(2017)['Start_Season'].value_counts().sort_index()
    t18 = df.groupby("Start_Year").get_group(2018)['Start_Season'].value_counts().sort_index()
    bins = np.array((range(1, 5, 1)))

    width = 0.35  # the width of the bars
    plt.bar(bins - width / 2, t17, width, color='SkyBlue', label='2017')
    plt.bar(bins + width / 2, t18, width, color='IndianRed', label='2018')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Trip count')
    plt.title('Ridership by Season and Year')
    plt.xticks(bins, ['Spring', 'Summer', 'Autumn', 'Winter'])
    plt.legend()
    plt.savefig('results/p27.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    # Monthly distribution

    bins = list(range(1, 13, 1))
    trip_duration = df.groupby('Start_Month')['Trip_Duration'].mean()

    fig, ax1 = plt.subplots(figsize=(15, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Month of the Year')
    ax1.set_ylabel('Average Trip Time by Trip (minutes)', color=color)
    ax1.bar(bins, trip_duration, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.setp(ax1, xticks=bins,
             xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'may', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax1.set_title("Ridership by Month", fontsize=15)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    average_month_trip_count = df.groupby(['Start_Month', 'Start_Year'], as_index=False).size().groupby(
        'Start_Month').mean().sort_index()
    color = 'tab:red'
    ax2.set_ylabel('Average Monthly Trip Count', color=color)  # we already handled the x-label with ax1
    ax2.plot(bins, average_month_trip_count, color=color, linewidth=3)
    ax2.set_ylim([0, max(average_month_trip_count) * 1.1])
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('results/p28.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.cla()
        plt.close()

    # Weekly distribution
    bins = list(range(1, 8, 1))
    trip_duration = df.groupby('Start_Weekday')['Trip_Duration'].mean()

    fig, ax1 = plt.subplots(figsize=(15, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Week days')
    ax1.set_ylabel('Average Trip Time by Trip (minutes)', color=color)
    ax1.bar(bins, trip_duration, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.setp(ax1, xticks=bins, xticklabels=weekdays)
    ax1.set_title("Ridership by Weekday", fontsize=15)

    tmp = df[["Start_Time", "Start_Weekday", "Start_Year"]].copy()
    tmp["Start_Weekday_Year"] = tmp["Start_Time"].dt.weekofyear
    avg_weekday_trip_count = tmp.groupby(['Start_Weekday', "Start_Year", "Start_Weekday_Year"]).size().groupby(
        "Start_Weekday").mean()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Average Daily Trip Count', color=color)  # we already handled the x-label with ax1
    ax2.set_ylim([0, max(avg_weekday_trip_count) * 1.1])
    ax2.plot(bins, avg_weekday_trip_count.sort_index(), color=color, linewidth=3)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('results/p29.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.cla()
        plt.close()

    # Hourly distribution
    bins = list(range(24))
    trip_duration = df.groupby('Start_Hour')['Trip_Duration'].mean()

    fig, ax1 = plt.subplots(figsize=(15, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Average Trip Time by Trip (minutes)', color=color)
    ax1.bar(bins, trip_duration, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.setp(ax1, xticks=bins)
    ax1.set_title("Ridership by Hour", fontsize=15)

    tmp = df[["Start_Time", "Start_Year", "Start_Hour"]].copy()
    tmp["Start_Day_Year"] = tmp["Start_Time"].dt.dayofyear
    avg_hour_trip_count = tmp.groupby(['Start_Hour', "Start_Year", "Start_Day_Year"]).size().groupby(
        "Start_Hour").mean()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Average Hourly Trip count', color=color)  # we already handled the x-label with ax1
    ax2.plot(bins, avg_hour_trip_count.sort_index(), color=color, linewidth=3)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('results/p30.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.cla()
        plt.close()

    # Weekday x Hour distribution

    tmp = df[["Start_Time", "Start_Hour", "Start_Weekday", "Start_Year"]].copy()
    tmp["Start_Weekday_Year"] = tmp["Start_Time"].dt.weekofyear
    tmp = tmp.groupby(['Start_Weekday', "Start_Hour", "Start_Year", "Start_Weekday_Year"]).size().groupby(
        ["Start_Weekday", "Start_Hour"]).mean().reset_index(name='counts')

    bins = list(range(24))
    plt.figure(figsize=(15, 7))
    plt.xlabel('Hourly distribution by weekday')
    plt.ylabel('Average Hourly Trip count')
    plt.title('Ridership by hour and weekday', fontsize=15)
    plt.xticks(bins)
    for i in range(1, 8, 1):
        plt.plot(bins, tmp[tmp['Start_Weekday'] == i]['counts'], linestyle='-', marker='o', label=weekdays[i - 1])
    plt.legend()
    plt.savefig('results/p31.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    # Holiday & Workingday relationship

    tmp = df[["Start_Time", "Start_Hour", "Start_Weekday", "Start_Year", "Start_Holiday"]].copy()
    tmp = df[df["Start_Weekday"] <= 5]
    tmp["Start_Day_Year"] = tmp["Start_Time"].dt.dayofyear
    tmp = tmp.groupby(['Start_Holiday', 'Start_Hour', "Start_Year", "Start_Day_Year"]).size().groupby(
        ['Start_Holiday', "Start_Hour"]).mean().reset_index(name='counts')

    bins = list(range(24))
    plt.figure(figsize=(15, 7))
    plt.xlabel('Hourly distribution')
    plt.ylabel('Average Hourly Trip count')
    plt.title('Ridership by hour', fontsize=15)
    plt.xticks(bins)
    for i in [True, False]:
        plt.plot(bins, tmp[tmp['Start_Holiday'] == i]['counts'], linestyle='-', marker='o', label="Holiday (Mon-Fri)" if i else "Workingday")
    plt.legend()
    plt.savefig('results/p32.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

def analyse_geo_pattern(df, title, show):
    lat, lng = df["Start_Latitude"].mean(), df["Start_Longitude"].mean()
    m = generate_dual_map([lat, lng], 14)

    day_count = (pd.to_datetime(df["Start_Time"]).max() - pd.to_datetime(df["Start_Time"]).min()) / np.timedelta64(1,
                                                                                                                   'D')
    tmp = df.copy()
    tmp["Count"] = 1
    drop_offs = tmp[['End_Latitude', 'End_Longitude', 'Count']].groupby(
        ['End_Latitude', 'End_Longitude']).sum().reset_index()
    tmp = df.copy()
    tmp["Count"] = 1
    pick_ups = tmp[['Start_Latitude', 'Start_Longitude', 'Count']].groupby(
        ['Start_Latitude', 'Start_Longitude']).sum().reset_index()
    pick_ups["Count"] = np.ceil(pick_ups["Count"] / day_count)
    drop_offs["Count"] = np.ceil(drop_offs["Count"] / day_count)

    print(len(pick_ups), "stations plotting...")
    colormap = cm.linear.YlOrBr_05.scale(0, max(pick_ups['Count'].max(), drop_offs['Count'].max()))
    colormap.caption = 'Daily Pick up Distribution'

    for i in range(len(pick_ups)):
        folium.Circle(
            location=[pick_ups.iloc[i]['Start_Latitude'], pick_ups.iloc[i]['Start_Longitude']],
            radius=70,
            color="black",
            weight=2,
            # dash_array= '5,5',
            fill_opacity=1,
            popup=str(pick_ups.iloc[i]['Count']),
            fill_color=colormap(pick_ups.iloc[i]['Count'])
        ).add_to(m.m1)

    for i in range(len(drop_offs)):
        folium.Circle(
            location=[drop_offs.iloc[i]['End_Latitude'], drop_offs.iloc[i]['End_Longitude']],
            radius=70,
            color="black",
            weight=2,
            # dash_array= '5,5',
            fill_opacity=1,
            popup=str(drop_offs.iloc[i]['Count']),
            fill_color=colormap(drop_offs.iloc[i]['Count'])
        ).add_to(m.m2)

    m.save("results/" + title + ".html")
    if show:
        webbrowser.open("file://" + os.path.realpath("results/" + title + ".html"))


def plot_stations(df, title, show):
    print(len(df.index), "stations plotting...")

    lat, lng = df["Latitude"].mean(), df["Longitude"].mean()
    m = generate_base_map([lat, lng], 12)

    for _, row in df.iterrows():
        folium.Circle(
            location=[row['Latitude'], row['Longitude']],
            radius=40,
            color="green",
            fill_color="green",
            fill_opacity=1,
            popup=str(row["Station_ID"]),
        ).add_to(m)

    m.save("results/" + title + ".html")
    if show:
        webbrowser.open("file://" + os.path.realpath("results/" + title + ".html"))


def plot_diff_stations(df, df2, title, show):
    print(len(df.index), "stations plotting...")

    lat, lng = df["Latitude"].mean(), df["Longitude"].mean()
    m = generate_base_map([lat, lng], 12)

    for _, row in df.iterrows():
        color = "red"
        sid = row["Station_ID"]
        lat = row['Latitude']
        lng = row['Longitude']

        if ((df2['Station_ID'] == sid) & np.isclose(df2['Latitude'], lat) & np.isclose(df2['Longitude'], lng)).any():
            color = "green"
        elif (df2['Station_ID'] == sid).any() and (
                ~np.isclose(df2['Latitude'], lat) & ~np.isclose(df2['Longitude'], lng)).all():
            color = "yellow"
        elif ((df2['Station_ID'] != sid) & np.isclose(df2['Latitude'], lat) & np.isclose(df2['Longitude'], lng)).any():
            color = "blue"

        folium.Circle(
            location=[lat, lng],
            radius=40,
            color=color,
            fill_color=color,
            fill_opacity=1,
            popup=str(sid)
        ).add_to(m)

    m.save("results/" + title + ".html")
    if show:
        webbrowser.open("file://" + os.path.realpath("results/" + title + ".html"))


def show_station_change(raw_trip_data_path, station_data, trip_data, show):
    location_raw_data = utils.read_raw_location_data(raw_trip_data_path)
    location_raw_data.rename(
        columns={'Start_Station_ID': 'Station_ID', "Start_Latitude": "Latitude", "Start_Longitude": "Longitude"},
        inplace=True
    )
    location_raw_data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    location_raw_data.fillna(0, inplace=True)
    location_raw_data['Station_ID'] = location_raw_data['Station_ID'].astype(np.int16)
    location_raw_data.drop_duplicates(inplace=True)
    location_raw_data.reset_index(inplace=True, drop=True)
    plot_stations(location_raw_data, "map3", show)
    plot_diff_stations(location_raw_data, station_data, "map4", show)
    location_data = utils.get_station_list(trip_data)
    plot_stations(location_data, "map5", show)


def plot_unbalance_network(df, filename, show):
    day_count = (pd.to_datetime(df["Start_Time"]).max() - pd.to_datetime(df["Start_Time"]).min()) / np.timedelta64(1,
                                                                                                                   'D')
    stations = utils.get_start_station_dict(df)

    # Get difference between two stations
    df_agg = df[["Start_Station_ID", "End_Station_ID"]].copy()
    df_agg = df_agg.loc[df_agg['End_Station_ID'].isin(list(stations.keys()))]

    df_agg = df_agg.groupby(["Start_Station_ID", "End_Station_ID"]).size().reset_index(name='Counts')

    # Find sum for each station
    out_list = df_agg.groupby('Start_Station_ID')['Counts'].sum().to_dict()
    in_list = df_agg.groupby('End_Station_ID')['Counts'].sum().to_dict()
    balance_list = dict()
    for s in stations.keys():
        balance_list[s] = (in_list.get(s, 0) - out_list.get(s, 0)) / day_count

    checked = []
    for i, row in df_agg.iterrows():
        from_id, to_id, out_count = row["Start_Station_ID"], row['End_Station_ID'], row["Counts"]

        in_count_series = df_agg.loc[
            (df_agg['Start_Station_ID'] == to_id) & (df_agg['End_Station_ID'] == from_id), "Counts"]
        in_count = 0 if in_count_series.empty else in_count_series.values[0]
        if (to_id, from_id) in checked:
            row["Counts"] = -in_count
        else:
            row["Counts"] = out_count - in_count
        checked.append((from_id, to_id))

    # Aggregate them to only positive edge
    df_agg = df_agg.loc[df_agg["Counts"] > 0]
    df_agg["Counts"] = df_agg["Counts"] / day_count

    # Plot the network
    g = nx.DiGraph()

    for _, row in df_agg.iterrows():
        from_id, to_id, count = row["Start_Station_ID"], row['End_Station_ID'], row["Counts"]
        g.add_edge(from_id, to_id, weight=count)

    # Sort edge by weight
    edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())
    tmp = list(map(list, zip(*sorted(list(zip(weights, edges))))))
    edges, weights = tmp[1], tmp[0]

    # Normalize node value
    values = list(balance_list.values())
    vmin = min(values)
    vmax = max(values)
    shift = max(abs(vmax), abs(vmin))
    # Normalize by log
    values = [math.log(x + 1) if x >= 0 else -math.log(-x + 1) for x in values]
    nvmin = min(values)
    nvmax = max(values)
    nshift = max(abs(nvmax), abs(nvmin))

    # Shift all values
    values = [x + nshift for x in values]

    fig = plt.figure(figsize=(10, 7))
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    nx.draw_networkx_nodes(g, stations, node_size=50, width=3, nodelist=list(balance_list.keys()),
                           node_color=values, cmap=plt.get_cmap("PiYG"),
                           vmin=0, vmax=nshift * 2)
    edges = nx.draw_networkx_edges(g, stations, node_size=50, arrowstyle='-|>',
                                   arrowsize=6, edge_color=weights, edgelist=edges,
                                   edge_cmap=plt.cm.Blues, width=2)
    # nx.draw_networkx_labels(g, stations)

    ax = plt.gca()

    divider = make_axes_locatable(ax)
    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(weights)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    cba = plt.colorbar(pc, cax=cax)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("PiYG"), norm=plt.Normalize(vmin=-shift, vmax=shift))
    sm._A = []
    cax = divider.append_axes("right", size="5%", pad=1.0)
    cbb = plt.colorbar(sm, cax=cax)

    cba.set_label('Route flow between stations')
    cbb.set_label('Average station balance')

    ax.set_axis_off()
    fig.tight_layout()
    plt.savefig("results/" + filename, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


def generate_base_map(location=[40.693943, -73.985880], zoom_start=12, tiles="Cartodb Positron"):
    base_map = folium.Map(location=location, control_scale=True, zoom_start=zoom_start, tiles=tiles)
    return base_map


def generate_dual_map(location=[40.693943, -73.985880], zoom_start=12, tiles="Cartodb Positron"):
    base_map = folium.plugins.DualMap(location=location, control_scale=True, zoom_start=zoom_start, tiles=tiles)
    return base_map


def analyse_demographic_pattern(raw_data_path, show):
    df = utils.read_raw_demographic_data(raw_data_path)
    df.loc[:, 'Trip_Duration'] = ((df.loc[:, 'Trip_Duration'] / 60).apply(np.ceil)).astype(np.int32)

    df.dropna(inplace=True)
    df["Age"] = 2019 - df["Birth_Year"]
    df.drop(df.loc[(df["Age"] > 90) | (df["Age"] < 4)].index, inplace=True)
    print(df.describe())

    # Gender distribution
    plt.axes(aspect='equal')
    plt.pie(df.groupby("Gender").size(), labels=['Masculine', 'Female'], autopct='%1.1f%%',
            pctdistance=0.6, labeldistance=1.05, radius=1)
    plt.title('Gender distribution')
    plt.savefig('results/p17.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    # Age & Trip duration relation
    tmp = df.groupby(['Gender', "Age"])["Trip_Duration"].mean().reset_index(name='counts')
    plt.figure(figsize=(15, 7))
    plt.xlabel('Age')
    plt.ylabel('Average Trip Duration (minutes)')
    plt.title('Gender & Age Ridership', fontsize=15)
    for i, label in enumerate(["Masculine", "Female"]):
        plt.plot(tmp[tmp['Gender'] == i + 1]['Age'], tmp[tmp['Gender'] == i + 1]['counts'], linestyle='-', marker='o',
                 label=label)
    plt.legend()
    plt.savefig('results/p18.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    # Schedule & Gender
    df["Hour"] = df["Start_Time"].dt.hour
    df["Day_Year"] = df["Start_Time"].dt.dayofyear
    df["Year"] = df["Start_Time"].dt.year
    bins = list(range(24))

    tmp = df.groupby(['Gender', "Hour", "Day_Year", "Year"]).size().groupby(["Gender", "Hour"]).mean().reset_index(
        name='counts')
    plt.figure(figsize=(15, 7))
    plt.xlabel('Hour')
    plt.ylabel('Average Hourly Trip Count')
    plt.title('Gender & Schedule relationship', fontsize=15)
    plt.xticks(bins)
    for i, label in enumerate(["Masculine", "Female"]):
        plt.plot(tmp[tmp['Gender'] == i + 1]['Hour'], tmp[tmp['Gender'] == i + 1]['counts'], linestyle='-', marker='o',
                 label=label)
    plt.legend()
    plt.savefig('results/p19.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    # Schedule & Age
    bins = list(range(24))

    tmp = df.groupby([pd.cut(df["Age"], np.arange(15, 60, 5)), "Hour", "Day_Year", "Year"]).size().groupby(
        ["Age", "Hour"]).mean().reset_index(name='counts')
    plt.figure(figsize=(15, 7))
    plt.xlabel('Hour')
    plt.ylabel('Average Hourly Trip Count')
    plt.title('Age & Schedule relationship', fontsize=15)
    plt.xticks(bins)
    for i in tmp["Age"].unique():
        plt.plot(tmp[tmp['Age'] == i]['Hour'], tmp[tmp['Age'] == i]['counts'], linestyle='-', marker='o', label=i)
    plt.legend()
    plt.savefig('results/p20.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


def analyse_weather_trip(df, show):
    temp = df.groupby("Temperature").agg({"Count":"mean"})
    plt.title("Hourly Trip Count by Temperature")
    plt.xlabel("Celsius")
    plt.ylabel("Average Hourly Trip Count")
    plt.plot(temp.index, temp["Count"])
    plt.savefig('results/p11.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    mt = temp.idxmax()
    print(df[df["Temperature"] == mt.tolist()[0]])

    vis = df.groupby("Visibility").agg({"Count":"mean"})
    plt.title("Hourly Trip Count by Visibility")
    plt.xlabel("Mile")
    plt.ylabel("Average Hourly Trip Count")
    plt.plot(vis.index, vis["Count"])
    plt.savefig('results/p12.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    hum = df.groupby("Humidity").agg({"Count":"mean"})
    plt.title("Hourly Trip Count by Humidity")
    plt.xlabel("Percentage")
    plt.ylabel("Average Hourly Trip Count")
    plt.plot(hum.index, hum["Count"])
    #plt.xlim((0,100))
    plt.savefig('results/p13.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    #mt = hum.sort_values("Count", ascending=False)
    #print(mt)
    print(df[df["Humidity"] <= 14])

    wind = df.groupby("Wind").agg({"Count":"mean"})
    plt.title("Hourly Trip Count by Wind")
    plt.xlabel("KPH")
    plt.ylabel("Average Hourly Trip Count")
    plt.plot(wind.index, wind["Count"])
    plt.savefig('results/p14.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    wind = df.groupby("Wind", as_index=False).agg({"Count":"mean"})
    wind = wind.groupby(pd.cut(wind["Wind"], np.arange(0, wind["Wind"].max()+1, 1))).sum()
    plt.title("Hourly Trip Count by Wind")
    plt.xlabel("KPH")
    plt.ylabel("Average Hourly Trip Count")
    plt.plot([x.left for x in wind.index], wind["Count"])
    plt.savefig('results/p15.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    plt.figure(figsize=(10, 7))
    cond = df.groupby("Condition")
    #indices = cond["Count"].mean().sort_values(ascending=False).index
    data = [tdf["Count"].to_numpy() for _, tdf in cond]
    plt.title("Hourly Trip Count by Weather Condition")
    plt.ylabel("Average Hourly Trip Count")

    bplot = plt.boxplot(data, 0, '', patch_artist=True)
    colors = ['lightgreen', 'pink', 'lightgreen', 'lightblue', 'lightblue', 'pink', 'lightblue', 'lightblue',
              'lightblue', 'pink', 'lightgreen', 'lightgreen', 'lightgreen', 'lightblue', 'lightgreen', 'lightblue']
    for b, c in zip(bplot["boxes"], colors):
        b.set_facecolor(c)

    plt.xticks(np.arange(len(list(cond.groups.keys())))+1, list(cond.groups.keys()), rotation=45)
    plt.savefig('results/p16.pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()
