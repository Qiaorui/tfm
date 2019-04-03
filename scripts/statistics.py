import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import utils
import folium
from folium.plugins import HeatMap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import branca.colormap as cm
import math
import webbrowser
import networkx as nx
import os
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


DPI = 300


def analyse_weather(df, start_year=None):
    if start_year is not None:
        df = df.loc[df['Datetime'].dt.year >= start_year]

    print(df.describe())
    print(df.info(memory_usage='deep'))

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

    """
    # Plot of average distance between stations by trip duration
    avg_time = df.groupby('Trip_Duration')['Distance'].mean()
    plt.figure(dpi=DPI)
    plt.title('Average distance between station by trip duration')
    plt.plot(avg_time.index, avg_time)
    plt.xlim(0, 180)
    plt.xlabel("Duration (minute)")
    plt.ylabel("Distance (km)")
    plt.show()
    """


def analyse_date_pattern(df):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Yearly and seasonally distribution
    men_means = df.groupby("Start_Year").get_group(2017)['Start_Season'].value_counts().sort_index()
    women_means = df.groupby("Start_Year").get_group(2018)['Start_Season'].value_counts().sort_index()
    bins = np.array((range(1, 5, 1)))

    width = 0.35  # the width of the bars
    plt.bar(bins - width/2, men_means, width, color='SkyBlue', label='2017')
    plt.bar(bins + width/2, women_means, width, color='IndianRed', label='2018')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Trip count')
    plt.title('Ridership by Season and Year')
    plt.xticks(bins, ['Spring', 'Summer', 'Autumn', 'Winter'])
    plt.legend()
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


def analyse_geo_pattern(df):
    lat, lng = df["Start_Latitude"].mean(), df["Start_Longitude"].mean()
    m = generate_dual_map([lat, lng], 14)

    day_count = (pd.to_datetime(df["Start_Time"]).max() - pd.to_datetime(df["Start_Time"]).min())/np.timedelta64(1, 'D')
    tmp = df.copy()
    tmp["Count"] = 1
    drop_offs = tmp[['End_Latitude', 'End_Longitude', 'Count']].groupby(
        ['End_Latitude', 'End_Longitude']).sum().reset_index()
    tmp = df.copy()
    tmp["Count"] = 1
    pick_ups = tmp[['Start_Latitude', 'Start_Longitude', 'Count']].groupby(
        ['Start_Latitude', 'Start_Longitude']).sum().reset_index()
    pick_ups["Count"] = np.ceil(pick_ups["Count"]/day_count)
    drop_offs["Count"] = np.ceil(drop_offs["Count"]/day_count)

    print(len(pick_ups), "stations plotting...")
    colormap = cm.linear.YlOrBr_05.scale(0, max(pick_ups['Count'].max(), drop_offs['Count'].max()))
    colormap.caption = 'Daily Pick up Distribution'

    for i in range(len(pick_ups)):
        folium.Circle(
            location=[pick_ups.iloc[i]['Start_Latitude'], pick_ups.iloc[i]['Start_Longitude']],
            radius= 70,
            color= "black",
            weight=2,
            #dash_array= '5,5',
            fill_opacity=1,
            popup= str(pick_ups.iloc[i]['Count']),
            fill_color= colormap(pick_ups.iloc[i]['Count'])
        ).add_to(m.m1)

    for i in range(len(drop_offs)):
        folium.Circle(
            location=[drop_offs.iloc[i]['End_Latitude'], drop_offs.iloc[i]['End_Longitude']],
            radius= 70,
            color= "black",
            weight=2,
            #dash_array= '5,5',
            fill_opacity=1,
            popup=str(drop_offs.iloc[i]['Count']),
            fill_color= colormap(drop_offs.iloc[i]['Count'])
        ).add_to(m.m2)

    #m.add_child(colormap)
    #m.m2.add_child(colormap)

    #folium.LayerControl(collapsed=False).add_to(m)
    m.save("map.html")
    webbrowser.open("file://" + os.path.realpath("map.html"))


def plot_stations(df, title="map"):
    print(len(df.index), "stations plotting...")

    lat, lng = df["Latitude"].mean(), df["Longitude"].mean()
    m = generate_base_map([lat, lng], 12)

    for _, row in df.iterrows():
        folium.Circle(
            location=[row['Latitude'], row['Longitude']],
            radius= 40,
            color= "green",
            fill_color= "green",
            fill_opacity=1,
            popup= str(row["Station_ID"]),
        ).add_to(m)

    m.save(title + ".html")
    webbrowser.open("file://" + os.path.realpath(title + ".html"))


def plot_diff_stations(df, df2, title="map"):
    print(len(df.index), "stations plotting...")

    lat, lng = df["Latitude"].mean(), df["Longitude"].mean()
    m = generate_base_map([lat, lng], 12)

    for _, row in df.iterrows():
        color = "red"
        id = row["Station_ID"]
        lat = row['Latitude']
        lng = row['Longitude']

        if ((df2['Station_ID'] == id) & np.isclose(df2['Latitude'], lat) & np.isclose(df2['Longitude'], lng)).any():
            color = "green"
        elif (df2['Station_ID'] == id).any() and (~np.isclose(df2['Latitude'], lat) & ~np.isclose(df2['Longitude'], lng)).all():
            color = "yellow"
        elif ((df2['Station_ID'] != id) & np.isclose(df2['Latitude'], lat) & np.isclose(df2['Longitude'], lng)).any():
            color = "blue"

        folium.Circle(
            location=[lat, lng],
            radius= 40,
            color= color,
            fill_color= color,
            fill_opacity=1,
            popup= str(id)
        ).add_to(m)

    m.save(title + ".html")
    webbrowser.open("file://" + os.path.realpath(title + ".html"))


def show_station_change(raw_trip_data_path, station_data, trip_data):
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
    plot_stations(location_raw_data)
    plot_diff_stations(location_raw_data, station_data)
    location_data = utils.get_station_list(trip_data)
    plot_stations(location_data)


def plot_unbalance_network(df):
    day_count = (pd.to_datetime(df["Start_Time"]).max() - pd.to_datetime(df["Start_Time"]).min()) / np.timedelta64(1,'D')
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

    for i, row in df_agg.iterrows():
        from_id, to_id, out_count = row["Start_Station_ID"], row['End_Station_ID'], row["Counts"]
        in_count_series = df_agg.loc[(df_agg['Start_Station_ID'] == to_id) & (df_agg['End_Station_ID'] == from_id), "Counts"]
        in_count = 0 if in_count_series.empty else in_count_series.values[0]
        row["Counts"] = in_count - out_count

    # Aggregate them to only positive edge
    df_agg = df_agg.loc[df_agg["Counts"] > 0]
    df_agg["Counts"] = df_agg["Counts"]/day_count

    # Plot the network
    g = nx.DiGraph()

    for _, row in df_agg.iterrows():
        from_id, to_id, count = row["Start_Station_ID"], row['End_Station_ID'], row["Counts"]
        g.add_edge(from_id, to_id, weight=count)
    edge_labels = dict([((u, v,), d['weight']) for u, v, d in g.edges(data=True)])

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
    values = [math.log(x+1) if x >= 0 else -math.log(-x+1) for x in values]
    nvmin = min(values)
    nvmax = max(values)
    nshift = max(abs(nvmax), abs(nvmin))

    # Shift all values
    values = [x + nshift for x in values]

    plt.figure(figsize=(20, 15))
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    nodes = nx.draw_networkx_nodes(g, stations, node_size=50, width=3, nodelist=list(balance_list.keys()),
                                   node_color=values, cmap=plt.get_cmap("PiYG"),
                                   vmin=0, vmax=nshift*2)
    edges = nx.draw_networkx_edges(g, stations, node_size=50, arrowstyle='-|>',
                                   arrowsize=6, edge_color=weights, edgelist=edges,
                                   edge_cmap=plt.cm.Blues, width=2)
    #nx.draw_networkx_labels(g, stations)

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
    plt.show()



def generate_base_map(location=[40.693943, -73.985880], zoom_start=12, tiles="Cartodb Positron"):
    base_map = folium.Map(location=location, control_scale=True, zoom_start=zoom_start, tiles=tiles)
    return base_map


def generate_dual_map(location=[40.693943, -73.985880], zoom_start=12, tiles="Cartodb Positron"):
    base_map = folium.plugins.DualMap(location=location, control_scale=True, zoom_start=zoom_start, tiles=tiles)
    return base_map
