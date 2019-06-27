from . import utils
import subprocess
import os
import numpy as np
import math
from pandas.tseries.holiday import USFederalHolidayCalendar


def preprocess_trips(raw_path, dest_path, stations):
    df = utils.read_raw_trip_data(raw_path)
    if df is None:
        return False
    print("Raw size:", df.info(memory_usage='deep'))

    print(df.describe())
    print(df.isnull().sum())
    print('Contains ', sum(df.isnull().sum()), ' NULL values.')
    print('Dropping null or incorrect rows')
    complete_size = len(df.index)
    df.dropna(subset=["Start_Time", "Stop_Time"], inplace=True)
    df.dropna(how='all', subset=["Start_Station_ID", "Start_Latitude", "Start_Longitude"], inplace=True)
    df.dropna(how='all', subset=["End_Station_ID", "End_Latitude", "End_Longitude"], inplace=True)

    complete_station(df, stations)

    low_freq = df['Start_Station_ID'].value_counts()
    low_freq = low_freq[low_freq < 10]
    print("Dropping:", low_freq.keys())
    df.drop(df.loc[df['Start_Station_ID'].isin(low_freq.keys())].index, inplace=True)

    df['Start_Station_ID'] = df['Start_Station_ID'].astype(np.int16)
    df['End_Station_ID'] = df['End_Station_ID'].astype(np.int16)

    print("Removed", complete_size - len(df.index), "rows")

    df.loc[:, 'Trip_Duration'] = ((df.loc[:, 'Trip_Duration'] / 60).apply(np.ceil)).astype(np.int32)

    df['Start_Hour'] = df['Start_Time'].dt.hour.astype(np.int8)
    df['Start_Weekday'] = (df['Start_Time'].dt.weekday + 1).astype(np.int8)
    df['Start_Month'] = df['Start_Time'].dt.month.astype(np.int8)
    df['Start_Season'] = df['Start_Time'].dt.quarter.astype(np.int8)
    df['Start_Year'] = df['Start_Time'].dt.year.astype(np.int16)

    df['Stop_Hour'] = df['Stop_Time'].dt.hour.astype(np.int8)
    df['Stop_Weekday'] = (df['Stop_Time'].dt.weekday + 1).astype(np.int8)
    df['Stop_Month'] = df['Stop_Time'].dt.month.astype(np.int8)
    df['Stop_Season'] = df['Stop_Time'].dt.quarter.astype(np.int8)
    df['Stop_Year'] = df['Stop_Time'].dt.year.astype(np.int16)

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df['Start_Time'].min(), end=df['Start_Time'].max())
    print('From {} until {}'.format(df['Start_Time'].min(), df['Start_Time'].max()))

    df['Start_Holiday'] = df['Start_Time'].dt.normalize().isin(holidays)
    df['Stop_Holiday'] = df['Stop_Time'].dt.normalize().isin(holidays)

    print(len(df[df['Start_Holiday'] == True]), 'trips done during holidays')

    df.to_csv(dest_path, index=False, chunksize=math.ceil(len(df.index) / 20))
    print(dest_path, "created")

    return True


def preprocess_weathers(raw_path, dest_path):
    df = utils.read_raw_weather_data(raw_path)

    print(df.isnull().sum())
    print('Contains ', sum(df.isnull().sum()), ' NULL values. Need to impute :', sum(df.isnull().sum()) > 0)

    target_script = os.path.join(os.path.dirname(__file__), 'preprocess.R')
    subprocess.call(["Rscript", target_script, raw_path, dest_path], shell=False)


def preprocess_locations(raw_path, dest_path):
    df = utils.read_raw_location_data(raw_path)

    print(df.isnull().sum())
    df.rename(columns={'Start_Latitude': 'Latitude', 'Start_Longitude': 'Longitude', "Start_Station_ID": "Station_ID",
                       "Start_Station_Name": "Station_Name", }, inplace=True)
    df = df.groupby(["Station_ID", "Station_Name", "Latitude", "Longitude"], as_index=False).agg(
        {"Start_Time": [np.min, np.max]})
    df.reset_index()
    cols = [x if not y else y for (x, y) in df.columns.values.tolist()]

    df.columns = cols
    df.rename(columns={'amin': "First_Time", 'amax': "Last_Time"}, inplace=True)

    df.to_csv(dest_path, index=False)
    print(dest_path, "created")
    return True


def remove_trip_outlier(df, th):
    # delete all rows with column 'Trip_Duration' has value more than defined threshold
    indexNames = df[df['Trip_Duration'] >= th].index
    df.drop(indexNames, inplace=True)


def complete_station(df, stations):
    print("Repairing station data...")
    thresh = 1  # 1km as threshold

    # Get Start station list
    ss = utils.get_station_list(df)
    ss.dropna(how="all", inplace=True)

    stations_list = stations.to_dict('records')
    stations.set_index("Station_ID", inplace=True, drop=False, verify_integrity=True)
    print("total", len(ss.index), " stations")

    closeness = ss.apply(lambda x: utils.closest(x, data=stations_list), axis=1, result_type="expand")
    complete_cases = closeness[closeness.Station_ID.isin(closeness[closeness.isnull().any(1)]["Station_ID"])].dropna()
    for _, row in complete_cases.iterrows():
        sid, cid = row["Station_ID"], row["Closest_Station_ID"]
        closeness.loc[(closeness["Station_ID"] == sid) & np.isnan(closeness["Latitude"]),
                      ["Closest_Station_ID"]] = cid

    print(closeness.sort_values("Distance", ascending=False))

    # To remove
    print("Remove following rows")
    remove_list = closeness[(closeness["Distance"] >= thresh) | (np.isnan(closeness["Closest_Station_ID"]))]
    print(remove_list)
    for _, row in remove_list.dropna().iterrows():
        sid, lat, lng = row["Station_ID"], row["Latitude"], row["Longitude"]
        df.drop(
            df.loc[((df['Start_Station_ID'] == sid) & (df['Start_Latitude'] == lat) & (df['Start_Longitude'] == lng))
                   | ((df['End_Station_ID'] == sid) & (df['End_Latitude'] == lat) &
                      (df['End_Longitude'] == lng))].index, inplace=True)

    closeness = closeness[~((closeness["Distance"] >= thresh) | (np.isnan(closeness["Closest_Station_ID"])))]
    # To displace
    print("Displace incomplete station info")
    for _, row in closeness[closeness.isnull().any(1)].iterrows():
        sid, cid = row["Station_ID"], row["Closest_Station_ID"]
        info = get_station_info(stations, cid)
        df.loc[(df["Start_Station_ID"] == sid) & np.isnan(df["Start_Latitude"]),
               ["Start_Station_ID", "Start_Latitude", "Start_Longitude"]] = info
        df.loc[(df["End_Station_ID"] == sid) & np.isnan(df["End_Latitude"]),
               ["End_Station_ID", "End_Latitude", "End_Longitude"]] = info

    for _, row in closeness.dropna().iterrows():
        sid, lat, lng = row["Station_ID"], row["Latitude"], row["Longitude"]
        cid = row["Closest_Station_ID"]
        info = get_station_info(stations, cid)
        df.loc[(df["Start_Station_ID"] == sid) & (df["Start_Latitude"] == lat) & (df["Start_Longitude"] == lng),
               ["Start_Station_ID", "Start_Latitude", "Start_Longitude"]] = info
        df.loc[(df["End_Station_ID"] == sid) & (df["End_Latitude"] == lat) & (df["End_Longitude"] == lng),
               ["End_Station_ID", "End_Latitude", "End_Longitude"]] = info

    df.drop(df.loc[~df['End_Station_ID'].isin(df["Start_Station_ID"].tolist())].index, inplace=True)
    df.dropna(inplace=True)


def get_station_info(df, sid):
    return df.loc[sid, ["Station_ID", "Latitude", "Longitude"]].tolist()


def aggregate_stations(stations):
    df = stations.set_index("Station_ID", drop=False, verify_integrity=True)
    threashold = 0.13 # unit in km

    while True:
        closeness = df.apply(lambda x: utils.closest(x, data=df[df['Station_ID']!=x['Station_ID']].to_dict('records')), axis=1, result_type="expand")
        remove_list = []
        for _, row in closeness[(closeness["Distance"] <= threashold) & (closeness['Station_ID'] < closeness['Closest_Station_ID'])].iterrows():
            if row['Closest_Station_ID'] in remove_list or row['Station_ID'] in remove_list:
                continue
            docks, lat, lng, sname = df.loc[row['Closest_Station_ID'], ['Docks', 'Latitude', 'Longitude', 'Station_Name']]
            df.loc[row["Station_ID"], ['Docks', 'Latitude', 'Longitude', 'Station_Name']] = [row['Docks'] + docks, np.mean([lat, row['Latitude']]), np.mean([lng, row['Longitude']]), row['Station_Name'] + " and " + sname]
            remove_list.append(row['Closest_Station_ID'])

        if len(remove_list) == 0:
            break
        print("Aggregating following station:", remove_list)
        df = df[~df['Station_ID'].isin(remove_list)]

    return df.reset_index(drop=True)
