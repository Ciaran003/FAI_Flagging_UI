import pandas as pd
import numpy as np
# import sunpy.timeseries
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# from astropy.visualization import time_support
from sunpy.time import TimeRange
from sunkit_instruments import goes_xrs #type:ignore
from sunpy.timeseries.sources.goes import XRSTimeSeries
from astropy import units as u
import matplotlib.dates as mdates

diff_time=300
# sat_no=None
metadata=None
unit=None

def process_data(data):
    """Process data to have only xrsa/xrsb and SatelliteNumber"""
    # data["time_tag"] = pd.to_datetime(data["time_tag"].str.replace("T", " ").str.replace("Z", ""), errors='coerce')
    data["time_tag"] = pd.to_datetime(data["time_tag"], errors='coerce')
    # Pivot only on energy
    df = data.pivot_table(
        index="time_tag",
        columns="energy",
        values="flux")
    df = df.rename(columns={
        "0.05-0.4nm": "xrsa",
        "0.1-0.8nm": "xrsb"})
    df['satellite'] = data['satellite'].iloc[0]
    return df

def timeseries(data):
    """
    create timeseries of data\n
    recreates metadata & units required for data to be usable in further functions
    """
    global metadata, unit
    # data=get_data()
    df1 = process_data(data)
    sat = int(df1["satellite"].iloc[0])
    df=pd.DataFrame()
    df["xrsa"]=df1["xrsa"]
    df["xrsb"]=df1["xrsb"]
    # sat_no=sat
    meta = {
        "instrument": "GOES-XRS",
        "observatory": f"GOES-{sat}",
        "telescop": f"GOES-{sat}",
        "goes": sat,
        "wave": {"xrsa": (0.5, 4),"xrsb": (1, 8)},
        # "satellite": sat
        }
    units = {"xrsa": u.W / u.m**2,"xrsb": u.W / u.m**2}
    metadata=meta
    unit=units
    ts = XRSTimeSeries(df, meta=meta, units=units)
    ts.goes_number = sat
    return ts

def running_difference(data,diff_time):
    """
    Create time series of running differences. represents flux(t) - flux(t - diff_time)
    Recreate in a new XRStimeseries, with original metadata
    """
    global metadata, unit
    ts=timeseries(data)
    df = ts.to_dataframe()
    df_diff = pd.DataFrame(index=df.index)    # Create running difference dataframe
    df_diff['xrsa'] = df['xrsa'] - df['xrsa'].shift(freq=f"{diff_time}s")  # xrsb rd
    df_diff['xrsb'] = df['xrsb'] - df['xrsb'].shift(freq=f"{diff_time}s")  # xrsb rd
    # Keep other columns in new df
    for col in df.columns: # due to df_diff, anything in df that was required was being dropped
        if col not in ['xrsa', 'xrsb']:
            df_diff[col] = df[col]
    # Set negative or zero differences to small positive value for temp calculation
    df_diff.loc[df_diff['xrsa'] <= 0, 'xrsa'] = 1e-10
    df_diff.loc[df_diff['xrsb'] <= 0, 'xrsb'] = 1e-10
    ts_diff = XRSTimeSeries(data=df_diff, meta=ts.meta, units=ts.units)# Create new XRSTimeSeries
    ts_diff.goes_number = ts.goes_number
    return ts_diff

def calc_temp_em(ts): #Calculate temperature and emission measure from timeseries
    """Calculate the temperature and emission of the running difference timeseries"""
    goes_temp_emiss = goes_xrs.calculate_temperature_em(ts)
    df_calc = goes_temp_emiss.to_dataframe()
    return df_calc


# ts=timeseries()
# print(ts.meta)
# ts2=running_difference(300)
# print(ts2.meta)
# print(calc_temp_em())