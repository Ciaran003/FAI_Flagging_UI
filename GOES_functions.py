import sys
import os
from sunpy.net import Fido, attrs as a
import sunpy.timeseries
import matplotlib.pyplot as plt
from astropy.visualization import time_support
from sunpy.time import TimeRange
import numpy as np
import pandas as pd
from sunkit_instruments import goes_xrs #type:ignore
from sunpy.timeseries.sources.goes import XRSTimeSeries
import matplotlib.dates as mdates

def timeseries_data(start_date,end_date,integration_time):
    """
    Get timeseries data from start to end time of input,
    Resample with integration time to reduce noise
    Convert back to timeseries, while keeping metadata
    """
    data = Fido.search(a.Time(start_date, end_date), a.Instrument.xrs) # fido gets the data from the noaa
    downloaded_data = Fido.fetch(data)  # download file
    timeseries = sunpy.timeseries.TimeSeries(downloaded_data)   # turn to timeseries
    if isinstance(timeseries, list):
        ts = timeseries[0].concatenate(timeseries[1:])
    else:
        ts = timeseries
    timerange = TimeRange(start_date, end_date)     # set time range
    timeseries_trunc = ts.truncate(timerange)   # truncate to the times in the input file
    # print(timeseries_trunc) # diagnostic to view the gap between the time values
    df = timeseries_trunc.to_dataframe()
    df_resampled = df.resample(f"{int(integration_time)}s").mean() # takes the mean value for every second
    df_resampled_smooth=df.resample("60s").mean()
    df_1s=df.resample("1s").mean()
    units_dict = {col: timeseries_trunc.units[col] for col in df_resampled.columns if col in timeseries_trunc.units}
    ts_integ_time=XRSTimeSeries(data=df_resampled, meta=timeseries_trunc.meta, units=units_dict)
    smooth_plot=XRSTimeSeries(data=df_resampled_smooth, meta=timeseries_trunc.meta, units=units_dict) # only used in plots, not in calculations
    ts_1s=XRSTimeSeries(data=df_1s, meta=timeseries_trunc.meta, units=units_dict)
    # print(ts_integ_time)  # diagnostic to view the gap between the time values after integration time
    print("Data Download Complete")
    return ts_integ_time,smooth_plot,ts_1s

def running_difference(ts, diff_time):
    """
    Create time series of running differences. represents flux(t) - flux(t - diff_time)
    Recreate in a new XRStimeseries, with original metadata
    """
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
    
    # Create units dictionary, REQUIRED TO CREATE XRSTS
    units_dict = {col: ts.units[col] for col in df_diff.columns if col in ts.units}
    ts_diff = XRSTimeSeries(data=df_diff, meta=ts.meta, units=units_dict)# Create new XRSTimeSeries
    print("Running Difference Done")
    return ts_diff

def calc_temp_em(ts): #Calculate temperature and emission measure from timeseries
    """Calculate the temperature and emission of the running difference timeseries"""
    goes_temp_emiss = goes_xrs.calculate_temperature_em(ts)
    df_calc = goes_temp_emiss.to_dataframe()
    print("Temp & Em completed")
    return df_calc

def FAI_flagging(temp_em_df, em_increment, temp_range, diff_time,event_gap):
    """
    FAI flagging.
    forms running differences of the raw flux values,
    then calculates T and EM from those differences. The resulting EM represents
    the emission measure of the excess flux above background over the diff_time period.
    flag when this excess EM > threshold AND T is in range.
    """
    df = temp_em_df.copy()
    df['EM_49'] = df['emission_measure'] / 1e49# Convert to EM_49 
    df['T_MK'] = df['temperature']
    
    # The EM here already represents the change/increase since we calculated it from flux differences
    df['FAI_flag'] = (
        (df['T_MK'] >= temp_range[0]) &  # T_a < T
        (df['T_MK'] <= temp_range[1]) &  # T < T_b
        (df['EM_49'] > em_increment)   # EM_49 > Y (this is the dEM threshold)
    )
    flagged = df[df['FAI_flag']].copy()
    flagged_times = []
    
    if len(flagged) > event_gap:
        prev_time = None
        for idx in flagged.index:
            if prev_time is None:
                # First flag ever
                flagged_times.append(idx)
                prev_time = idx
            else:
                # Check if this is a new event (gap > 3 minutes from last flag)
                time_gap = (idx - prev_time).total_seconds() / 60.0
                if time_gap > event_gap:
                    flagged_times.append(idx)
                prev_time = idx
    
    flagged_times = pd.DatetimeIndex(flagged_times)

    # Print diagnostic info
    valid_em = df[df['EM_49'].notna() & np.isfinite(df['EM_49'])]['EM_49']
    valid_temp = df[df['T_MK'].notna() & np.isfinite(df['T_MK'])]['T_MK']
    print(f"\n  Valid EM points: {len(valid_em)}")
    print(f"  EM range: {valid_em.min():.6f} - {valid_em.max():.6f} EM_49")
    print(f"  EM points > threshold ({em_increment}): {(valid_em > em_increment).sum()}")
    print(f"  Temperature range: {valid_temp.min():.2f} - {valid_temp.max():.2f} MK")
    print(f"  Temp points in range [{temp_range[0]}, {temp_range[1]}]: {((valid_temp >= temp_range[0]) & (valid_temp <= temp_range[1])).sum()}")
    print(f"  Points meeting ALL criteria: {df['FAI_flag'].sum()}")
    print(f"  Distinct events (after grouping): {len(flagged_times)}")
    
    return df, flagged_times

def anticipation_plot(ts, flagged_times,saved_graph_path, start,extension):

    """
    Plot original timeseries of xrs data with times flagged for being potential precursors to flare beginning
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    time_support()# Enable time support for better date formatting
    df = ts.to_dataframe()    # Get data for plotting
    
    # Plot both channels
    ax.plot(df.index, df['xrsb'], label='GOES 1-8 Å', color='black', linewidth=1.0)
    ax.plot(df.index, df['xrsa'], label='GOES 0.5-4 Å', color='black', linewidth=1.0)
    # Add vertical lines for FAI flags
    y_min, y_max = ax.get_ylim()
    for flag_time in flagged_times:
        ax.axvline(x=flag_time, color='red', linestyle='-', linewidth=1, 
                  alpha=0.8, zorder=10)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    plt.title(f"Anticipation Plot w/ Params {extension}")
    ax.set_ylabel('X-ray Flux, Watts/m$^{2}$', fontsize=11)
    ax.set_xlabel('Start Time: ' + start.split()[0] + ' ' + start.split()[1] + ' UT', fontsize=11)
    ax.grid(False)
    # ax.legend(loc='upper left', fontsize=10)
    hours_span = (df.index[-1] - df.index[0]).total_seconds() / 3600
    if hours_span <= 6:
        # For short time spans, show HH:MM
        date_format = mdates.DateFormatter('%H:%M')
    else:
        # For longer spans, show MM-DD HH:MM
        date_format = mdates.DateFormatter('%m-%d %H:%M')
    
    ax.xaxis.set_major_formatter(date_format)
    
    # Optionally set locator for better tick spacing
    if hours_span <= 6:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Tick every hour
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))  # Minor ticks every 15 min
    
    plt.gcf().autofmt_xdate(rotation=0)# Format x-axis to show time
    plt.tight_layout()
    
    # Print flag information
    print(f"\n{'='*60}")
    print(f"FAI FLAGS DETECTED: {len(flagged_times)}")
    print(f"{'='*60}")
    for i, ft in enumerate(flagged_times, 1):
        print(f"  Flag {i}: {ft}")
    print(f"{'='*60}\n")
    
    
    if not os.path.exists(saved_graph_path):
        os.makedirs(saved_graph_path)
    filename = f"anticipating_flare_{extension}.png"
    plt.savefig(f"{saved_graph_path}/{filename}", dpi=150, bbox_inches='tight')
    print(f"Plot saved to {saved_graph_path}/{filename}")
    # if show_plot == "yes":
    #     plt.show()
    plt.close()
    return fig

def em_temp_plot(temp_em, saved_graph_path,start,start_extension):
    """
    Plot the [EM,T], showing how temperature changes with emission
    """

    temp=temp_em["temperature"]
    em=temp_em["emission_measure"]/1e49 # scale down
    
    # valid = np.isfinite(temp) & np.isfinite(em) & (temp > 0) & (em > 0)
    # temp = temp[valid]
    # em = em[valid]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(em,temp)
    ax.set_xscale('log')
    # ax.set_xlim(0.01,10)
    # ax.set_ylim(10,)
    ax.set_xlabel("Emission Measure$^{49}$, cm$^{-3}$", fontsize=12)
    ax.set_ylabel("Temperature, MK", fontsize=12)
    ax.set_title(f"[EM,T] beginning at {start}")
    # ax.set_xlim(0.01,10)
    # ax.set_xticks(np.logspace)


    filename_EM = f"[EM,T]_{start_extension}.png"
    plt.savefig(f"{saved_graph_path}/{filename_EM}", dpi=150, bbox_inches='tight')
    print(f"Plot saved to {saved_graph_path}/{filename_EM}")
    # if show_plot == "yes":
    #     plt.show()
    plt.close()
    return fig