import sys
import os
from sunpy.net import Fido, attrs as a
import sunpy.timeseries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

def FAI_flagging(ts, temp_em_df, em_increment, temp_range, diff_time, event_gap):
    """
    FAI flagging.
    ts: original (raw) timeseries
    temp_em_df: T and EM calculated from running differences
    """
    diff_time = float(diff_time)
    df = temp_em_df.copy()
    df_flux = ts.to_dataframe()
    df = df.merge(
        df_flux[["xrsa", "xrsb"]],
        how="inner",           # use 'inner' to keep only overlapping times
        left_index=True,
        right_index=True
    )

    # DIAGNOSTIC: Check index alignment
    print(f"\nDIAGNOSTIC - Index alignment:")
    print(f"  df (temp_em) index length: {len(df.index)}")
    print(f"  df_flux (ts) index length: {len(df_flux.index)}")
    print(f"  df index range: {df.index[0]} to {df.index[-1]}")
    print(f"  df_flux index range: {df_flux.index[0]} to {df_flux.index[-1]}")
    print(f"  Indices equal: {df.index.equals(df_flux.index)}")
    
    if len(df_flux.index) < 2:
        raise RuntimeError("Not enough time points to compute time delta (need at least 2 samples).")

    # Compute time step
    dt_seconds = (df_flux.index[1] - df_flux.index[0]).total_seconds()
    if dt_seconds <= 0 or np.isnan(dt_seconds):
        dt_seconds = 1.0
    
    shift_rows = int(round(diff_time / dt_seconds))
    if shift_rows < 1:
        shift_rows = 1

    print(f"  diff_time: {diff_time}, dt_seconds: {dt_seconds}, shift_rows: {shift_rows}")

    flux_raw = df_flux["xrsb"].astype(float)
    flux_raw = flux_raw.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

    flux = flux_raw.values.astype(float)
    time = (flux_raw.index - flux_raw.index[0]).total_seconds().astype(float)

    # Smooth flux slightly to suppress noise
    flux_smooth = pd.Series(flux).rolling(window=3, center=True, min_periods=1).mean().values

    # Compute simple finite-difference slope (in physical units, W/m^2 per second)
    dFdt = np.gradient(flux_smooth, time)
    df["slope"] = pd.Series(dFdt, index=flux_raw.index)
    df["slope"] = df["slope"].rolling(window=5, center=True, min_periods=1).mean()
    # Convert to standard units
    df['EM_49'] = df['emission_measure'] / 1e49
    df['T_MK'] = df['temperature']

    # Only set truly tiny slopes to zero
    df.loc[np.abs(df["slope"]) < 1e-10, "slope"] = 0.0

    
    print(f"  Slope range: {df['slope'].min():.2e} to {df['slope'].max():.2e}")
    
    # DIAGNOSTIC: Check a specific flagged time
    # flagged_check = df[(df['T_MK'] >= temp_range[0]) & 
    #                    (df['T_MK'] <= temp_range[1]) & 
    #                    (df['EM_49'] > em_increment)]
    
    # if len(flagged_check) > 0:
    #     print(f"\n  Sample of points meeting EM & T criteria:")
    #     sample_idx = flagged_check.head(5).index
    #     for idx in sample_idx:
    #         slope_val = df.loc[idx, 'slope']
    #         em_val = df.loc[idx, 'EM_49']
    #         t_val = df.loc[idx, 'T_MK']
    #         print(f"    {idx}: slope={slope_val:.2e}, EM={em_val:.4f}, T={t_val:.2f}")
    
    # Flag when all criteria met
    df["slope"] = df["slope"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['FAI_flag'] = (
        (df['T_MK'] >= temp_range[0]) &
        (df['T_MK'] <= temp_range[1]) &
        (df['EM_49'] > em_increment) &
        (df["slope"] > 0)
    )
    
    flagged = df[df['FAI_flag']].copy()
    flagged_times = []
    
    if len(flagged) > 0:
        prev_time = None
        for idx in flagged.index:
            if prev_time is None:
                flagged_times.append(idx)
                prev_time = idx
            else:
                time_gap = (idx - prev_time).total_seconds() / 60.0
                if time_gap > event_gap:
                    flagged_times.append(idx)
                prev_time = idx

    flagged_times = pd.DatetimeIndex(flagged_times)

    if len(flagged_times) > 0:
        print("\n=== FLAG DETAILS ===")
        for i, t in enumerate(flagged_times, 1):
            if t in df.index:
                slope_val = df.loc[t, "slope"]
                em_val = df.loc[t, "EM_49"]
                temp_val = df.loc[t, "T_MK"]
                flux_val = df.loc[t, "xrsb"] if "xrsb" in df.columns else np.nan
                print(f"Flag {i}: {t} | slope={slope_val:.3e}, EM_49={em_val:.4f}, T={temp_val:.2f} MK, Flux={flux_val:.3e}")
        print("======================\n")
    else:
        print("No flagged times found.\n")
    # Diagnostics
    valid_em = df[df['EM_49'].notna() & np.isfinite(df['EM_49'])]['EM_49']
    valid_temp = df[df['T_MK'].notna() & np.isfinite(df['T_MK'])]['T_MK']
    valid_slope = df[df['slope'].notna() & np.isfinite(df['slope'])]['slope']
    
    print(f"\n  Valid EM points: {len(valid_em)}")
    print(f"  Positive slope points: {(valid_slope > 0).sum()}")
    print(f"  Negative/zero slope points: {(valid_slope <= 0).sum()}")
    print(f"  EM range: {valid_em.min():.6f} - {valid_em.max():.6f} EM_49")
    print(f"  EM points > threshold ({em_increment}): {(valid_em > em_increment).sum()}")
    print(f"  Temperature range: {valid_temp.min():.2f} - {valid_temp.max():.2f} MK")
    print(f"  Temp points in range [{temp_range[0]}, {temp_range[1]}]: {((valid_temp >= temp_range[0]) & (valid_temp <= temp_range[1])).sum()}")
    print(f"  Points meeting ALL criteria: {df['FAI_flag'].sum()}")
    print(f"  Distinct events (after grouping): {len(flagged_times)}")
    flagged = df[df["FAI_flag"]]
    print("\n=== FAI Flagged Events ===")

    return df, flagged_times

def flare_class(ax):
    """
    Add horizontal lines at 10^-x W/m^2 with GOES flare class labels.
    """
    # GOES flare classes in W/m^2
    flare_classes = {
        1e-8: 'A',
        1e-7: 'B',
        1e-6: 'C',
        1e-5: 'M',
        1e-4: 'X'
    }

    y_min, y_max = ax.get_ylim()
    # Only include classes that are within the plotted range
    for flux, label in flare_classes.items():
        if y_min <= flux <= y_max:
            # Horizontal line
            ax.axhline(y=flux, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
            # Label on the right
            ax.text(ax.get_xlim()[1], flux, f' {label}', va='center', ha='left', fontsize=9, color='black')

def anticipation_plot(ts, flagged_times,saved_graph_path, start,extension,
                      integration_time, diff_time,
                      em_increment, temp_range, event_gap):

    """
    Plot original timeseries of xrs data with times flagged for being potential precursors to flare beginning
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    time_support()# Enable time support for better date formatting
    df = ts.to_dataframe()    # Get data for plotting
    param_text = (
        f"Integration Time = {integration_time}s\n"
        f"Difference Time = {diff_time}s\n"
        f"ΔEM > {em_increment}×10⁴⁹ cm⁻³\n"
        f"T ∈ [{temp_range[0]}, {temp_range[1]}] MK\n"
        f"Event gap = {event_gap} min")
    # Plot both channels
    ax.plot(df.index, df['xrsb'], label='GOES 1-8 Å', color='limegreen', linewidth=1.0)
    ax.plot(df.index, df['xrsa'], label='GOES 0.5-4 Å', color='blue', linewidth=1.0)
    ax.plot([],[],label=param_text,color="white")
    ax.legend()
    flare_class(ax)
    # Add vertical lines for FAI flags
    y_min, y_max = ax.get_ylim()
    for flag_time in flagged_times:
        ax.axvline(x=flag_time, color='red', linestyle='-', linewidth=1, 
                  alpha=0.8, zorder=10)
    # Set y-axis to log scale
    ax.set_yscale('log')
    # plt.title(f"Anticipation Plot w/ Params {extension}")
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

def diagnostic_plot(temp_em, saved_graph_path,start,start_extension):
    """
    Plot the [EM,T], showing how temperature changes with emission
    """

    temp=temp_em["temperature"]
    em=temp_em["emission_measure"]/1e49 # scale down
    
    # valid = np.isfinite(temp) & np.isfinite(em) & (temp > 0) & (em > 0)
    # temp = temp[valid]
    # em = em[valid]

    # fig, ax = plt.subplots(3,1,figsize=(10,6))
    fig = plt.figure(figsize=(10,6),constrained_layout=True)
    ax = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                          gridspec_kw={'width_ratios':[2, 1]})
    fig.suptitle("Diagnostic Plots: Emission Measure vs Temperature vs Time", fontsize=16)
    ax["Left"].scatter(em,temp)
    ax["Left"].set_xscale('log')
    ax["Left"].set_xlabel("Emission Measure$^{49}$, cm$^{-3}$", fontsize=12)
    ax["Left"].set_ylabel("Temperature, MK", fontsize=12)
    ax["Left"].xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax["Left"].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
    ax["Left"].xaxis.set_minor_formatter(ticker.NullFormatter())
    ax["Left"].xaxis.set_major_formatter(ticker.LogFormatterExponent(base=10))
    
    df =temp_em
    ax["TopRight"].scatter(df.index,temp)
    ax["TopRight"].set_ylabel("T (MK)", fontsize=12)
    ax["BottomRight"].set_xlabel("Time", fontsize=12)
    ax["BottomRight"].set_ylabel("EM $10^{49}$", fontsize=12)
    plt.setp(ax["TopRight"].get_xticklabels(), visible=False)

    ax["BottomRight"].scatter(df.index,em)
    ax["BottomRight"].xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Tick every hour
    ax["BottomRight"].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    filename_EM = f"Diagnostic_{start_extension}.png"
    plt.savefig(f"{saved_graph_path}/{filename_EM}", dpi=150, bbox_inches='tight')
    print(f"Plot saved to {saved_graph_path}/{filename_EM}")

    plt.close()
    return fig
