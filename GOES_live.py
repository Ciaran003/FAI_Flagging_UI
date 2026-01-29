import pandas as pd
import numpy as np
# import sunpy.timeseries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.visualization import time_support
import sunpy
from sunpy.time import TimeRange
from sunpy import timeseries as ts
from sunkit_instruments import goes_xrs #type:ignore
from sunpy.timeseries.sources.goes import XRSTimeSeries
from astropy import units as u
import matplotlib.dates as mdates
import os

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
        values="observed_flux")
    df = df.rename(columns={
        "0.05-0.4nm": "xrsa",
        "0.1-0.8nm": "xrsb"})
    df['satellite'] = data['satellite'].iloc[0]
    # print(df)
    return df

def timeseries(data):
    """
    create timeseries of data\n
    recreates metadata & units required for data to be usable in further functions
    """
    global metadata, unit
    df1 = process_data(data)
    sat = int(df1["satellite"].iloc[0])
    df = pd.DataFrame()
    df["xrsa"] = df1["xrsa"]
    df["xrsb"] = df1["xrsb"]
    
    meta = {
        "TELESCOP": f"GOES {sat}",  # This is the format it expects: "GOES 18"
        "instrument": "XRS",
        "observatory": f"GOES-{sat}",
        "goes_number": sat,
        "wave": {"xrsa": (0.5, 4), "xrsb": (1, 8)},
    }
    units = {"xrsa": u.W / u.m**2, "xrsb": u.W / u.m**2}
    metadata = meta
    unit = units
    
    ts = XRSTimeSeries(df, meta=meta, units=units)
    
    return ts

def running_difference(data, diff_time):
    """
    Create time series of running differences. represents flux(t) - flux(t - diff_time)
    Recreate in a new XRStimeseries, with original metadata
    """
    global metadata, unit
    ts = timeseries(data)
    df = ts.to_dataframe()
    df_diff = pd.DataFrame(index=df.index)    # Create running difference dataframe
    df_diff['xrsa'] = df['xrsa'] - df['xrsa'].shift(freq=f"{diff_time}s")  # xrsa rd
    df_diff['xrsb'] = df['xrsb'] - df['xrsb'].shift(freq=f"{diff_time}s")  # xrsb rd
    
    # Keep other columns in new df
    for col in df.columns:  # due to df_diff, anything in df that was required was being dropped
        if col not in ['xrsa', 'xrsb']:
            df_diff[col] = df[col]
    
    # Set negative or zero differences to small positive value for temp calculation
    df_diff.loc[df_diff['xrsa'] <= 0, 'xrsa'] = 1e-10
    df_diff.loc[df_diff['xrsb'] <= 0, 'xrsb'] = 1e-10
    
    # Get satellite number from original timeseries
    sat = ts.meta.metadata[0][2].get('goes_number', 18)  # fallback to 18 if not found
    
    new_meta = {
        "TELESCOP": f"GOES {sat}",  # This is the format XRSTimeSeries looks for
        "instrument": "XRS",
        "observatory": f"GOES-{sat}",
        "goes_number": sat,
        "wave": {"xrsa": (0.5, 4), "xrsb": (1, 8)},
    }
    
    # Create new XRSTimeSeries with proper metadata
    ts_diff = XRSTimeSeries(data=df_diff, meta=new_meta, units=ts.units)
    
    return ts_diff

def calc_temp_em(ts): #Calculate temperature and emission measure from timeseries
    """Calculate the temperature and emission of the running difference timeseries"""
    goes_temp_emiss = goes_xrs.calculate_temperature_em(ts)
    df_calc = goes_temp_emiss.to_dataframe()
    return df_calc

def FAI_flagging(ts, temp_em_df, em_increment, temp_range, diff_time, event_gap,out):
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
    # print(f"\nDIAGNOSTIC - Index alignment:")
    # print(f"  df (temp_em) index length: {len(df.index)}")
    # print(f"  df_flux (ts) index length: {len(df_flux.index)}")
    # print(f"  df index range: {df.index[0]} to {df.index[-1]}")
    # print(f"  df_flux index range: {df_flux.index[0]} to {df_flux.index[-1]}")
    # print(f"  Indices equal: {df.index.equals(df_flux.index)}")
    
    if len(df_flux.index) < 2:
        raise RuntimeError("Not enough time points to compute time delta (need at least 2 samples).")

    # Compute time step
    dt_seconds = (df_flux.index[1] - df_flux.index[0]).total_seconds()
    if dt_seconds <= 0 or np.isnan(dt_seconds):
        dt_seconds = 1.0
    
    shift_rows = int(round(diff_time / dt_seconds))
    if shift_rows < 1:
        shift_rows = 1

    # print(f"  diff_time: {diff_time}, dt_seconds: {dt_seconds}, shift_rows: {shift_rows}")

    flux_raw = df_flux["xrsb"].astype(float)
    # flux_raw = flux_raw.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    flux_raw = flux_raw.replace([np.inf, -np.inf], np.nan).ffill()

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

    
    # print(f"  Slope range: {df['slope'].min():.2e} to {df['slope'].max():.2e}")
    
    # Flag when all criteria met
    df["slope"] = df["slope"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['FAI_flag'] = (
        (df['xrsb'] >= 1e-6) &
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
        # print("\n=== FLAG DETAILS ===")
        for i, t in enumerate(flagged_times, 1):
            if t in df.index:
                slope_val = df.loc[t, "slope"]
                em_val = df.loc[t, "EM_49"]
                temp_val = df.loc[t, "T_MK"]
                flux_val = df.loc[t, "xrsb"] if "xrsb" in df.columns else np.nan
                # print(f"Flag {i}: {t} | slope={slope_val:.3e}, EM_49={em_val:.4f}, T={temp_val:.2f} MK, Flux={flux_val:.3e}")
        # print("======================\n")
    else:
        print("No flagged times found.\n")
    # Diagnostics
    valid_em = df[df['EM_49'].notna() & np.isfinite(df['EM_49'])]['EM_49']
    valid_temp = df[df['T_MK'].notna() & np.isfinite(df['T_MK'])]['T_MK']
    valid_slope = df[df['slope'].notna() & np.isfinite(df['slope'])]['slope']
    
    # print(f"\n  Valid EM points: {len(valid_em)}")
    # print(f"  Positive slope points: {(valid_slope > 0).sum()}")
    # print(f"  Negative/zero slope points: {(valid_slope <= 0).sum()}")
    # print(f"  EM range: {valid_em.min():.6f} - {valid_em.max():.6f} EM_49")
    # print(f"  EM points > threshold ({em_increment}): {(valid_em > em_increment).sum()}")
    # print(f"  Temperature range: {valid_temp.min():.2f} - {valid_temp.max():.2f} MK")
    # print(f"  Temp points in range [{temp_range[0]}, {temp_range[1]}]: {((valid_temp >= temp_range[0]) & (valid_temp <= temp_range[1])).sum()}")
    # print(f"  Points meeting ALL criteria: {df['FAI_flag'].sum()}")
    # print(f"  Distinct events (after grouping): {len(flagged_times)}")
    flagged = df[df["FAI_flag"]]
    # print("\n=== FAI Flagged Events ===")
    # out_string=
    # with open(out,"a")as f:
    #     for t in flagged_times:
    #         f.write(f"{t.isoformat()}\n")
    times=flagged.index
    # print(times)
    # Convert DatetimeIndex to DataFrame
    new_df = pd.DataFrame(flagged_times, columns=["flagged_time"])

    if os.path.exists(out):
        old_df = pd.read_csv(out, parse_dates=["flagged_time"])
        new_df = new_df[~new_df["flagged_time"].isin(old_df["flagged_time"])]

    # Only write if there is something new
    if not new_df.empty:
        new_df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)



    # out=pd.DataFrame(flagged_times, columns=['flagged_time']).to_csv('flags.csv', mode='a', header=False, index=False)
    # print(out)
    # flagged.to_csv('flags.csv', mode='a', header=False, index=False)
    return df, flagged_times,out

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
        1e-4: 'X',
        1e-3: "X10"
    }

    y_min, y_max = ax.get_ylim()
    # Only include classes that are within the plotted range
    for flux, label in flare_classes.items():
        if y_min <= flux <= y_max:
            # Horizontal line
            ax.axhline(y=flux, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
            # Label on the right
            ax.text(ax.get_xlim()[1], flux, f' {label}', va='center', ha='left', fontsize=9, color='black')

# def anticipation_plot(ts, flagged_times,saved_graph_path, start,extension,
#                       integration_time, diff_time,
#                       em_increment, temp_range, event_gap):



def anticipation_plot(ts, flagged_times,diff_time,
                      em_increment, temp_range, event_gap,sgp):
    """
    Plot timeseries of xrs data with times flagged for being potential precursors to flare beginning
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    time_support()# Enable time support for better date formatting
    df = ts.to_dataframe()    # Get data for plotting
    param_text = (
        # f"Integration Time = {integration_time}s\n"
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
    ax.set_xlabel("Time")
    # ax.set_xlabel('Start Time: ' + start.split()[0] + ' ' + start.split()[1] + ' UT', fontsize=11)
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
    # print(f"\n{'='*60}")
    # print(f"FAI FLAGS DETECTED: {len(flagged_times)}")
    # print(f"{'='*60}")
    # for i, ft in enumerate(flagged_times, 1):
    #     print(f"  Flag {i}: {ft}")
    # print(f"{'='*60}\n")
    
    
    # if not os.path.exists(saved_graph_path):
    #     os.makedirs(saved_graph_path)
    # filename = f"anticipating_flare_{extension}.png"
    # plt.savefig(f"{saved_graph_path}/{filename}", dpi=150, bbox_inches='tight')
    # print(f"Plot saved to {saved_graph_path}/{filename}")
    # if show_plot == "yes":
    plt.savefig(f"{sgp}/anticipation-plot")
    # plt.show()
    plt.close()
    return fig

# def diagnostic_plot(temp_em, saved_graph_path,start,start_extension,flagged_times):
#     """
#     Plot the [EM,T], showing how temperature changes with emission
#     """

#     temp=temp_em["temperature"]
#     em=temp_em["emission_measure"]/1e49 # scale down
#     time = mdates.date2num(temp_em.index)
    

#     fig = plt.figure(figsize=(10,6),constrained_layout=True)
#     ax = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
#                           gridspec_kw={'width_ratios':[2, 1]})

#     sc = ax["Left"].scatter(em, temp, c=time, cmap='viridis', s=30)
#     ax["Left"].set_xscale('log')
#     ax["Left"].set_xlabel("Emission Measure $10^{49}$, cm$^{-3}$", fontsize=12)
#     ax["Left"].set_ylabel("Temperature, MK", fontsize=12)
#     ax["Left"].xaxis.set_major_locator(ticker.LogLocator(base=10.0))
#     ax["Left"].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
#     ax["Left"].xaxis.set_minor_formatter(ticker.NullFormatter())
#     ax["Left"].xaxis.set_major_formatter(ticker.LogFormatterExponent(base=10))

#     # Add colorbar showing time
#     cbar = fig.colorbar(sc, ax=ax["Left"], orientation='horizontal', pad=0.02)
#     cbar.set_label("Time", fontsize=12)
#     cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


#     df=temp_em
#     ax["TopRight"].scatter(df.index,temp)
#     ax["TopRight"].set_ylabel("T (MK)", fontsize=12)
#     ax["BottomRight"].set_xlabel("Time", fontsize=12)
#     ax["BottomRight"].set_ylabel("EM $10^{49}$", fontsize=12)
#     ax["TopRight"].xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Tick every hour
#     ax["TopRight"].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#     plt.setp(ax["TopRight"].get_xticklabels(), visible=False)
#     for flag_time in flagged_times:
#         ax["TopRight"].axvline(x=flag_time, color='red', linestyle='-', linewidth=1, 
#                   alpha=0.8, zorder=10)
        
#     ax["BottomRight"].scatter(df.index,em)
#     ax["BottomRight"].xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Tick every hour
#     ax["BottomRight"].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#     for flag_time in flagged_times:
#             ax['BottomRight'].axvline(x=flag_time, color='red', linestyle='-', linewidth=1, 
#                       alpha=0.8, zorder=10)

#     filename_EM = f"Diagnostic_{start_extension}.png"
#     plt.savefig(f"{saved_graph_path}/{filename_EM}", dpi=150, bbox_inches='tight')
#     print(f"Plot saved to {saved_graph_path}/{filename_EM}")
#     plt.close()
#     return fig

def EMT_plot(temp_em,sgp):
    """Plot EM vs T colored by time"""
    df=temp_em
    temp = temp_em["temperature"]
    em = temp_em["emission_measure"] / 1e49  # scale down
    time = mdates.date2num(temp_em.index)

    fig, ax = plt.subplots(figsize=(10,6))

    sc = ax.scatter(em, temp, c=time, cmap='viridis', s=30)

    ax.set_xscale('log')
    ax.set_xlabel("Emission Measure $10^{49}$ cm$^{-3}$", fontsize=12)
    ax.set_ylabel("Temperature (MK)", fontsize=12)

    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_major_formatter(ticker.LogFormatterExponent(base=10))
    
    hours_span = (df.index[-1] - df.index[0]).total_seconds() / 3600
    if hours_span <= 6:
        # For short time spans, show HH:MM
        date_format = mdates.DateFormatter('%H:%M')
    else:
        # For longer spans, show MM-DD HH:MM
        date_format = mdates.DateFormatter('%m-%d %H:%M')
    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.02)
    cbar.set_label("Time", fontsize=12)
    cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.savefig(f"{sgp}/emission-temp-plot")
    # plt.show()
    plt.close(fig)

    return fig

def T_t_plot(temp_em, flagged_times,sgp):
    df = temp_em
    temp = df["temperature"]

    fig, ax = plt.subplots(figsize=(10,6))

    ax.scatter(df.index, temp, s=20)

    ax.set_ylabel("Temperature (MK)", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)

    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
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
    # plt.setp(ax.get_xticklabels(), visible=True)

    for flag_time in flagged_times:
        ax.axvline(
            x=flag_time,
            color='red',
            linestyle='-',
            linewidth=1,
            alpha=0.8,
            zorder=10
        )
    plt.savefig(f"{sgp}/temp-time-plot")
    # plt.show()
    plt.close(fig)

    return fig

def EM_t_plot(temp_em, flagged_times,sgp):
    df = temp_em
    em = df["emission_measure"] / 1e49  # scale to 10^49

    fig, ax = plt.subplots(figsize=(10,6))

    ax.scatter(df.index, em, s=20)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Emission Measure ($10^{49}$ cm$^{-3}$)", fontsize=12)

    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%D:%H'))
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

    for flag_time in flagged_times:
        ax.axvline(
            x=flag_time,
            color='red',
            linestyle='-',
            linewidth=1,
            alpha=0.8,
            zorder=10
        )
    plt.savefig(f"{sgp}/emission-time-plot")
    # plt.show()
    plt.close(fig)

    return fig
