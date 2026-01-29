from functions import GOES_live as goes
import os
import sys
import pandas as pd
goes.timeseries
import pdb
from functions import live_inputs as live

base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
sgp=os.path.join(base_dir,"saved_graphs")
flag_path=os.path.join(base_dir,"flagged_times")
for file in os.listdir(flag_path):
    if file.startswith("flags"):
        out=file
      
for file in os.listdir(base_dir):
        if file.startswith("xrays") and file.endswith(".json"):
            data_file = file
data_path=os.path.join(base_dir, data_file)
data=pd.read_json(data_path)

input_file="live_params.txt"
input_path=os.path.join(base_dir,input_file)

# diff_time,EM_inc,temp_range,FAI_duration=live.parameters(input_path)
# data=live.data_fetch()   # fetch data from NOAA


# ts_og=goes.timeseries(data) # timeseries for data
# ts_rd=goes.running_difference(data,diff_time)   # ts with running difference
# tem_og=goes.calc_temp_em(ts_og)     # temperature-emission calcs for ts (used for plots)
# tem_rd=goes.calc_temp_em(ts_rd)     # temperature-emission calcs for ts running difference (used for flagging)

# flagged_df,flagged_times,flag_csv=goes.FAI_flagging(
#         ts_og,
#         tem_rd,
#         EM_inc,
#         temp_range,
#         diff_time,
#         FAI_duration,
#         out)

# anticipation_plot=goes.anticipation_plot(
#         ts_og,
#         flagged_times,
#         str(diff_time),
#         str(EM_inc),
#         temp_range,
#         str(FAI_duration),
#         sgp)

# EMT_plot=goes.EMT_plot(
#             tem_og,
#             sgp)

# T_t_plot=goes.T_t_plot(
#       tem_og,
#       flagged_times,
#       sgp)

# EM_t_plot=goes.EM_t_plot(
#       tem_og,
#       flagged_times,
#       sgp)

def programme_run():
    diff_time,EM_inc,temp_range,FAI_duration=live.parameters(input_path)
    # data=live.data_fetch()   # fetch data from NOAA
    ts_og=goes.timeseries(data) # timeseries for data
    ts_rd=goes.running_difference(data,diff_time)   # ts with running difference
    tem_og=goes.calc_temp_em(ts_og)     # temperature-emission calcs for ts (used for plots)
    tem_rd=goes.calc_temp_em(ts_rd)     # temperature-emission calcs for ts running difference (used for flagging)

    flagged_df,flagged_times,flag_csv=goes.FAI_flagging(
            ts_og,
            tem_rd,
            EM_inc,
            temp_range,
            diff_time,
            FAI_duration,
            out)

    anticipation_plot=goes.anticipation_plot(
            ts_og,
            flagged_times,
            str(diff_time),
            str(EM_inc),
            temp_range,
            str(FAI_duration),
            sgp)

    EMT_plot=goes.EMT_plot(
                tem_og,
                sgp)

    T_t_plot=goes.T_t_plot(
          tem_og,
          flagged_times,
          sgp)

    EM_t_plot=goes.EM_t_plot(
          tem_og,
          flagged_times,
          sgp)

import sched
import time

def sched_programme(sc):
    sc.enter(60, 1, sched_programme, (sc,))
    programme_run()

my_scheduler = sched.scheduler(time.time, time.sleep)
my_scheduler.enter(60, 1, sched_programme, (my_scheduler,))
my_scheduler.run()
