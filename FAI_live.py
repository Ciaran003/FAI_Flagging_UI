from functions import GOES_live as goes
import os
import sys
import pandas as pd
goes.timeseries

base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
for file in os.listdir(base_dir):
        if file.startswith("xrays") and file.endswith(".json"):
            data_file = file
data_path=os.path.join(base_dir, data_file)
data=pd.read_json(data_path)

ts_og=goes.timeseries(data)
ts_rd=goes.running_difference(data,300)

print(ts_og.satellite_number)
# print(ts_rd.meta)
# print(goes.calc_temp_em(ts_og))
