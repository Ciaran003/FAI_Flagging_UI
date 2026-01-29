def parameters(input):
    """Reads the input txt file and parses for input parameters"""
    with open(input) as file:
        lines = [line for line in file]
    for line in lines:
        if line.startswith("Difference time = "):
            diff_time= int(line[len("Difference time = "):-1]) # convets to integer from string in text file
        elif line.startswith("EM increment = "):
            EM_inc=float(line[len("EM increment = "):-1])   #converts to float, since number is usually <1
        elif line.startswith("Temp range = "):
            temp_range_str=(line[len("Temp range = "):-1])  # kept as string to split at the comma
            temp_range=[int(i) for i in temp_range_str.split(",")]  # each item is converted to integer and split apart by the comma, creating range
        elif line.startswith("FAI duration (mins) = "):
            FAI_duration=int(line[len("FAI duration (mins) = "):-1])    # converted to string to separate different events
    return diff_time,EM_inc,temp_range,FAI_duration

import requests #type:ignore
from bs4 import BeautifulSoup
import pandas as pd

def data_fetch():
    """Fetches data from NOAA and converts to a dataframe"""
    url="https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
    response = requests.get(url) # pings the url
    response.raise_for_status() #checks for errors
    data=response.json()    #returns the json file requested
    data_df=pd.DataFrame(data)  # convert to pandas df
    return data_df
