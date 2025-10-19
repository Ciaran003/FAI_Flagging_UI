import os
import sys
from tkinter import *
import tkinter as tk
from tkinter import ttk
from functions import GOES_functions as gf#type: ignore
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from datetime import datetime, timedelta


base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
sgp = os.path.join(base_dir, "saved_graphs")
inputs={}

def get_current_time():
    """Returns the current time as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_time_minus_lookback(lookback_duration):
    """Returns the time 'lookback_duration' ago from the current time."""
    lookback_map = {
        "1 Hour": timedelta(hours=1),
        "6 Hours": timedelta(hours=6),
        "1 Day": timedelta(days=1),
        "1 Week": timedelta(weeks=1)
        }
    return (datetime.now() - lookback_map.get(lookback_duration, timedelta(hours=1))).strftime("%Y-%m-%d %H:%M:%S")

def submit():
    param_text = "Current Parameters:\n"
    # Handle Custom Data Input
    if date_var.get() == "Custom Data":
        inputs["Start"] = start_date_entry.get()
        inputs["End"] = end_date_entry.get()
    else:
        # Handle Live Data
        inputs["Start"] = get_time_minus_lookback(lookback_dropdown.get())
        inputs["End"] = get_current_time()

    # Clean up the date output to just show the date (YYYY-MM-DD)
    start_date_clean = inputs.get('Start', '(').split()[0]  # Extract just the date (YYYY-MM-DD)
    end_date_clean = inputs.get('End', '(').split()[0]      # Extract just the date (YYYY-MM-DD)
    
    if date_var.get() == "Custom Data":
        param_text += f"Start Date: {inputs['Start']}\n"
        param_text += f"End Date: {inputs['End']}\n"
    else:
        param_text += f"Start Date: {inputs['Start']}\n"
        param_text += f"End Date: {inputs['End']}\n"
    
    # Now display the other parameters from the parameter_entries
    for key, entry in parameter_entries.items():
        value = entry.get()
        display_key = key.split('(')[0].strip()
        param_text += f"{display_key}: {value}\n"

    params_display_label.config(text=param_text)

    print("Submitted Parameters:")
    print(f"Start Date: {inputs['Start']}")
    print(f"End Date: {inputs['End']}")
    for key, entry in parameter_entries.items():
        print(f"{key}: {entry.get()}")
    inputs["Integration Time"] = int(parameter_entries["Integration Time (s)"].get())
    inputs["Difference Time"] = parameter_entries["Difference Time (s)"].get()
    inputs["EM Increment"] = float(parameter_entries["EM Increment ⁴⁹ cm⁻³"].get())
    temp_range_str = parameter_entries["Temperature Range (MK)"].get()
    inputs["Temperature Range"] = [float(i) for i in temp_range_str.split(",")]
    inputs["FAI Duration"] = int(parameter_entries["FAI Duration (mins)"].get())
root=tk.Tk()
root.title("FAI Flagging")
root.minsize(1000,600)
# params frame
params_frame=tk.Frame(root,width=200,height=400)
params_frame.pack(padx=5,pady=5,side=tk.LEFT,fill=tk.Y)
tk.Label(
    params_frame,
    text="Parameters"
).pack(padx=5,pady=5)
tk.Label(params_frame).pack(padx=5, pady=5)
params_display_label = tk.Label(params_frame, text="Parameters:\n(none)", justify="left")
params_display_label.pack(padx=5, pady=5)

# Input parameters and settings
notebook=ttk.Notebook(params_frame)
notebook.pack(expand=True,fill="both")

def toggle_timeframe_fields(show_custom):
    if show_custom:
        start_date_label.pack(anchor="w", padx=10, pady=(10, 0))
        start_date_entry.pack(anchor="w", padx=10, pady=(10, 0))
        end_date_label.pack(anchor="w", padx=10, pady=(10, 0))
        end_date_entry.pack(anchor="w", padx=10, pady=(10, 0))
        lookback_label.pack_forget()
        lookback_dropdown.pack_forget()
    else:
        start_date_label.pack_forget()
        start_date_entry.pack_forget()
        end_date_label.pack_forget()
        end_date_entry.pack_forget()
        lookback_label.pack(anchor="w", padx=10, pady=(10, 0))
        lookback_dropdown.pack(anchor="w", padx=10, pady=(10, 0))

date_tab=tk.Frame(notebook)
date_var=tk.StringVar(value="None")
date_entries = {}  # Dictionary to hold text inputs

options=[
    "Start (YYYY-MM-DD HR:MIN:SEC)",
    "End (YYYY-MM-DD HR:MIN:SEC)",
]

custom_radio = tk.Radiobutton(date_tab, text="Custom Data", variable=date_var, value="Custom Data", command=lambda: toggle_timeframe_fields(True))
custom_radio.pack(anchor="w", padx=10, pady=(10, 0))

live_radio = tk.Radiobutton(date_tab, text="Live Data", variable=date_var, value="Live Data", command=lambda: toggle_timeframe_fields(False))
live_radio.pack(anchor="w", padx=10, pady=(10, 0))

start_date_label = tk.Label(date_tab, text="Start Date:")
start_date_label.pack(anchor="w", padx=10, pady=(10, 0))
start_date_entry = tk.Entry(date_tab)
start_date_entry.pack(anchor="w", padx=10, pady=(10, 0))

end_date_label = tk.Label(date_tab, text="End Date:")
end_date_label.pack(anchor="w", padx=10, pady=(10, 0))
end_date_entry = tk.Entry(date_tab)
end_date_entry.pack(anchor="w", padx=10, pady=(10, 0))

lookback_label = tk.Label(date_tab, text="Lookback Duration:")
lookback_label.pack(anchor="w", padx=10, pady=(10, 0))

lookback_options = ["1 Hour", "6 Hours", "1 Day", "1 Week"]
lookback_dropdown = ttk.Combobox(date_tab, values=lookback_options)
lookback_dropdown.pack(anchor="w", padx=10, pady=(10, 0))

date_entries = {
    "Start (YYYY-MM-DD HR:MIN:SEC)": start_date_entry,
    "End (YYYY-MM-DD HR:MIN:SEC)": end_date_entry
}

# parameter tab
params_tab=tk.Frame(notebook)
params_var=tk.StringVar(value="None")
parameter_entries = {}  # Dictionary to hold text inputs

parameters = [
    "Integration Time (s)",
    "Difference Time (s)",
    f"EM Increment ⁴⁹ cm⁻³",
    "Temperature Range (MK)",
    "FAI Duration (mins)"
]

for param in parameters:
    tk.Label(
        params_tab,
        text=param,
    ).pack(anchor="w",padx=10, pady=(10, 0))
    entry=tk.Entry(params_tab)
    entry.pack(anchor="w",padx=10,pady=(10,0),fill="x")
    parameter_entries[param]=entry

notebook.add(date_tab,text="Data Times")
notebook.add(params_tab,text="Parameters")

def process_data():
    start_extension=inputs["Start"].replace(' ', '_').replace(':', '-')
    integ_time_extenstion=str(inputs["Integration Time"])
    diff_time_extension=str(inputs["Difference Time"])
    em_extension=str(inputs["EM Increment"])
    temp_range_extension=str(inputs["Temperature Range"]).replace("(","").replace(")","").replace(", ","-")
    event_gap_extension=str(inputs["FAI Duration"])
    extension=start_extension+"_"+integ_time_extenstion+"_"+diff_time_extension+"_"+em_extension+"_"+temp_range_extension+"_"+event_gap_extension

    ts_og, smooth_plot,ts_1s=gf.timeseries_data(
        inputs["Start"],inputs["End"],inputs["Integration Time"])
    ts_diff=gf.running_difference(ts_og,inputs["Difference Time"])
    t_em_diff=gf.calc_temp_em(ts_diff)  # used in flagging calculations
    t_em_og=gf.calc_temp_em(ts_1s)      # used in temperature and emission plot
    flagged_df,flagged_times=gf.FAI_flagging(
        t_em_diff,
        inputs["EM Increment"],
        inputs["Temperature Range"],
        inputs["Difference Time"],
        inputs["FAI Duration"]
        )
    anticipation_plot=gf.anticipation_plot(
        smooth_plot,
        flagged_times,
        sgp,
        inputs["Start"],
        extension
        )
    temperature_emission_plot=gf.em_temp_plot(
        t_em_og,
        sgp,
        inputs["Start"],
        start_extension
    )
    gf.em_temp_plot(
        t_em_og,
        sgp,
        inputs["Start"],
        start_extension
    )

    update_plot_list()

def submit_run():
    submit()
    process_data()
#Button(params_tab,text="submit",command=submit).pack()
Button(params_tab,text="Run",command=submit_run).pack()
# submit()

image_frame=tk.Frame(root,width=800,height=400)
image_frame.pack(padx=5,pady=5,side=tk.RIGHT)

plot_list_frame = tk.Frame(image_frame, width=150)
plot_list_frame.pack(side=tk.RIGHT, fill=tk.Y)

plot_display_frame = tk.Frame(image_frame)
plot_display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
plot_listbox = tk.Listbox(plot_list_frame)
plot_listbox.pack(fill=tk.BOTH, expand=True)

def update_plot_list():
    plot_listbox.delete(0, tk.END)
    for fname in os.listdir(sgp):
        if fname.endswith(".png"):
            plot_listbox.insert(tk.END, fname)

update_plot_list()

def show_selected_plot(event):
    selection = plot_listbox.curselection()
    if selection:
        fname = plot_listbox.get(selection[0])
        filepath = os.path.join(sgp, fname)

        for widget in plot_display_frame.winfo_children():
            widget.destroy()

        img = tk.PhotoImage(file=filepath)  # or use PIL if needed for more formats
        label = tk.Label(plot_display_frame, image=img)
        label.image = img  # Keep a reference!
        label.pack(fill=tk.BOTH, expand=True)

plot_listbox.bind("<<ListboxSelect>>", show_selected_plot)

root.mainloop()
