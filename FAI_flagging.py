import os
import sys
from tkinter import *
import tkinter as tk
from tkinter import ttk
from functions import GOES_functions as gf#type: ignore
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk



base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
sgp = os.path.join(base_dir, "saved_graphs")
inputs={}
def submit():
    param_text = "Current Parameters:\n"
    for key, entry in parameter_entries.items():
        value = entry.get()
        # param_text += f"{key}: {value}\n"
        display_key = key.split('(')[0].strip()
        param_text += f"{display_key}: {value}\n"
    params_display_label.config(text=param_text)

    print("Submitted Parameters:")
    for key, entry in parameter_entries.items():
        print(f"{key}: {entry.get()}")

    print("\nSubmitted Settings:")
    # for key, var in settings_vars.items():
    #     print(f"{key}: {var.get()}")
    inputs["Start"] = parameter_entries["Start (YYYY-MM-DD HR:MIN:SEC)"].get()
    inputs["End"] = parameter_entries["End (YYYY-MM-DD HR:MIN:SEC)"].get()
    inputs["Integration Time"] =int( parameter_entries["Integration Time (s)"].get())
    inputs["Difference Time"] = int(parameter_entries["Difference Time (s)"].get())
    inputs["EM Increment"] = float(parameter_entries[f"EM Increment ⁴⁹ cm⁻³"].get())
    temp_range_str=(parameter_entries["Temperature Range (MK)"].get())
    temp_range = [float(i) for i in temp_range_str.split(",")]
    inputs["Temperature Range"] = temp_range
    inputs["FAI Duration"] = int(parameter_entries["FAI Duration (mins)"].get())
    
root=tk.Tk()
root.title("FAI Flagging")
# root.state('-zoomed',True)
# root.update_idletasks()
# root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
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

# parameter tab
params_tab=tk.Frame(notebook)
params_var=tk.StringVar(value="None")
parameter_entries = {}  # Dictionary to hold text inputs

parameters = [
    "Start (YYYY-MM-DD HR:MIN:SEC)",
    "End (YYYY-MM-DD HR:MIN:SEC)",
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

# settings_tab = tk.Frame(notebook)
# settings_vars = {}

# settings = ["Save Plot", "Show Plot"]
# options = ["yes", "no"]

# for setting in settings:
#     tk.Label(settings_tab, text=setting).pack(anchor="w", padx=10, pady=(10, 0))
#     var = tk.StringVar(value="yes")
#     dropdown = tk.OptionMenu(settings_tab, var, *options)
#     dropdown.pack(anchor="w", padx=10, pady=(0, 10))
#     settings_vars[setting] = var

# notebook.add(date_tab,text="Data Times")
notebook.add(params_tab,text="Parameters")
# notebook.add(settings_tab,text="Settings")

tk.Button(params_tab,text="Submit",command=submit).pack(pady=10)

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
    # if os.path.exists(filepath):
    #     for widget in plot_display_frame.winfo_children():
    #         widget.destroy()

    #     try:
    #         img = tk.PhotoImage(file=filepath)
    #     except tk.TclError:
    #         from PIL import Image, ImageTk  # fallback for non-PNGs
    #         img = Image.open(filepath)
    #         img = ImageTk.PhotoImage(img)

    #     label = tk.Label(plot_display_frame, image=img)
    #     label.image = img  # Prevent garbage collection
    #     label.pack(fill=tk.BOTH, expand=True)

Button(params_tab,text="Run",command=process_data).pack()


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
