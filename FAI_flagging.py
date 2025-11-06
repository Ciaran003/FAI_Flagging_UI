import os
import sys
from tkinter import *
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from functions import GOES_functions as gf#type: ignore
from functions import time_inputs as time #type: ignore
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import io
import tkinter.scrolledtext as scrolledtext

base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
sgp = os.path.join(base_dir, "saved_graphs")
inputs={}

def submit():
    param_text = "Parameters:\n"
    # Handle Custom Data Input
    if date_var.get() == "Specific Data":
        inputs["Start"] = start_date_entry.get()
        inputs["End"] = end_date_entry.get()
    # Handle Live Data
    elif date_var.get() == "Live Data":
        inputs["Start"] = time.get_time_minus_lookback(lookback_dropdown.get())
        inputs["End"] = time.get_current_time().strftime('%Y-%m-%d %H:%M:%S')
    # Handle Live Data
    elif date_var.get() == "Bulk Data":
        csv=os.path.join(base_dir,csv_entry.get())
        starts,ends,flare_class= time.batch_times(csv)
        inputs["Start"] =starts
        inputs["End"] =ends
    
    if date_var.get() == "Specific Data":
        param_text += f"Start Date: {inputs['Start']}\n"
        param_text += f"End Date: {inputs['End']}\n"
    elif date_var.get() == "Live Data":
        param_text += f"Start Date: {inputs['Start']}\n"
        param_text += f"End Date: {inputs['End']}\n"
    elif date_var.get() == "Bulk Data":
        param_text += f"Start Date: Bulk\n"
        param_text += f"End Date: Bulk\n"
    
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
params_frame.pack(padx=10,pady=10,side=tk.LEFT,fill=tk.Y)
tk.Label(
    params_frame,
    text="Parameters"
).pack(padx=5,pady=5)
tk.Label(params_frame).pack(padx=5, pady=5)
# params_display_label = tk.Label(params_frame, text="Parameters:\n(none)", justify="left")
initial_params_text = "Parameters:\n"
for param in [
    "Start Date: ",
    "End Date: ",
    "Integration Time (s): ",
    "Difference Time (s): ",
    "EM Increment ⁴⁹ cm⁻³: ",
    "Temperature Range (MK): ",
    "FAI Duration (mins): "
]:
    initial_params_text += f"{param}\n"

params_display_label = tk.Label(params_frame, text=initial_params_text, justify="left")
params_display_label.pack(padx=5, pady=5)

# Input parameters and settings
notebook=ttk.Notebook(params_frame)
notebook.pack(expand=True,fill="both")

def toggle_timeframe_fields(show_custom):
    if show_custom==1:
        start_date_label.pack(anchor="w", padx=10, pady=(10, 0))
        start_date_entry.pack(anchor="w", padx=10, pady=(10, 0))
        end_date_label.pack(anchor="w", padx=10, pady=(10, 0))
        end_date_entry.pack(anchor="w", padx=10, pady=(10, 0))
        lookback_label.pack_forget()
        lookback_dropdown.pack_forget()
        csv_label.pack_forget()
        csv_entry.pack_forget()
    elif show_custom==2:
        start_date_label.pack_forget()
        start_date_entry.pack_forget()
        end_date_label.pack_forget()
        end_date_entry.pack_forget()
        csv_label.pack_forget()
        csv_entry.pack_forget()
        lookback_label.pack(anchor="w", padx=10, pady=(10, 0))
        lookback_dropdown.pack(anchor="w", padx=10, pady=(10, 0))
    elif show_custom==3:
        csv_label.pack(anchor="w", padx=10, pady=(10, 0))
        csv_entry.pack(anchor="w", padx=10, pady=(10, 0))
        lookback_label.pack_forget()
        lookback_dropdown.pack_forget()
        start_date_label.pack_forget()
        start_date_entry.pack_forget()
        end_date_label.pack_forget()
        end_date_entry.pack_forget()

date_tab=tk.Frame(notebook)
date_var=tk.StringVar(value="None")
date_entries = {}  # Dictionary to hold text inputs

options=[
    "Start (YYYY-MM-DD HR:MIN:SEC)",
    "End (YYYY-MM-DD HR:MIN:SEC)",
]

custom_radio = tk.Radiobutton(date_tab, text="Specific Data", variable=date_var, value="Specific Data", command=lambda: toggle_timeframe_fields(1))
custom_radio.pack(anchor="w", padx=10, pady=(10, 0))

live_radio = tk.Radiobutton(date_tab, text="Live Data", variable=date_var, value="Live Data", command=lambda: toggle_timeframe_fields(2))
live_radio.pack(anchor="w", padx=10, pady=(10, 0))

bulk_radio = tk.Radiobutton(date_tab, text="Bulk Data", variable=date_var, value="Bulk Data", command=lambda: toggle_timeframe_fields(3))
bulk_radio.pack(anchor="w", padx=10, pady=(10, 0))

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

csv_label = tk.Label(date_tab, text="CSV file:")
csv_label.pack(anchor="w", padx=10, pady=(10, 0))
csv_entry = tk.Entry(date_tab)
csv_entry.pack(anchor="w", padx=10, pady=(10, 0))

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
    "EM Increment ⁴⁹ cm⁻³",
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
        ts_og,
        t_em_diff,
        inputs["EM Increment"],
        inputs["Temperature Range"],
        inputs["Difference Time"],
        inputs["FAI Duration"])

    anticipation_plot=gf.anticipation_plot(
        smooth_plot,
        flagged_times,
        sgp,
        inputs["Start"],
        extension,
        integ_time_extenstion,
        diff_time_extension,
        em_extension,
        inputs["Temperature Range"],
        event_gap_extension
)
    
    temperature_emission_plot=gf.diagnostic_plot(
        t_em_og,
        sgp,
        inputs["Start"],
        start_extension)
    print("FINISHED")

def bulk_process():
    for i, start_date in enumerate(inputs["Start"]):
        end_date=inputs["End"][i]

        # start_extension=inputs["Start"].replace(' ', '_').replace(':', '-')
        start_extension=str(start_date).replace(' ', '_').replace(':', '-')
        integ_time_extenstion=str(inputs["Integration Time"])
        diff_time_extension=str(inputs["Difference Time"])
        em_extension=str(inputs["EM Increment"])
        temp_range_extension=str(inputs["Temperature Range"]).replace("(","").replace(")","").replace(", ","-")
        event_gap_extension=str(inputs["FAI Duration"])
        extension=start_extension+"_"+integ_time_extenstion+"_"+diff_time_extension+"_"+em_extension+"_"+temp_range_extension+"_"+event_gap_extension

        ts_og, smooth_plot,ts_1s=gf.timeseries_data(
            start_date,end_date,inputs["Integration Time"])
        ts_diff=gf.running_difference(ts_og,inputs["Difference Time"])
        t_em_diff=gf.calc_temp_em(ts_diff)  # used in flagging calculations
        t_em_og=gf.calc_temp_em(ts_1s)      # used in temperature and emission plot
        flagged_df,flagged_times=gf.FAI_flagging(
            ts_og,
            t_em_diff,
            inputs["EM Increment"],
            inputs["Temperature Range"],
            inputs["Difference Time"],
            inputs["FAI Duration"])

        anticipation_plot=gf.anticipation_plot(
            smooth_plot,
            flagged_times,
            sgp,
            start_date,
            extension,
            integ_time_extenstion,
            diff_time_extension,
            em_extension,
            inputs["Temperature Range"],
            event_gap_extension
    )

        diagnostic_plot=gf.diagnostic_plot(
            t_em_og,
            sgp,
            start_date,
            start_extension)
    print("FINISHED")
def submit_run():
    submit()
    if date_var.get() == "Specific Data" or date_var.get() == "Live Data":
        process_data()
    else:
        bulk_process()
    update_plot_list()
# def submit_run():
#     submit()
#     bulk_process()
#     update_plot_list()
# Button(params_tab,text="submit",command=submit).pack()
Button(params_tab,text="Run",command=submit_run).pack()
# submit()

### Frames 
image_frame = tk.Frame(root, width=800, height=400)
image_frame.pack(padx=5, pady=5, side=tk.RIGHT, fill=tk.BOTH, expand=True)

plot_list_frame = tk.Frame(image_frame)
plot_list_frame.pack(side=tk.LEFT, anchor="n", padx=5, pady=5, fill=tk.Y)

# plot_display_frame = tk.Frame(image_frame)
# plot_display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

# Container on the right side for both plot + console
right_display_container = tk.Frame(image_frame)
right_display_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Top: plot display area
plot_display_frame = tk.Frame(right_display_container, height=400)
plot_display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Bottom: console area
console_frame = tk.Frame(right_display_container, height=150)
console_frame.pack(side=tk.BOTTOM, fill=tk.X)

console_text = scrolledtext.ScrolledText(
    console_frame, wrap="word", height=8,
    bg="white", fg="black", insertbackground="white"#, font=("Consolas", 10)
)
console_text.pack(expand=True, fill="both", padx=5, pady=5)


# --- Canvas + Scrollbar for thumbnails ---
canvas = tk.Canvas(plot_list_frame,width=120)
scrollbar = ttk.Scrollbar(plot_list_frame, orient="vertical", command=canvas.yview)
thumb_container = tk.Frame(canvas)

thumb_container.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=thumb_container, anchor="n")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# --- Display selected image ---
def show_selected_plot(filepath):
    for widget in plot_display_frame.winfo_children():
        widget.destroy()
        # def resize_image(event):
        #     show_selected_plot(filepath)  # reload and rescale image
        # plot_display_frame.bind("<Configure>", lambda e: resize_image(e))


    img = Image.open(filepath)
    img_tk = ImageTk.PhotoImage(img)
    label = tk.Label(plot_display_frame, image=img_tk)
    label.image = img_tk  # keep reference
    # label.pack(fill=tk.BOTH, expand=True)
    label.pack(expand=True, anchor="center")

def update_plot_list():
    # Clear previous thumbnails
    for widget in thumb_container.winfo_children():
        widget.destroy()

    # Get and sort images by modification time (newest first)
    png_files = [
        os.path.join(sgp, f)
        for f in os.listdir(sgp)
        if f.lower().endswith(".png")
    ]
    png_files.sort(key=os.path.getmtime, reverse=True)

    thumb_size = (100, 100)
    thumb_refs = []

    for filepath in png_files:
        img = Image.open(filepath)
        img.thumbnail(thumb_size)
        img_tk = ImageTk.PhotoImage(img)
        thumb_refs.append(img_tk)

        # Directly pack the thumbnail (no frame, no filename)
        label = tk.Label(
            thumb_container,
            image=img_tk,
            cursor="hand2",
            bd=2,
            relief=tk.RIDGE
        )
        label.image = img_tk

        # Pack it to fill horizontally (this removes the gap)
        label.pack(padx=5, pady=5)

        # Bind click event
        label.bind("<Button-1>", lambda e, fp=filepath: show_selected_plot(fp))

    # Keep references so they aren't garbage-collected
    thumb_container.thumbs = thumb_refs

update_plot_list()



# Redirect stdout and stderr to the console text widget
class ConsoleRedirector(io.StringIO):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, message):
        # Insert text in the widget, scroll to the end
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        # Update immediately for live feedback
        self.text_widget.update_idletasks()

    def flush(self):
        pass  # required for file-like behavior but not used here

# Redirect print() and errors
sys.stdout = ConsoleRedirector(console_text)
sys.stderr = ConsoleRedirector(console_text)






root.mainloop()
