# FAI_Flagging_UI
UI programme for FAI flagging

The GUI application can be started by running FAI_flagging_UI.py
This opens a GUI which can apply data time ranges and the desired parameters, while easily displaying the plots created.
The live functionality listed within this GUI code uses Fido to retrieve the data,
as fido is processed data, there is roughly a 3 day delay in "live" data (newest data is 3 days old).

To run flagging live, FAI_live.py can be used.
This runs in a terminal,
writing flare times detected into a csv,
along with the class of the flag (may occur at different class to peak flare).

To run both of these programmes successfully, some python functions must be installed.
In my experience (linux Mint), these packages needed to be installed within a virtual environment.

Another requirement for this to be executed successfully is file structure:
- FAI_flagging.py
- FAI_live.py
- functions
  -> GOES_functions.py
  -> GOES_live.py
  -> live_inputs.py
  -> time_inputs.py
- flagged_times
- saved_graphs
- batch.csv
