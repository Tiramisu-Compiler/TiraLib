# import the tiralib library
from pathlib import Path
from tiralib.config import BaseConfig
from tiralib.tiramisu import Schedule, TiramisuProgram, tiramisu_actions

# initialize the TiraLib configuration
BaseConfig.init()

# Load the Tiramisu Program from the file and create a TiraLibCPP server
# IMPORTANT! This needs a working path to the TiraLibCPP library set in the config file of TiraLib
cpp_code = Path("./examples/function_blur_MINI_generator.cpp").read_text()
tiramisu_program = TiramisuProgram.init_server(
    cpp_code=cpp_code,
    load_annotations=True,
    load_tree=True,
    reuse_server=True,
)

# The rest of the code is the same as in using TiraLib alone

# Create a schedule object
schedule = Schedule(tiramisu_program)

# Apply the optimizations to the schedule
schedule.add_optimizations([tiramisu_actions.Parallelization([("comp_blur", 0)])])

# execute the schedule and get the execution times
execution_times = schedule.execute()

print(execution_times)
