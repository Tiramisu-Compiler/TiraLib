# import the tiralib library
from tiralib.tiramisu import TiramisuProgram, Schedule, tiramisu_actions
from tiralib.config.config import BaseConfig

# initialize the TiraLib configuration
BaseConfig.init()

# Load the Tiramisu Program from the file
tiramisu_program = TiramisuProgram.from_file(
    "./examples/function_blur_MINI_generator.cpp", load_annotations=True, load_tree=True
)

# Create a schedule object
schedule = Schedule(tiramisu_program)

# Apply the optimizations to the schedule
schedule.add_optimizations([tiramisu_actions.Parallelization([("comp_blur", 0)])])

# execute the schedule and get the execution times
execution_times = schedule.execute()

print(execution_times)
