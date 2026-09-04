# TiraLib Python: A Tiramisu Compiler Python Frontend For Loading Tiramisu Programs and Building and Executing Schedules

## Introduction
TiraLib Python is a Python frontend for the Tiramisu compiler. It allows users to build schedules for Tiramisu programs and execute them. It also allows users to generate C++ code for their Tiramisu schedules and execute it.

## Installation
To install TiraLib Python, you need to install the Tiramisu compiler first. Please follow the instructions [here](https://github.com/Tiramisu-Compiler/tiramisu).

### Respository
Then, you can install tiralib Python by cloning this repository and running the following command:
```
cd tiralib
poetry install
```

### Install as library from GitHub
You can also install tiralib Python as a library from github by running the following command:
```
poetry add git+https://github.com/Tiramisu-Compiler/TiraLib
```
or using pip:
```
pip install git+https://github.com/Tiramisu-Compiler/TiraLib
```

## Usage and Features

### Activating the Virtual Environment
If you installed TiraLib using poetry in a .venv environement then you need to activate the virtual environment created by Poetry:
```bash
poetry shell
```

### Configuration File
TiraLib Python uses a configuration file to specify the paths to the Tiramisu compiler and the Tiramisu runtime. The configuration file is named `config.yaml` and should be placed in the root directory of the project. The configuration file should have the following format (more details can be found in the config module):

```yaml
env_vars:
  CXX: "${CXX}"
  CC: "${CC}"

dependencies:
    includes:
        - path to where the include of dependencies are
    libs:
        - path to lib where dependencies are installed
```

A `config.yaml.example` file is provided in the root directory of the project. You can use it as a template for your configuration file.

Before running any TiraLib Python code, you need to load the configuration file using the following code:

```python
from tiralib.config import BaseConfig

BaseConfig.init()
```


### Loading a Tiramisu Program
To load a Tiramisu program, you need to create a `TiramisuProgram` object and pass the path to the Tiramisu program to its `from_file` constructor, and set the `load_annotations` and `load_tree` parameters to `True` if you want to load the annotations and the AST tree of the Tiramisu program respectively:

```python
from tiralib.tiramisu import TiramisuProgram, Schedule, tiramisu_actions
from tiralib.config import BaseConfig

BaseConfig.init()

tiramisu_program = TiramisuProgram.from_file(
    "./examples/function_blur_MINI_generator.cpp", load_annotations=True, load_tree=True
)

print(tiramisu_program.tree)

```

### Building a Schedule
To build a schedule for a Tiramisu program, you need to create a `Schedule` object and pass the `TiramisuProgram` object to its constructor:

```python
from tiralib.tiramisu import TiramisuProgram, Schedule, tiramisu_actions
from tiralib.config import BaseConfig

BaseConfig.init()

tiramisu_program = TiramisuProgram.from_file(
    "./examples/function_blur_MINI_generator.cpp", load_annotations=True, load_tree=True
)

schedule = Schedule(tiramisu_program)
```

### Scheduling
tiralib Python provides a set of code transformations that can be used to build schedules for Tiramisu programs. These transformations are implemented as `TiramisuAction` objects.

To add a transformation to a schedule, you need to call the `add_optimizations` method of the `Schedule` object and pass the `TiramisuAction` object to it:

```python
from tiralib.tiramisu import TiramisuProgram, Schedule, tiramisu_actions
from tiralib.config import BaseConfig

BaseConfig.init()

tiramisu_program = TiramisuProgram.from_file(
    "./examples/function_blur_MINI_generator.cpp", load_annotations=True, load_tree=True
)

schedule = Schedule(tiramisu_program)

schedule.add_optimizations([tiramisu_actions.Parallelization([("comp_blur", 0)])])
```

You can find the list of all the transformations implemented in tiralib Python [here](./tiralib/tiramisu/tiramisu_actions/)

### Legality Checking

To check the legality of a schedule, you need to call the `is_legal` method of the `Schedule` object:

```python
from tiralib.tiramisu import TiramisuProgram, Schedule, tiramisu_actions
from tiralib.config import BaseConfig

BaseConfig.init()

tiramisu_program = TiramisuProgram.from_file(
    "./examples/function_blur_MINI_generator.cpp", load_annotations=True, load_tree=True
)

schedule = Schedule(tiramisu_program)

schedule.add_optimizations([tiramisu_actions.Parallelization([("comp_blur", 0)])])

if schedule.is_legal():
    print("The schedule is legal")
else:
    print("The schedule is illegal")
```

### Execution

To execute a schedule, you need to call the `execute` method of the `Schedule` object:

```python
from tiralib.tiramisu import TiramisuProgram, Schedule, tiramisu_actions
from tiralib.config import BaseConfig

BaseConfig.init()

tiramisu_program = TiramisuProgram.from_file(
    "./examples/function_blur_MINI_generator.cpp", load_annotations=True, load_tree=True
)

schedule = Schedule(tiramisu_program)

schedule.add_optimizations([tiramisu_actions.Parallelization([("comp_blur", 0)])])

execution_times = schedule.execute()

print(execution_times)
```

### Measurement Harnesses

Execution-time measurement is performed by a *harness* — the generated
wrapper binary that allocates and initializes buffers, calls the compiled
function, and prints one time (in milliseconds) per measurement. TiraLib
provides two:

- **`DefaultHarness`** (used unless stated otherwise): buffers are filled
  once with a random constant, and the kernel is timed `min_runs` /
  `max_runs` times back-to-back in a single (warm) process with
  `std::chrono`. Fast — well suited for autotuning searches.

- **`PolybenchHarness`**: reproduces the PolyBench/C 4.2.1 measurement
  methodology for PolyBench-derived programs, making results directly
  comparable with other tools evaluated on PolyBench. Each measurement runs
  in a **fresh process** (like PolyBench's `time_benchmark.sh`) and follows
  the exact PolyBench protocol using the vendored, unmodified
  `polybench.h`/`polybench.c`: page-aligned `polybench_alloc_data`
  allocation, deterministic PolyBench `init_array` initialization, cache
  flush + `gettimeofday` timer (`polybench_timer_start`/`stop`), one kernel
  invocation. Times are still reported in milliseconds.

By default the `PolybenchHarness` also runs the kernel on an
**OpenMP-backed parallel runtime with bound threads**
(`runtime="openmp"`, `omp_schedule="static,1"`, `OMP_PROC_BIND=close` /
`OMP_PLACES=cores`): Halide parallel loops execute as OpenMP loops (via
`halide_set_custom_parallel_runtime`), so the thread team is created and
pinned by the untimed PolyBench cache flush *before* the timer starts —
the same conditions the stock PolyBench harness gives OpenMP-based tools
(e.g. Pluto). This measured a 1.26× geometric-mean improvement (2.0× on
parallel schedules) over the Halide thread pool across the 150 PolyBench
program/size combinations, with deterministic scheduling and sub-1%
run-to-run deviation. Pass `runtime="halide"` to keep Halide's own
work-stealing pool. In both modes the wrapper manages the OpenMP binding
environment for its measurement children itself, so stray
`OMP_PROC_BIND` settings in the calling environment cannot skew or
serialize measurements.

The harness is selected when loading the program. `TiramisuProgram.from_file`
**auto-detects** PolyBench programs: if an init sidecar
(`<function_name>_init.h`, generated by the `polybench_tiramisu`
repository's `tools/generate_polybench_init_sidecars.py`) exists next to the
generator file, the `PolybenchHarness` is used; otherwise the
`DefaultHarness`. You can always override explicitly:

```python
from tiralib.tiramisu.harness import DefaultHarness, PolybenchHarness

# auto-detection (PolybenchHarness if a *_init.h sidecar is present):
program = TiramisuProgram.from_file("function_gemm_MINI_generator.cpp")

# force the default harness (e.g. for fast autotuning searches):
program = TiramisuProgram.from_file(
    "function_gemm_MINI_generator.cpp", harness=DefaultHarness()
)

# explicit PolyBench harness with options:
program = TiramisuProgram.from_file(
    "function_gemm_MINI_generator.cpp",
    harness=PolybenchHarness("function_gemm_MINI_init.h", cache_size_kb=32770),
)

times = Schedule(program).execute(min_runs=5)  # 5 fresh-process measurements
```

To aggregate the measurements the way PolyBench's `time_benchmark.sh` does
(5 runs, drop min and max, mean of the middle 3, variance check):

```python
from tiralib.tiramisu.harness import polybench_normalized_time

normalized, deviation_pct = polybench_normalized_time(times)
```

With the `PolybenchHarness`, setting the environment variable
`TIRALIB_DUMP_ARRAYS=1` when running the generated wrapper dumps the
live-out arrays to stderr in PolyBench's `POLYBENCH_DUMP_ARRAYS` format, so
schedule correctness can be checked by diffing against the reference
PolyBench/C output. Both harnesses work with both execution backends (the
pure-Python `CompilingService` and the TiraLibCpp server).

Wrapper protocol (identical for both harnesses; this is what
`schedule.execute()` drives and what external scripts can rely on):

| Channel | Meaning |
|---|---|
| `MIN_RUNS` (env) | guaranteed, non-abortable measurements |
| `MAX_RUNS` (env) | measurement cap when a time budget is set (`inf` = unlimited) |
| `TIME_BUDGET` (env) | total budget in ms; aborts the in-flight run once `MIN_RUNS` are done |
| `TIRALIB_DUMP_ARRAYS` (env) | PolybenchHarness only: dump live-out arrays to stderr |
| stdout | one float per completed measurement, **milliseconds** |

A "measurement" is one warm in-process kernel invocation under the
`DefaultHarness`, and one full PolyBench-style fresh-process run
(deterministic init → cache flush → single kernel invocation) under the
`PolybenchHarness`.

The init sidecars (`<function_name>_init.h`) consumed by the
`PolybenchHarness` are generated by — and documented in — the
[polybench-tiramisu](https://github.com/Mascinissa/polybench-tiramisu)
repository, which also validates the harness against stock PolyBench/C.
See `examples/polybench_example.py` for a runnable example.


## Development

### Testing
To run the tests, you need to activate the virtual environment created by Poetry:
```bash
poetry shell
```

Then, you can run the tests using the following command:

```bash
pytest
```

### Coverage
To run the tests and generate the coverage report, you need to activate the virtual environment created by Poetry:
```bash
poetry shell
```

Then, you can run the tests using the following command:

```bash
pytest --cov
```

Finally, you can generate the coverage report using the following command:

```bash
coverage report
```

For HTML coverage report, you can use the following command:

```bash
coverage html --include="tiralib/**/*"
```

### Code Formatting
The library uses the ruff code formatter. To format the code, you need to activate the virtual environment created by Poetry:
```bash
poetry shell
```

Then, you can format the code using the following command:

```bash
ruff format .
```