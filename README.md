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
coverage run -m pytest
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