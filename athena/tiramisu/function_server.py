import json
import logging
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from athena.utils.config import BaseConfig

if TYPE_CHECKING:
    from athena.tiramisu.schedule import Schedule
    from athena.tiramisu.tiramisu_program import TiramisuProgram

templateWithEverythinginUtils = """
#include <tiramisu/tiramisu.h>
#include <HermesII/actions.h>
#include <HermesII/utils.h>

using namespace tiramisu;

int main(int argc, char *argv[])
{{
    // check the number of arguemnts is 2 or 3
    assert(argc == 1 || argc == 2 || argc == 3 && "Invalid number of arguments");
    // get the operation to perform
    Operation operation = Operation::legality;

    if (argc >= 2)
    {{
        operation = get_operation_from_string(argv[1]);
    }}
    // get the schedule string if provided
    std::string schedule_str = "";
    if (argc == 3)
        schedule_str = argv[2];

    std::string function_name = "{name}";

    {body}

    schedule_str_to_result_str(function_name, schedule_str, operation, {buffers});
    return 0;
}}
"""  # noqa: E501


class ResultInterface:
    def __init__(self, result_str: bytes) -> None:
        decoded = result_str.decode("utf-8")
        decoded = decoded.strip().replace("\n", "\\n")

        self.halide_ir = None
        # extract halide ir and the result dict
        if "Generated Halide" in decoded:
            regex = r"Generated Halide IR:([\w\W\s]*)(?=\{\"name)(.*)"
            match = re.search(regex, decoded, re.MULTILINE | re.DOTALL)
            self.halide_ir = match.group(1)
            print(self.halide_ir.replace("\\n", "\n"))
            decoded = match.group(2)
        result_dict = json.loads(decoded)

        self.name = result_dict["name"]
        self.legality = result_dict["legality"] == 1
        self.isl_ast = result_dict["isl_ast"]
        self.success = result_dict["success"]

        # convert exec_times to list of floats
        self.exec_times = (
            [float(x) for x in result_dict["exec_times"].split()]
            if result_dict["exec_times"]
            else []
        )
        self.additional_info = (
            result_dict["additional_info"] if "additional_info" in result_dict else None
        )

    def __str__(self) -> str:
        isl_ast = self.isl_ast.replace("\n", ",")
        return f"ResultInterface(name={self.name},legality={self.legality},isl_ast={isl_ast},exec_times={self.exec_times},success={self.success})"  # noqa: E501

    def __repr__(self) -> str:
        return self.__str__()


class FunctionServer:
    def __init__(self, tiramisu_program: "TiramisuProgram", reuseServer: bool = False):
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")

        if not tiramisu_program.original_str:
            raise ValueError("Tiramisu program not initialized")

        self.tiramisu_program = tiramisu_program

        server_path_cpp = (
            Path(BaseConfig.base_config.workspace)
            / f"{tiramisu_program.name}_server.cpp"
        )

        server_path = (
            Path(BaseConfig.base_config.workspace) / f"{tiramisu_program.name}_server"
        )

        if reuseServer and server_path.exists():
            logging.info("Server code already exists. Skipping generation")
            return

        # Generate the server code
        server_code = FunctionServer._generate_server_code_from_original_string(
            tiramisu_program
        )

        # Write the server code to a file
        server_path_cpp.write_text(server_code)

        # Write the wrapper code to a file
        wrapper_path = (
            Path(BaseConfig.base_config.workspace)
            / f"{tiramisu_program.name}_wrapper.cpp"
        )
        wrapper_path.write_text(tiramisu_program.wrappers["cpp"])

        # Write the wrapper header to a file
        wrapper_header_path = (
            Path(BaseConfig.base_config.workspace)
            / f"{tiramisu_program.name}_wrapper.h"
        )

        wrapper_header_path.write_text(tiramisu_program.wrappers["h"])

        # compile the server code
        self._compile_server_code()

    @classmethod
    def _generate_server_code_from_original_string(
        self, tiramisu_program: "TiramisuProgram"
    ):
        original_str = tiramisu_program.original_str
        # Generate function
        body = re.findall(
            r"int main\([\w\s,*]+\)\s*\{([\W\w\s]*)tiramisu::codegen",
            original_str,
        )[0]
        name = re.findall(r"tiramisu::init\(\"(\w+)\"\);", original_str)[0]
        # Remove the wrapper include from the original string
        wrapper_str = f'#include "{name}_wrapper.h"'
        original_str = original_str.replace(wrapper_str, f"// {wrapper_str}")
        buffers_vector = re.findall(
            r"(?<=tiramisu::codegen\()\{[&\w,\s]+\}", original_str
        )[0]

        # fill the template
        function_str = templateWithEverythinginUtils.format(
            name=name,
            body=body,
            buffers=buffers_vector,
        )
        return function_str

    def _compile_server_code(self):
        env_vars = " && ".join(
            [
                f"export {key}={value}"
                for key, value in BaseConfig.base_config.env_vars.items()
            ]
        )

        lib = "lib"
        if Path.exists(
            Path(BaseConfig.base_config.env_vars["TIRAMISU_ROOT"])
            / "3rdParty/Halide/install/lib64"
        ):
            lib = "lib64"

        compileCommand = f"cd {BaseConfig.base_config.workspace} && {env_vars} && export FUNC_NAME={self.tiramisu_program.name} && $CXX -I$TIRAMISU_ROOT/3rdParty/Halide/install/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include -I$TIRAMISU_HERMESII_PATH/include -fvisibility-inlines-hidden -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -ffunction-sections -pipe -isystem $CONDA_ENV/include -ldl -g -fno-rtti -lpthread -std=c++17 -MD -MT ${{FUNC_NAME}}.cpp.o -MF ${{FUNC_NAME}}.cpp.o.d -o ${{FUNC_NAME}}.cpp.o -c ${{FUNC_NAME}}_server.cpp && $CXX -fvisibility-inlines-hidden -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -ffunction-sections -pipe -isystem $CONDA_ENV/include -ldl -g -fno-rtti -lpthread ${{FUNC_NAME}}.cpp.o -o ${{FUNC_NAME}}_server -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/install/{lib}  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/install/{lib}:$TIRAMISU_ROOT/3rdParty/isl/build/lib:$TIRAMISU_HERMESII_PATH/lib $TIRAMISU_HERMESII_PATH/lib/libHermesII.so -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl -lsqlite3 -lz"  # noqa: E501

        # run the command and retrieve the execution status
        try:
            subprocess.check_output(compileCommand, shell=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error while compiling server code: {e}")
            logging.error(e.output)
            print(e.output)
            print(e.stderr)
            raise e

    def run(
        self,
        operation: Literal["execution", "legality"] = "legality",
        schedule: "Schedule | None" = None,
        nbr_executions: int = 30,
    ):
        assert (
            operation
            in [
                "execution",
                "legality",
            ]
        ), f"Invalid operation {operation}. Valid operations are: execution, legality, annotations"  # noqa: E501

        env_vars = " && ".join(
            [
                f"export {key}={value}"
                for key, value in BaseConfig.base_config.env_vars.items()
            ]
        )

        command = f'{env_vars} && cd {BaseConfig.base_config.workspace} && NB_EXEC={nbr_executions} ./{self.tiramisu_program.name}_server {operation} "{schedule or ""}"'  # noqa: E501

        # run the command and retrieve the execution status
        try:
            output = subprocess.check_output(command, shell=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error while running server code: {e}")
            logging.error(e.output)
            print(e.output)
            print(e.stderr)
            raise e
        return ResultInterface(output)

    def get_annotations(self):
        env_vars = " && ".join(
            [
                f"export {key}={value}"
                for key, value in BaseConfig.base_config.env_vars.items()
            ]
        )

        command = f"{env_vars} && cd {BaseConfig.base_config.workspace} && ./{self.tiramisu_program.name}_server annotations"  # noqa: E501

        # run the command and retrieve the execution status
        try:
            output = subprocess.check_output(command, shell=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error while running server code: {e}")
            logging.error(e.output)
            print(e.output)
            print(e.stderr)
            raise e

        return output.decode("utf-8").strip()
