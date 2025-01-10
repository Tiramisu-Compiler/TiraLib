import json
import logging
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from tiralib.config import BaseConfig

if TYPE_CHECKING:
    from tiralib.tiramisu.schedule import Schedule
    from tiralib.tiramisu.tiramisu_program import TiramisuProgram

logger = logging.getLogger(__name__)

templateWithEverythinginUtils = """
#include <tiramisu/tiramisu.h>
#include <TiraLibCPP/actions.h>
#include <TiraLibCPP/utils.h>

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
    """Result interface for the function server."""

    def __init__(self, result_str: bytes) -> None:
        """Initialize the result interface.

        Args:
            result_str (bytes): The result string.
        """
        decoded = result_str.decode("utf-8")
        decoded = decoded.strip().replace("\n", "\\n")

        self.halide_ir = None
        # extract halide ir and the result dict
        if "Generated Halide" in decoded:
            regex = r"Generated Halide IR:([\w\W\s]*)(?=\{\"name)(.*)"
            match = re.search(regex, decoded, re.MULTILINE | re.DOTALL)
            if match is None:
                raise ValueError(f"Could not parse the result string: {decoded}")
            self.halide_ir = match.group(1)
            logger.debug(self.halide_ir.replace("\\n", "\n"))
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
        """Return a string representation of the object."""
        isl_ast = self.isl_ast.replace("\n", ",")
        return f"ResultInterface(name={self.name},legality={self.legality},isl_ast={isl_ast},exec_times={self.exec_times},success={self.success})"  # noqa: E501

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return self.__str__()


class FunctionServer:
    """Function server class."""

    def __init__(self, tiramisu_program: "TiramisuProgram", reuse_server: bool = False):
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")

        if not tiramisu_program.original_str:
            raise ValueError("Tiramisu program not initialized")
        if not tiramisu_program.wrappers:
            raise ValueError("Tiramisu program wrappers not initialized")

        self.tiramisu_program = tiramisu_program

        server_path_cpp = (
            Path(BaseConfig.base_config.workspace)
            / f"{tiramisu_program.name}_server.cpp"
        )

        server_path = (
            Path(BaseConfig.base_config.workspace) / f"{tiramisu_program.name}_server"
        )

        if reuse_server and server_path.exists():
            logger.info("Server code already exists. Skipping generation")
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
        if original_str is None:
            raise ValueError("Original string not initialized")
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
        """Compile the server code."""
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")

        libs = ":".join(BaseConfig.base_config.dependencies.libs)
        env_vars = " && ".join(
            [
                f"export {key}={value}"
                for key, value in BaseConfig.base_config.env_vars.items()
            ]
        )

        env_vars += f" && export LD_LIBRARY_PATH={libs}:$LD_LIBRARY_PATH"
        env_vars += f" && export LIBRARY_PATH={libs}:$LIBRARY_PATH"
        env_vars += (
            f" && export CPATH={':'.join(BaseConfig.base_config.dependencies.includes)}"
        )

        libs = ":".join(BaseConfig.base_config.dependencies.libs)

        compile_command = f"cd {BaseConfig.base_config.workspace} && {env_vars} && export FUNC_NAME={self.tiramisu_program.name} && $CXX -fvisibility-inlines-hidden -ftree-vectorize  -fstack-protector-strong -fno-plt -O3 -ffunction-sections -pipe -ldl -g -fno-rtti -lpthread -std=c++17 -MD -MT ${{FUNC_NAME}}.cpp.o -MF ${{FUNC_NAME}}.cpp.o.d -o ${{FUNC_NAME}}.cpp.o -c ${{FUNC_NAME}}_server.cpp && $CXX -fvisibility-inlines-hidden -ftree-vectorize  -fstack-protector-strong -fno-plt -O3 -ffunction-sections -pipe -ldl -g -fno-rtti -lpthread ${{FUNC_NAME}}.cpp.o -o ${{FUNC_NAME}}_server -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl -lTiraLibCPP {'-lsqlite3' if BaseConfig.base_config.tiralib_cpp.use_sqlite else ''} -lz"  # noqa: E501

        # run the command and retrieve the execution status
        try:
            subprocess.check_output(compile_command, shell=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while compiling server code: {e}")
            logger.error(e.output)
            logger.error(e.stderr)
            raise e

    def run(
        self,
        operation: Literal["execution", "legality"] = "legality",
        schedule: "Schedule | None" = None,
        nbr_executions: int = 30,
    ):
        """Run the server code."""
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")
        assert operation in [
            "execution",
            "legality",
        ], (
            f"Invalid operation {operation}. Valid operations are: execution, legality, annotations"
        )  # noqa: E501

        env_vars = " && ".join(
            [
                f"export {key}={value}"
                for key, value in BaseConfig.base_config.env_vars.items()
            ]
        )

        libs = ":".join(BaseConfig.base_config.dependencies.libs)
        env_vars += f" && export LD_LIBRARY_PATH={libs}:$LD_LIBRARY_PATH"
        env_vars += f" && export LIBRARY_PATH={libs}:$LIBRARY_PATH"
        env_vars += (
            f" && export CPATH={':'.join(BaseConfig.base_config.dependencies.includes)}"
        )

        command = f'{env_vars} && cd {BaseConfig.base_config.workspace} && NB_EXEC={nbr_executions} ./{self.tiramisu_program.name}_server {operation} "{schedule or ""}"'  # noqa: E501

        # run the command and retrieve the execution status
        try:
            output = subprocess.check_output(command, shell=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while running server code: {e}")
            logger.error(e.output)
            logger.error(e.stderr)
            raise e
        return ResultInterface(output)

    def get_annotations(self):
        """Run the server code to get the annotations."""
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")
        env_vars = " && ".join(
            [
                f"export {key}={value}"
                for key, value in BaseConfig.base_config.env_vars.items()
            ]
        )
        libs = ":".join(BaseConfig.base_config.dependencies.libs)
        env_vars += f" && export LD_LIBRARY_PATH={libs}:$LD_LIBRARY_PATH"
        env_vars += f" && export LIBRARY_PATH={libs}:$LIBRARY_PATH"
        env_vars += (
            f" && export CPATH={':'.join(BaseConfig.base_config.dependencies.includes)}"
        )

        command = f"{env_vars} && cd {BaseConfig.base_config.workspace} && ./{self.tiramisu_program.name}_server annotations"  # noqa: E501

        # run the command and retrieve the execution status
        try:
            output = subprocess.check_output(command, shell=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while running server code: {e}")
            logger.error(e.output)
            logger.error(e.stderr)
            raise e

        return output.decode("utf-8").strip()
