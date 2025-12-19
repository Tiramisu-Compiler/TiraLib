from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import TYPE_CHECKING, List, Literal

from tiralib.config import BaseConfig
from tiralib.tiramisu.tiramisu_tree import TiramisuTree

if TYPE_CHECKING:
    from tiralib.tiramisu.schedule import Schedule
    from tiralib.tiramisu.tiramisu_actions.tiramisu_action import TiramisuAction
    from tiralib.tiramisu.tiramisu_program import TiramisuProgram

logger = logging.getLogger(__name__)


class CompilingService:
    """Compile Tiramisu code and run it to get the results.

    Class responsible of compiling the generated code and running it
    to get the results Contains nothing but class methods
    """

    @classmethod
    def compile_legality(cls, schedule: Schedule, with_ast: bool = False):
        """Compile and run legality of the schedule.

        Args:
            schedule (Schedule): The schedule to check legality for
            with_ast (bool, optional): If true, the AST will be returned. Defaults to False.

        Returns:
            bool: True if the schedule is legal, False otherwise
        """
        assert BaseConfig.base_config
        assert schedule.tiramisu_program

        output_path = os.path.join(
            BaseConfig.base_config.workspace,
            f"{schedule.tiramisu_program.temp_files_identifier}_legality",
        )

        cpp_code = cls.get_legality_code(schedule=schedule, with_ast=with_ast)

        logger.debug("Legality Code: \n" + cpp_code)

        result = cls.run_cpp_code(cpp_code=cpp_code, output_path=output_path)

        if with_ast:
            result_lines = result.split("\n")
            legality_result = result_lines[0]
            legality_result = legality_result.strip()
            if legality_result not in ["0", "1"]:
                raise Exception(f"Error in legality check: {legality_result}")
            ast = TiramisuTree.from_isl_ast_string_list(
                isl_ast_string_list=result_lines[1:]
            )
            return legality_result == "1", ast

        else:
            result = result.strip()
            if result not in ["0", "1"]:
                raise Exception(f"Error in legality check: {result}")
            return result == "1", None

    @classmethod
    def get_legality_code(cls, schedule: Schedule, with_ast: bool = False):
        """Construct the code to check legality of the schedule.

        Args:
            schedule (Schedule): The schedule to check legality for
            with_ast (bool, optional): If true, the AST will be returned. Defaults to False.

        Returns:
            str: The code to check legality of the schedule
        """
        assert schedule.tiramisu_program
        assert schedule.tiramisu_program.cpp_code
        assert schedule.tree

        # Add code to the original file to get legality result
        legality_check_lines = """
    prepare_schedules_for_legality_checks(true);
    perform_full_dependency_analysis();
    bool is_legal=true;

"""
        for optim in schedule.optims_list:
            legality_check_lines += "    " + optim.legality_check_string

        legality_check_lines += """
    prepare_schedules_for_legality_checks(true);
    is_legal &= check_legality_of_function();
    std::cout << is_legal << std::endl;
"""

        if with_ast:
            legality_check_lines += """
    auto fct = tiramisu::global::get_implicit_function();

    fct->gen_time_space_domain();
    fct->gen_isl_ast();
    fct->print_isl_ast_representation();
"""

        cpp_code = schedule.tiramisu_program.cpp_code.replace(
            schedule.tiramisu_program.code_gen_line, legality_check_lines
        )
        return cpp_code

    @classmethod
    def compile_annotations(cls, tiramisu_program: TiramisuProgram):
        """Compile and return the annotations of the program.

        Args:
            tiramisu_program (TiramisuProgram): The program to get the annotations for

        Returns:
            str: The annotations of the program
        """
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")

        if not tiramisu_program.cpp_code:
            raise ValueError("Tiramisu program not initialized")

        output_path = os.path.join(
            BaseConfig.base_config.workspace,
            f"{tiramisu_program.temp_files_identifier}_annotations",
        )
        # Add code to the original file to get json annotations

        get_json_lines = """
            auto ast = tiramisu::auto_scheduler::syntax_tree(tiramisu::global::get_implicit_function(), {});
            std::string program_json = tiramisu::auto_scheduler::evaluate_by_learning_model::get_program_json(ast);
            std::cout << program_json;
            """  # noqa: E501

        cpp_code = tiramisu_program.cpp_code.replace(
            tiramisu_program.code_gen_line, get_json_lines
        )
        return cls.run_cpp_code(cpp_code=cpp_code, output_path=output_path)

    @classmethod
    def compile_isl_ast_tree(
        cls,
        tiramisu_program: TiramisuProgram,
        schedule: Schedule | None = None,
    ):
        """Compile and return the isl ast of the program.

        Args:
            tiramisu_program (TiramisuProgram): The program to get the isl ast for
            schedule (Schedule, optional): The schedule to get the isl ast for. Defaults to None.

        Returns:
            str: The isl ast of the program
        """
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")

        if not tiramisu_program.cpp_code:
            raise ValueError("Tiramisu program not initialized")

        output_path = os.path.join(
            BaseConfig.base_config.workspace,
            f"{tiramisu_program.temp_files_identifier}_isl_ast",
        )
        get_isl_ast_lines = ""
        if schedule:
            for optim in schedule.optims_list:
                # if optim.is_parallelization():
                get_isl_ast_lines += "    " + optim.tiramisu_optim_str

        get_isl_ast_lines += """
    auto fct = tiramisu::global::get_implicit_function();

    fct->gen_time_space_domain();
    fct->gen_isl_ast();
    fct->print_isl_ast_representation();
"""

        cpp_code = tiramisu_program.cpp_code.replace(
            tiramisu_program.code_gen_line, get_isl_ast_lines
        )
        return cls.run_cpp_code(cpp_code=cpp_code, output_path=output_path)

    @classmethod
    def run_cpp_code(cls, cpp_code: str, output_path: str) -> str:
        """Compile and run the generated code.

        Args:
            cpp_code (str): The code to compile and run
            output_path (str): The path of the output file

        Returns:
            str: The output of the code
        """
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")

        env_vars = CompilingService.get_env_vars()
        shell_script = [
            # Compile intermidiate tiramisu file
            "$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -fopenmp -std=c++17 -O0 -o {}.o -c -x c++ -".format(
                output_path
            ),
            # Link generated file with executer
            "$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -fopenmp -std=c++17 -O0 {}.o -o {}.out -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl".format(
                output_path, output_path
            ),
            # Run the program
            "{}.out".format(output_path),
            # Clean generated files
            "rm {}.out {}.o".format(output_path, output_path),
        ]
        try:
            compiler = subprocess.run(
                ["\n".join(env_vars + shell_script)],
                input=cpp_code,
                capture_output=True,
                text=True,
                shell=True,
                check=True,
            )

            if compiler.stdout:
                return compiler.stdout
            else:
                print(compiler.stderr)
                raise Exception("Compiler returned no output")

        except subprocess.CalledProcessError as e:
            logger.error(f"Process terminated with error code: {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            logger.error(env_vars + shell_script)
            raise e
        except Exception as e:
            raise e

    @classmethod
    def call_skewing_solver(
        cls,
        schedule: Schedule,
        loop_levels: List[int],
        comps_skewed_loops: List[str],
    ):
        """Call the skewing solver to generate the skewing code.

        Args:
            schedule (Schedule): The schedule to skew
            loop_levels (List[int]): The levels of the loops to skew
            comps_skewed_loops (List[str]): The components of the loops to skew

        Returns:
            Tuple[int, int]: The factors to skew the loops by
        """
        assert schedule.tiramisu_program

        if BaseConfig.base_config is None:
            raise Exception("The base config is not loaded yet")
        legality_cpp_code = cls.get_legality_code(schedule)
        to_replace = re.findall(
            r"std::cout << is_legal << std::endl;", legality_cpp_code
        )[0]
        header = """
        function * fct = tiramisu::global::get_implicit_function();\n"""
        legality_cpp_code = legality_cpp_code.replace(
            "is_legal &= check_legality_of_function();", ""
        )
        legality_cpp_code = legality_cpp_code.replace("bool is_legal=true;", "")
        legality_cpp_code = re.sub(
            r"is_legal &= loop_parallelization_is_legal.*\n",
            "",
            legality_cpp_code,
        )
        legality_cpp_code = re.sub(
            r"is_legal &= loop_unrolling_is_legal.*\n", "", legality_cpp_code
        )

        solver_lines = (
            header
            + "\n\tauto auto_skewing_result = fct->skewing_local_solver({"
            + ", ".join([f"&{comp}" for comp in comps_skewed_loops])
            + "}"
            + ",{},{},1);\n".format(*loop_levels)
        )

        solver_lines += """
        std::vector<std::pair<int,int>> outer1, outer2,outer3;
        tie( outer1,  outer2,  outer3 )= auto_skewing_result;
        if (outer1.size()>0){
            std::cout << outer1.front().first;
            std::cout << ",";
            std::cout << outer1.front().second;
            std::cout << ",";
        }else {
            std::cout << "None,None,";
        }
        if(outer2.size()>0){
            std::cout << outer2.front().first;
            std::cout << ",";
            std::cout << outer2.front().second;
            std::cout << ",";
        }else {
            std::cout << "None,None,";
        }
        if(outer3.size()>0){
            std::cout << outer3.front().first;
            std::cout << ",";
            std::cout << outer3.front().second;
        }else {
            std::cout << "None,None";
        }

            """

        solver_code = legality_cpp_code.replace(to_replace, solver_lines)
        logger.debug("Skewing Solver Code:\n" + solver_code)
        output_path = os.path.join(
            BaseConfig.base_config.workspace,
            f"{schedule.tiramisu_program.temp_files_identifier}_skewing_solver",
        )

        result_str = cls.run_cpp_code(cpp_code=solver_code, output_path=output_path)
        result_str = result_str.split(",")

        # Skewing Solver returns 3 solutions in form of tuples:
        # - the first tuple is for outer parallelism.
        # - second is for inner parallelism , and last one is for locality.

        if result_str[0] != "None":
            # Means we have a solution for outer parallelism
            fac1 = int(result_str[0])
            fac2 = int(result_str[1])
            return fac1, fac2
        if result_str[2] != "None":
            # Means we have a solution for inner parallelism
            fac1 = int(result_str[2])
            fac2 = int(result_str[3])
            return fac1, fac2
        else:
            return None

    @classmethod
    def get_schedule_code(
        cls,
        tiramisu_program: TiramisuProgram,
        optims_list: List[TiramisuAction],
    ):
        """Generate the schedule code to apply the optimizations on the program.

        Args:
            tiramisu_program (TiramisuProgram): The program to apply the optimizations on
            optims_list (List[TiramisuAction]): The optimizations to apply

        Returns:
            str: The code to apply the optimizations on the program
        """
        if not tiramisu_program.cpp_code:
            raise ValueError("The program is not loaded yet")
        # Add code to the original file to get the schedule code
        schedule_code = ""
        for optim in optims_list:
            schedule_code += optim.tiramisu_optim_str + "\n"

        # Add code gen line to the schedule code
        schedule_code += "\n    " + tiramisu_program.code_gen_line + "\n"
        cpp_code = tiramisu_program.cpp_code.replace(
            tiramisu_program.code_gen_line, schedule_code
        )
        cpp_code = cpp_code.replace(
            f"// {tiramisu_program.wrapper_str}", tiramisu_program.wrapper_str
        )
        return cpp_code

    @classmethod
    def write_to_disk(cls, cpp_code: str, output_path: str, extension: str = ".cpp"):
        """Write the code to a file.

        Args:
            cpp_code (str): The code to write
            output_path (str): The path of the output file
            extension (str, optional): The extension of the output file. Defaults to ".cpp".
        """
        with open(output_path + extension, "w") as f:
            f.write(cpp_code)

    @classmethod
    def get_cpu_exec_times(  # noqa: C901
        cls,
        tiramisu_program: TiramisuProgram,
        optims_list: List[TiramisuAction],
        min_runs: int = 1,
        max_runs: int | None = None,
        time_budget: float | None = None,
        delete_files: bool = True,
    ) -> List[float]:
        """Return the execution times of the program.
        Parameters
        ----------
        `tiramisu_program` : TiramisuProgram
            The program to get the execution times for
        `optims_list` : List[TiramisuAction]
             The optimizations to apply
        `min_runs` : int
            Defines the minimal number of times the schedule will be executed.
            The parameter guarantees that the execution time will be measured
            at least min_run times regardless of the set time budget.
        `time_budget` : float or None
            Defines the duration (in milliseconds) allocated for execution
            time measurements.
            Once the time_budget is consumed, no more runs will be performed
            and ongoing execution will be aborted.
            Measurements completed within the time_budget (if any) will be saved.
            The min_runs first measurements are not abortable and will execute
            till completion even if time_budget is fully consumed.
        `max_runs` : int or None
            Defines the maximal number of times the schedule will be executed.
            This allows to stop taking measurements before time_budget is
            fully consumed. max_runs is only taken into consideration if
            time_budget is defined. If time_budget is defined and max_runs is
            None, max_run will be set to infinity.
        `delete_files` : bool
            Defines whether to delete the temporary generated files or not.

        Returns
        -------
            List[float]: The execution times of the program
        """
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")
        if (
            not tiramisu_program.name
            or not tiramisu_program.cpp_code
            or not tiramisu_program.wrappers
        ):
            raise ValueError("The program is not loaded yet")
        cpp_code = cls.get_schedule_code(tiramisu_program, optims_list)
        # Write the code to a file
        output_path = os.path.join(
            BaseConfig.base_config.workspace, tiramisu_program.temp_files_identifier
        )

        cls.write_to_disk(cpp_code, output_path + "_schedule")

        if tiramisu_program.wrapper_obj:
            # write the object file to disk
            with open(output_path + "_wrapper", "wb") as f:
                f.write(tiramisu_program.wrapper_obj)
            # write the wrapper header file needed by the schedule file
            cls.write_to_disk(
                tiramisu_program.wrappers["h"], output_path + "_wrapper", ".h"
            )
            # give it execution rights to be able to run it
            subprocess.check_output(["chmod", "+x", output_path + "_wrapper"])
        else:
            # write the wrappers
            cls.write_to_disk(
                tiramisu_program.wrappers["cpp"], output_path + "_wrapper"
            )
            cls.write_to_disk(
                tiramisu_program.wrappers["h"], output_path + "_wrapper", ".h"
            )

        env_vars = CompilingService.get_env_vars()

        results: list[float] = []
        shell_script = [
            # Compile intermidiate tiramisu file
            f"cd {BaseConfig.base_config.workspace}",
            f"$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -fopenmp -std=c++17 -O0 -o {tiramisu_program.temp_files_identifier}.o -c {tiramisu_program.temp_files_identifier}_schedule.cpp",
            # Link generated file with executer
            f"$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -fopenmp -std=c++17 -O0 {tiramisu_program.temp_files_identifier}.o -o {tiramisu_program.temp_files_identifier}.out -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl",
            # Run the program
            f"./{tiramisu_program.temp_files_identifier}.out",
            f"$CXX -shared -o {tiramisu_program.temp_files_identifier}.so {tiramisu_program.temp_files_identifier}.o",  # noqa: E501
        ]
        if not tiramisu_program.wrapper_obj:
            shell_script += [
                # compile the wrapper
                f"$CXX -std=c++17 -fno-rtti -o {tiramisu_program.temp_files_identifier}_wrapper -ltiramisu -lHalide -ldl -lpthread -fopenmp -lm {tiramisu_program.temp_files_identifier}_wrapper.cpp ./{tiramisu_program.temp_files_identifier}.so -ltiramisu -lHalide -ldl -lpthread -fopenmp -lm -lisl"
            ]
        try:
            # run the compilation of the generator and wrapper
            compiler = subprocess.run(
                [" && ".join(env_vars + shell_script)],
                capture_output=True,
                text=True,
                shell=True,
                check=True,
            )

            halide_repr = compiler.stdout
            logger.debug(f"Generated Halide code:\n{halide_repr}")

            # if a minimal number if executions is set, perform them without a timeout
            # if min_runs > 0:
            # run the wrapper and get the execution time
            compiler = subprocess.run(
                [
                    " && ".join(
                        env_vars
                        + CompilingService.get_n_runs_script(
                            min_runs=min_runs,
                            max_runs=max_runs,
                            time_budget=time_budget,
                            tiramisu_program=tiramisu_program
                        )
                    )
                ],
                capture_output=True,
                text=True,
                shell=True,
                check=True, # This ensures CRASHES (segfaults) are raised as Exceptions
            )

            # If we reached this line, the C++ wrapper exited cleanly (Exit Code 0)
            if compiler.stdout:
                results += [float(x) for x in compiler.stdout.split()]
            else:
                # Exit Code 0, but no output.
                # This happens if Time Budget < 1st Run Duration.
                if min_runs == 0:
                    # This is a valid outcome: The budget killed it before the first run finished.
                    logger.info(f"Target {tiramisu_program.name}: Time budget exhausted before 1st run. Returning empty.")
                    # results remains empty list
                else:
                    # If min_runs > 0, we were GUARANTEED at least one print. 
                    # If empty -> Logic Error or Flush failure (Unlikely with current cpp)
                    logger.error("No output from schedule execution despite min_runs > 0")
                    logger.error(compiler.stderr)
                    logger.error(compiler.stdout)
                    logger.error(
                        f"The following schedule execution returned no results despite min_runs > 0: {tiramisu_program.name}, schedule: {optims_list} \n\n {cpp_code}\n\n"  # noqa: E501
                    )
                    raise ScheduleExecutionError("Schedule execution returned 0 results, expected > 0")

            if delete_files:
                CompilingService.delete_temporary_files(
                    tiramisu_program=tiramisu_program
                )
            return results

        except subprocess.CalledProcessError as e:
            logger.error(f"Process terminated with error code: {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            logger.error(f"Output: {e.stdout}")
            raise ScheduleExecutionError(
                f"Schedule execution crashed: function: {tiramisu_program.name}, schedule: {optims_list}"  # noqa: E501
            )

    @classmethod
    def get_n_runs_script(
        cls,
        tiramisu_program: TiramisuProgram,
        min_runs: int = 1,
        max_runs: int |  None = None,
        time_budget: float |  None = None,

    ):
        """Get the script to run the program n times."""
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")

        env_vars = CompilingService.get_env_vars()
        return env_vars + [
            # cd to the workspace
            f"cd {BaseConfig.base_config.workspace}",
            #  set the env variables
            f"export MIN_RUNS={min_runs}",
            f'export MAX_RUNS={max_runs if max_runs else "inf"}',
            f'export TIME_BUDGET={time_budget if time_budget else "-1"}',
            # run the wrapper
            f"./{tiramisu_program.temp_files_identifier}_wrapper"
        ]

    @classmethod
    def delete_temporary_files(cls, tiramisu_program: TiramisuProgram):
        """Delete files temporary and intermediate files"""
        assert BaseConfig.base_config, "BaseConfig not initialized"
        subprocess.run(
            [
                # cd to the workspace and clean generated files
                f"cd {BaseConfig.base_config.workspace} && rm {tiramisu_program.temp_files_identifier}*"
            ],
            capture_output=True,
            text=True,
            shell=True,
            check=False,
        )

    @classmethod
    def get_env_vars(cls):
        """Get the environment variables."""
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")

        env_vars = [
            f"export {key}={value}"
            for key, value in BaseConfig.base_config.env_vars.items()
        ]

        libs = ":".join(BaseConfig.base_config.dependencies.libs)
        env_vars += [
            f"export LD_LIBRARY_PATH={libs}",
            f"export LIBRARY_PATH={libs}",
            f"export CPATH={':'.join(BaseConfig.base_config.dependencies.includes)}",
        ]

        return env_vars


class ScheduleExecutionError(Exception):
    """Raised when the execution of the schedule crashes."""

    pass
