import json
import random
import re
from typing import Any

from tiralib.tiramisu.compiling_service import CompilingService
from tiralib.tiramisu.function_server import FunctionServer
from tiralib.tiramisu.tiramisu_tree import TiramisuTree


class TiramisuProgram:
    """This class represents a tiramisu function. It contains all the neccessary
    information about the function to be able to generate the code for it.

    Attributes
    ----------
    `name`: str
        The name of the function
    `cpp_code`: str
        The code of the function
    `tree`: TiramisuTree
        The tree of the function
    `IO_buffer_names`: list[str]
        The names of the input/output buffers
    `buffer_sizes`: list[list[str]]
        The sizes of the input/output buffers
    `annotations`: Dict
        The annotations of the function
    `isl_ast_string`: str
        The isl ast of the function
    `wrapper_obj`: bytes
        The wrapper object of the function
    `server`: FunctionServer
        TiralibCpp server object
    """

    def __init__(self: "TiramisuProgram"):
        self.name: str
        self.cpp_code: str
        self.tree: TiramisuTree
        self.IO_buffer_names: list[str]
        self.buffer_sizes: list[list[str]]
        self.annotations: dict[str, Any] | None = None
        self.isl_ast_string: str | None = None
        self.wrapper_obj: bytes | None = None
        self.server: FunctionServer | None = None

    @property
    def wrappers(self):
        buffers_init_lines = ""
        for i, buffer_name in enumerate(self.IO_buffer_names):
            buffers_init_lines += f"""
    double *c_{buffer_name} = (double*)malloc({"*".join(self.buffer_sizes[i][::-1])}* sizeof(double));
    parallel_init_buffer(c_{buffer_name}, {"*".join(self.buffer_sizes[i][::-1])}, (double){str(random.randint(1, 10))});
    Halide::Buffer<double> {buffer_name}(c_{buffer_name}, {",".join(self.buffer_sizes[i][::-1])});
    """  # noqa: E501

        wrapper_cpp_code = wrapper_cpp_template.replace("$func_name$", self.name)
        wrapper_cpp_code = wrapper_cpp_code.replace(
            "$buffers_init$", buffers_init_lines
        )
        wrapper_cpp_code = wrapper_cpp_code.replace(
            "$func_params$",
            ",".join([name + ".raw_buffer()" for name in self.IO_buffer_names]),
        )

        wrapper_h_code = wrapper_h_template.replace("$func_name$", self.name)
        wrapper_h_code = wrapper_h_code.replace(
            "$func_params$",
            ",".join(["halide_buffer_t *" + name for name in self.IO_buffer_names]),
        )

        return {
            "cpp": wrapper_cpp_code,
            "h": wrapper_h_code,
        }

    @classmethod
    def from_annotations(
        cls,
        annotations: dict[str, Any],
        cpp_code: str,
        load_tree: bool = True,
        wrapper_obj: bytes | None = None,
    ) -> "TiramisuProgram":
        # Initiate an instante of the TiramisuProgram class
        tiramisu_prog = cls()
        tiramisu_prog.cpp_code = cpp_code
        tiramisu_prog.annotations = annotations
        tiramisu_prog.load_code_lines()

        if wrapper_obj:
            tiramisu_prog.wrapper_obj = wrapper_obj

        if load_tree:
            tiramisu_prog.tree = TiramisuTree.from_annotations(
                tiramisu_prog.annotations
            )
        return tiramisu_prog

    @classmethod
    def from_file(
        cls,
        file_path: str,
        load_annotations: bool = False,
        load_isl_ast: bool = False,
        load_tree: bool = False,
    ) -> "TiramisuProgram":
        """This function loads a tiramisu function from its cpp file and its
        wrapper files.

        Parameters
        ----------
        `file_path`: str
            The path to the cpp file of the tiramisu function
        `load_annotations`: bool
            A flag to indicate if the annotations should be loaded or not
        `load_isl_ast`: bool
            A flag to indicate if the isl ast should be loaded or not
        `load_tree`: bool
            A flag to indicate if the tree should be constructed or not

        Returns
        -------
        `tiramisu_prog`: TiramisuProgram
            An instance of the TiramisuProgram class
        """
        # Initiate an instante of the TiramisuProgram class

        tiramisu_prog = cls()
        with open(file_path, "r") as f:
            tiramisu_prog.cpp_code = f.read()
        tiramisu_prog.load_code_lines()

        if load_annotations:
            tiramisu_prog.annotations = json.loads(
                CompilingService.compile_annotations(tiramisu_prog)
            )
        elif load_isl_ast:
            tiramisu_prog.isl_ast_string = CompilingService.compile_isl_ast_tree(
                tiramisu_prog
            )

        if load_tree:
            if tiramisu_prog.annotations:
                assert tiramisu_prog.annotations is not None
                tiramisu_prog.tree = TiramisuTree.from_annotations(
                    tiramisu_prog.annotations
                )
            elif tiramisu_prog.isl_ast_string:
                tiramisu_prog.tree = TiramisuTree.from_isl_ast_string_list(
                    tiramisu_prog.isl_ast_string.split("\n")
                )
            else:
                raise Exception(
                    "You should load either the annotations or the isl ast\
                    string to load the tree"
                )

        # After taking the neccessary fields return the instance
        return tiramisu_prog

    @classmethod
    def init_server(
        cls,
        cpp_code: str,
        load_annotations: bool = False,
        load_isl_ast: bool = False,
        load_tree: bool = False,
        reuse_server: bool = False,
    ) -> "TiramisuProgram":
        # Initiate an instante of the TiramisuProgram class
        tiramisu_prog = cls()
        tiramisu_prog.cpp_code = cpp_code
        tiramisu_prog.load_code_lines()
        tiramisu_prog.server = FunctionServer(tiramisu_prog, reuse_server=reuse_server)

        if load_annotations:
            annotations_str = tiramisu_prog.server.get_annotations()
            tiramisu_prog.annotations = json.loads(annotations_str)
        elif load_isl_ast:
            result = tiramisu_prog.server.run()
            tiramisu_prog.isl_ast_string = result.isl_ast

        if load_tree:
            if tiramisu_prog.annotations:
                assert tiramisu_prog.annotations is not None
                tiramisu_prog.tree = TiramisuTree.from_annotations(
                    tiramisu_prog.annotations
                )
            elif tiramisu_prog.isl_ast_string:
                tiramisu_prog.tree = TiramisuTree.from_isl_ast_string_list(
                    tiramisu_prog.isl_ast_string.split("\n")
                )
            else:
                raise Exception(
                    "You should load either the annotations or the isl ast \
                    string to load the tree"
                )

        # After taking the neccessary fields return the instance
        return tiramisu_prog

    def load_code_lines(self):
        """This function loads the file code , it is necessary to generate
        legality check code and annotations
        """
        self.name = re.findall(r"tiramisu::init\(\"(\w+)\"\);", self.cpp_code)[0]
        self.body = re.findall(
            r"int main\([\w\s,*]+\)\s*\{([\W\w\s]*)tiramisu::codegen",
            self.cpp_code,
        )[0]
        # Remove the wrapper include from the original string
        self.wrapper_str = f'#include "{self.name}_wrapper.h"'
        self.cpp_code = self.cpp_code.replace(
            self.wrapper_str, f"// {self.wrapper_str}"
        )
        self.code_gen_line: str = re.findall(r"tiramisu::codegen\({.+;", self.cpp_code)[
            0
        ]
        buffers_vect = re.findall(r"{(.+)}", self.code_gen_line)[0]
        self.IO_buffer_names = re.findall(r"\w+", buffers_vect)
        self.buffer_sizes = []
        for buf_name in self.IO_buffer_names:
            sizes_vect = re.findall(r"buffer " + buf_name + ".*{(.*)}", self.cpp_code)[
                0
            ]
            self.buffer_sizes.append(re.findall(r"\d+", sizes_vect))

    def __str__(self) -> str:
        return f"TiramisuProgram(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()


wrapper_cpp_template = """#include "Halide.h"
#include "$func_name$_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;

int main(int, char **argv){

$buffers_init$

    //halide_set_num_threads(48);

    int nb_execs = get_nb_exec();

    double duration;

    for (int i = 0; i < nb_execs; ++i) {
        auto begin = std::chrono::high_resolution_clock::now();
        $func_name$($func_params$);
        auto end = std::chrono::high_resolution_clock::now();

        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000;
        std::cout << duration << " " << std::flush;

    }
    std::cout << std::endl;
    return 0;
}"""  # noqa: E501
wrapper_h_template = """#include <tiramisu/utils.h>
#include <sys/time.h>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>
#include <limits>

#define NB_THREAD_INIT 48
struct args {
    double *buf;
    unsigned long long int part_start;
    unsigned long long int part_end;
    double value;
};

void *init_part(void *params)
{
   double *buffer = ((struct args*) params)->buf;
   unsigned long long int start = ((struct args*) params)->part_start;
   unsigned long long int end = ((struct args*) params)->part_end;
   double val = ((struct args*) params)->value;
   for (unsigned long long int k = start; k < end; k++){
       buffer[k]=val;
   }
   pthread_exit(NULL);
}

void parallel_init_buffer(double* buf, unsigned long long int size, double value){
    pthread_t threads[NB_THREAD_INIT];
    struct args params[NB_THREAD_INIT];
    for (int i = 0; i < NB_THREAD_INIT; i++) {
        unsigned long long int start = i*size/NB_THREAD_INIT;
        unsigned long long int end = std::min((i+1)*size/NB_THREAD_INIT, size);
        params[i] = (struct args){buf, start, end, value};
        pthread_create(&threads[i], NULL, init_part, (void*)&(params[i]));
    }
    for (int i = 0; i < NB_THREAD_INIT; i++)
        pthread_join(threads[i], NULL);
    return;
}
#ifdef __cplusplus
extern "C" {
#endif
int $func_name$($func_params$);
#ifdef __cplusplus
}  // extern "C"
#endif


int get_beam_size(){
    if (std::getenv("BEAM_SIZE")!=NULL)
        return std::stoi(std::getenv("BEAM_SIZE"));
    else{
        std::cerr<<"error: Environment Variable BEAM_SIZE not declared"<<std::endl;
        exit(1);
    }
}

int get_max_depth(){
    if (std::getenv("MAX_DEPTH")!=NULL)
        return std::stoi(std::getenv("MAX_DEPTH"));
    else{
        std::cerr<<"error: Environment Variable MAX_DEPTH not declared"<<std::endl;
        exit(1);
    }
}

void declare_memory_usage(){
    setenv("MEM_SIZE", std::to_string((double)(256*192+320*256+320*192)*8/1024/1024).c_str(), true); // This value was set by the Code Generator
}

int get_nb_exec() {
    const char* env_var = std::getenv("NB_EXEC");
    if (env_var != nullptr) {
        std::string env_value(env_var);
        if (env_value == "inf") {
            return std::numeric_limits<int>::max(); // Use maximum int value to represent +infinity
        } else {
            return std::stoi(env_value);
        }
    } else {
        return 30; // Default value
    }
}
"""  # noqa: E501
