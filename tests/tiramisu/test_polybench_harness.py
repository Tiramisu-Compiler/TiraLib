import subprocess
from pathlib import Path

import pytest

from tiralib.config import BaseConfig
from tiralib.tiramisu.compiling_service import CompilingService
from tiralib.tiramisu.harness import (
    DefaultHarness,
    PolybenchHarness,
    polybench_normalized_time,
)
from tiralib.tiramisu.schedule import Schedule
from tiralib.tiramisu.tiramisu_program import TiramisuProgram

GEMM_GENERATOR = "tests/data/polybench/function_gemm_MINI_generator.cpp"
GEMM_SIDECAR = "tests/data/polybench/function_gemm_MINI_init.h"


# ---------------------------------------------------------------------------
# Harness selection
# ---------------------------------------------------------------------------


def test_polybench_harness_autodetected():
    BaseConfig.init()
    program = TiramisuProgram.from_file(GEMM_GENERATOR)
    assert isinstance(program.harness, PolybenchHarness)
    assert program.IO_buffer_names == ["b_A", "b_B", "b_C"]
    assert program.buffer_types == ["p_float64", "p_float64", "p_float64"]


def test_explicit_harness_overrides_autodetection():
    BaseConfig.init()
    program = TiramisuProgram.from_file(GEMM_GENERATOR, harness=DefaultHarness())
    assert isinstance(program.harness, DefaultHarness)
    assert "parallel_init_buffer" in program.wrappers["cpp"]


def test_no_sidecar_uses_default_harness():
    BaseConfig.init()
    # tests/data/function_2mm_MINI.cpp has no *_init.h next to it
    program = TiramisuProgram.from_file("tests/data/function_2mm_MINI.cpp")
    assert isinstance(program.harness, DefaultHarness)


def test_missing_sidecar_raises():
    with pytest.raises(FileNotFoundError):
        PolybenchHarness("/nonexistent/path_init.h")


def test_invalid_sidecar_raises(tmp_path: Path):
    bad = tmp_path / "function_foo_MINI_init.h"
    bad.write_text("// not a sidecar\n")
    with pytest.raises(ValueError):
        PolybenchHarness(bad)


def test_from_code():
    BaseConfig.init()
    code = Path(GEMM_SIDECAR).read_text()
    harness = PolybenchHarness.from_code(code, cache_size_kb=2048)
    assert harness.sidecar_code == code
    assert harness.init_sidecar_path is None
    assert harness.cache_size_kb == 2048
    program = TiramisuProgram.from_file(GEMM_GENERATOR, harness=harness)
    cpp = program.wrappers["cpp"]
    assert "tiralib_polybench_init_arrays" in cpp
    assert "#define POLYBENCH_CACHE_SIZE_KB 2048" in cpp


def test_from_code_invalid_raises():
    with pytest.raises(ValueError):
        PolybenchHarness.from_code("// not a sidecar")
    with pytest.raises(ValueError):
        # exactly one source must be provided
        PolybenchHarness()
    with pytest.raises(ValueError):
        PolybenchHarness(GEMM_SIDECAR, init_sidecar_code="// both")


# ---------------------------------------------------------------------------
# Wrapper generation
# ---------------------------------------------------------------------------


def test_polybench_wrapper_content():
    BaseConfig.init()
    program = TiramisuProgram.from_file(GEMM_GENERATOR)
    wrappers = program.wrappers

    cpp = wrappers["cpp"]
    # PolyBench measurement methodology, inlined verbatim
    assert "#define POLYBENCH_TIME 1" in cpp
    assert "polybench_timer_start" in cpp
    assert "polybench_flush_cache" in cpp
    assert "polybench_alloc_data" in cpp
    # per-benchmark sidecar
    assert "tiralib_polybench_init_arrays" in cpp
    assert "tiralib_polybench_dump_arrays" in cpp
    # fresh-process-per-measurement protocol
    assert "--tiralib-single-run" in cpp
    # no random warm-init of the default harness
    assert "parallel_init_buffer" not in cpp

    header = wrappers["h"]
    assert "function_gemm_MINI" in header
    assert 'extern "C"' in header


def test_polybench_openmp_runtime_default():
    """The frozen measurement policy: OpenMP runtime, schedule(static,1),
    bound threads — baked into the generated wrapper."""
    BaseConfig.init()
    program = TiramisuProgram.from_file(GEMM_GENERATOR)
    harness = program.harness
    assert harness.runtime == "openmp"
    assert harness.omp_schedule == "static,1"
    cpp = program.wrappers["cpp"]
    assert "tiralib_install_openmp_runtime();" in cpp
    assert "#pragma omp parallel for schedule(static,1)" in cpp
    assert 'setenv("OMP_PROC_BIND", "close", 1)' in cpp
    assert 'setenv("OMP_PLACES", "cores", 1)' in cpp
    assert "sched_setaffinity" in cpp  # orchestrator affinity reset


def test_polybench_halide_runtime_option():
    """runtime='halide' keeps Halide's pool and scrubs the OpenMP binding
    vars that would otherwise serialize it."""
    BaseConfig.init()
    harness = PolybenchHarness(GEMM_SIDECAR, runtime="halide")
    program = TiramisuProgram.from_file(GEMM_GENERATOR, harness=harness)
    cpp = program.wrappers["cpp"]
    assert "tiralib_install_openmp_runtime" not in cpp
    assert 'unsetenv("OMP_PROC_BIND");' in cpp
    assert 'unsetenv("GOMP_CPU_AFFINITY");' in cpp
    assert "sched_setaffinity" in cpp


def test_polybench_runtime_options_validated():
    with pytest.raises(ValueError):
        PolybenchHarness(GEMM_SIDECAR, runtime="tbb")
    with pytest.raises(ValueError):
        PolybenchHarness(GEMM_SIDECAR, omp_schedule="static,x")
    with pytest.raises(ValueError):
        PolybenchHarness(GEMM_SIDECAR, omp_schedule="auto")
    with pytest.raises(ValueError):
        PolybenchHarness(GEMM_SIDECAR, omp_proc_bind="master")
    # valid variants construct fine
    PolybenchHarness(GEMM_SIDECAR, omp_schedule="dynamic")
    PolybenchHarness(GEMM_SIDECAR, omp_schedule="guided,4", omp_proc_bind="spread")


def test_polybench_wrapper_options():
    BaseConfig.init()
    harness = PolybenchHarness(GEMM_SIDECAR, cache_size_kb=1024, flush_cache=False)
    program = TiramisuProgram.from_file(GEMM_GENERATOR, harness=harness)
    cpp = program.wrappers["cpp"]
    assert "#define POLYBENCH_CACHE_SIZE_KB 1024" in cpp
    assert "#define POLYBENCH_NO_FLUSH_CACHE 1" in cpp


def test_polybench_wrapper_typed_buffers():
    BaseConfig.init()
    generator = Path(
        "/workspace/polybench_tiramisu/function_floyd_warshall_MINI/"
        "function_floyd_warshall_MINI_generator.cpp"
    )
    if not generator.exists():
        pytest.skip("polybench_tiramisu repo not available")
    program = TiramisuProgram.from_file(str(generator))
    assert program.buffer_types == ["p_int32"]
    cpp = program.wrappers["cpp"]
    assert "int *c_b_paths = (int*) polybench_alloc_data" in cpp
    assert "Halide::Buffer<int> b_paths" in cpp


# ---------------------------------------------------------------------------
# time_benchmark.sh-style normalization
# ---------------------------------------------------------------------------


def test_polybench_normalized_time():
    # 5 measurements: drop min (1) and max (5), mean of [2, 3, 4] = 3
    mean, variance = polybench_normalized_time([3.0, 1.0, 5.0, 2.0, 4.0])
    assert mean == pytest.approx(3.0)
    assert variance == pytest.approx(100.0 / 3.0)


def test_polybench_normalized_time_requires_3():
    with pytest.raises(ValueError):
        polybench_normalized_time([1.0, 2.0])


# ---------------------------------------------------------------------------
# Execution (integration: compiles and runs the wrapper)
# ---------------------------------------------------------------------------


def test_polybench_execute():
    BaseConfig.init()
    program = TiramisuProgram.from_file(
        GEMM_GENERATOR, load_annotations=True, load_tree=True
    )
    assert isinstance(program.harness, PolybenchHarness)
    schedule = Schedule(program)
    results = schedule.execute(min_runs=3)
    assert results is not None
    assert len(results) == 3
    assert all(t > 0 for t in results)


def test_polybench_execute_max_runs_with_budget():
    BaseConfig.init()
    program = TiramisuProgram.from_file(
        GEMM_GENERATOR, load_annotations=True, load_tree=True
    )
    schedule = Schedule(program)
    exec_time = schedule.execute(1)[0]
    # generous budget: max_runs is the binding limit
    results = schedule.execute(
        min_runs=2, max_runs=4, time_budget=10000 * exec_time + 1000
    )
    assert len(results) == 4


def test_polybench_server_backend():
    """The self-contained polybench wrapper also works when compiled and
    invoked by the TiraLibCpp server backend."""
    BaseConfig.init()
    cpp_code = Path(GEMM_GENERATOR).read_text()
    program = TiramisuProgram.init_server(
        cpp_code=cpp_code,
        load_isl_ast=True,
        load_tree=True,
        harness=PolybenchHarness(GEMM_SIDECAR),
    )
    assert isinstance(program.harness, PolybenchHarness)
    schedule = Schedule(program)
    results = schedule.execute(min_runs=2)
    assert len(results) == 2
    assert all(t > 0 for t in results)
    program.server.delete_temporary_files()


def test_polybench_dump_arrays():
    """TIRALIB_DUMP_ARRAYS=1 dumps live-out arrays in PolyBench format."""
    BaseConfig.init()
    program = TiramisuProgram.from_file(
        GEMM_GENERATOR, load_annotations=True, load_tree=True
    )
    schedule = Schedule(program)
    schedule.execute(min_runs=1, delete_files=False)

    env_vars = CompilingService.get_env_vars()
    command = " && ".join(
        env_vars
        + [
            f"cd {BaseConfig.base_config.workspace}",
            "TIRALIB_DUMP_ARRAYS=1 MIN_RUNS=1 "
            f"./{program.temp_files_identifier}_wrapper",
        ]
    )
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, check=True
    )
    assert "==BEGIN DUMP_ARRAYS==" in result.stderr
    assert "begin dump: C" in result.stderr
    assert "==END   DUMP_ARRAYS==" in result.stderr
    # stdout still carries exactly one measurement
    assert len(result.stdout.split()) == 1

    CompilingService.delete_temporary_files(program)
