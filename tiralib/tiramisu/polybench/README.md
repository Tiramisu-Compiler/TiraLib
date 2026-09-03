# Vendored PolyBench/C 4.2.1 harness utilities

`polybench.h` and `polybench.c` are verbatim, unmodified copies of
`utilities/polybench.h` and `utilities/polybench.c` from
**PolyBench/C 4.2.1** (stamped May 10, 2016, by Louis-Noël Pouchet and
Tomofumi Yuki, <http://polybench.sourceforge.net>).

They are used by `tiralib.tiramisu.harness.PolybenchHarness`, which inlines
their contents into the generated measurement wrapper so that TiraLib times
PolyBench-derived Tiramisu programs with the *exact* PolyBench measurement
methodology (page-aligned allocation, cache flush before timing, one kernel
invocation per fresh process, `gettimeofday`-based timer).

Do not edit these files. If you need different harness behavior, configure
`PolybenchHarness` (e.g. `cache_size_kb`, `flush_cache`) instead — options are
translated into the standard PolyBench `#define`s
(`POLYBENCH_CACHE_SIZE_KB`, `POLYBENCH_NO_FLUSH_CACHE`, ...).
