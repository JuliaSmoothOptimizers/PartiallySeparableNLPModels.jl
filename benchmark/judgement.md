# Benchmark Report for *PartiallySeparableNLPModel*

## Job Properties
* Time of benchmarks:
    - Target: 22 Jun 2020 - 16:35
    - Baseline: 22 Jun 2020 - 16:31
* Package commits:
    - Target: f20733
    - Baseline: 4592ec
* Julia commits:
    - Target: 2d5741
    - Baseline: 2d5741
* Julia command flags:
    - Target: None
    - Baseline: None
* Environment variables:
    - Target: None
    - Baseline: None

## Results
A ratio greater than `1.0` denotes a possible regression (marked with :x:), while a ratio less
than `1.0` denotes a possible improvement (marked with :white_check_mark:). Only significant results - results
that indicate possible regressions or improvements - are shown below (thus, an empty table means that all
benchmark results remained invariant between builds).

| ID                                         | time ratio                   | memory ratio  |
|--------------------------------------------|------------------------------|---------------|
| `["SPS_function", "Hessien ros 100 var"]`  |                1.16 (5%) :x: | 1.02 (1%) :x: |
| `["SPS_function", "Hessien ros 1000 var"]` |                1.11 (5%) :x: | 1.02 (1%) :x: |
| `["SPS_function", "Hessien ros 200 var"]`  | 0.64 (5%) :white_check_mark: | 1.02 (1%) :x: |
| `["SPS_function", "Hessien ros 2000 var"]` |                1.57 (5%) :x: | 1.02 (1%) :x: |
| `["SPS_function", "Hessien ros 500 var"]`  |                1.31 (5%) :x: | 1.02 (1%) :x: |
| `["SPS_function", "Hessien ros 5000 var"]` | 0.70 (5%) :white_check_mark: | 1.02 (1%) :x: |
| `["SPS_function", "OBJ ros 100 var"]`      | 0.79 (5%) :white_check_mark: |    1.00 (1%)  |
| `["SPS_function", "OBJ ros 200 var"]`      |                1.37 (5%) :x: |    1.00 (1%)  |
| `["SPS_function", "OBJ ros 2000 var"]`     |                1.94 (5%) :x: |    1.00 (1%)  |
| `["SPS_function", "OBJ ros 500 var"]`      |                1.13 (5%) :x: |    1.00 (1%)  |
| `["SPS_function", "OBJ ros 5000 var"]`     |                2.13 (5%) :x: |    1.00 (1%)  |
| `["SPS_function", "grad ros 100 var"]`     |                1.31 (5%) :x: |    1.00 (1%)  |
| `["SPS_function", "grad ros 1000 var"]`    |                1.47 (5%) :x: |    1.00 (1%)  |
| `["SPS_function", "grad ros 200 var"]`     |                1.39 (5%) :x: |    1.00 (1%)  |
| `["SPS_function", "grad ros 2000 var"]`    |                2.54 (5%) :x: |    1.00 (1%)  |
| `["SPS_function", "grad ros 500 var"]`     |                1.29 (5%) :x: |    1.00 (1%)  |
| `["SPS_function", "grad ros 5000 var"]`    |                2.24 (5%) :x: |    1.00 (1%)  |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["SPS_function"]`

## Julia versioninfo

### Target
```
Julia Version 1.3.1
Commit 2d5741174c (2019-12-30 21:36 UTC)
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
      Microsoft Windows [version 10.0.18362.476]
  CPU: Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz: 
              speed         user         nice          sys         idle          irq
       #1  1498 MHz    1829656            0      1555171     24202281       563875  ticks
       #2  1498 MHz    1744765            0       595921     25246203        31828  ticks
       #3  1498 MHz    2572656            0       884203     24130031        15109  ticks
       #4  1498 MHz    1893281            0       622921     25070687        11843  ticks
       #5  1498 MHz    2124937            0       787343     24674609        13625  ticks
       #6  1498 MHz    1243437            0       420437     25923015         7031  ticks
       #7  1498 MHz    1866187            0       569281     25151406        13140  ticks
       #8  1498 MHz    2700125            0      1286765     23599984         7078  ticks
       
  Memory: 31.775043487548828 GB (16665.6875 MB free)
  Uptime: 27586.0 sec
  Load Avg:  0.0  0.0  0.0
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, cannonlake)
```

### Baseline
```
Julia Version 1.3.1
Commit 2d5741174c (2019-12-30 21:36 UTC)
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
      Microsoft Windows [version 10.0.18362.476]
  CPU: Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz: 
              speed         user         nice          sys         idle          irq
       #1  1498 MHz    1790468            0      1538078     24047328       561421  ticks
       #2  1498 MHz    1706750            0       587625     25081281        31546  ticks
       #3  1498 MHz    2510296            0       870359     23995000        14734  ticks
       #4  1498 MHz    1843250            0       613812     24918593        11625  ticks
       #5  1498 MHz    2072359            0       774312     24528984        13515  ticks
       #6  1498 MHz    1187234            0       410171     25778250         6890  ticks
       #7  1498 MHz    1799375            0       559015     25017250        13000  ticks
       #8  1498 MHz    2612468            0      1274078     23489093         6984  ticks
       
  Memory: 31.775043487548828 GB (17668.48046875 MB free)
  Uptime: 27375.0 sec
  Load Avg:  0.0  0.0  0.0
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, cannonlake)
```