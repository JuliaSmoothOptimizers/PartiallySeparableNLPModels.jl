# Benchmark Report for *PartiallySeparableNLPModel*

## Job Properties
* Time of benchmarks:
    - Target: 3 Jun 2020 - 00:00
    - Baseline: 2 Jun 2020 - 23:59
* Package commits:
    - Target: eb1978
    - Baseline: eb1978
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

| ID                                       | time ratio                   | memory ratio |
|------------------------------------------|------------------------------|--------------|
| `["SPS_function", "Hessien ros 20 var"]` |                1.17 (5%) :x: |   1.00 (1%)  |
| `["SPS_function", "Hessien ros 30 var"]` | 0.73 (5%) :white_check_mark: |   1.00 (1%)  |
| `["SPS_function", "OBJ ros 10 var"]`     | 0.88 (5%) :white_check_mark: |   1.00 (1%)  |
| `["SPS_function", "OBJ ros 30 var"]`     | 0.53 (5%) :white_check_mark: |   1.00 (1%)  |
| `["SPS_function", "grad ros 10 var"]`    | 0.87 (5%) :white_check_mark: |   1.00 (1%)  |
| `["SPS_function", "grad ros 20 var"]`    |                1.38 (5%) :x: |   1.00 (1%)  |
| `["SPS_function", "grad ros 30 var"]`    | 0.91 (5%) :white_check_mark: |   1.00 (1%)  |

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
       #1  1498 MHz    7761375            0      3595171     40847750      1008250  ticks
       #2  1498 MHz    4915281            0      1968687     45320093       274656  ticks
       #3  1498 MHz    7796484            0      1747593     42660000        70078  ticks
       #4  1498 MHz    3504062            0       940781     47759203        19734  ticks
       #5  1498 MHz    5282328            0      1531359     45390390        44093  ticks
       #6  1498 MHz   11909765            0      1423921     38870390        29140  ticks
       #7  1498 MHz   10080609            0      1983203     40140265        53156  ticks
       #8  1498 MHz    4098640            0       891500     47213906        21500  ticks
       
  Memory: 31.775043487548828 GB (15444.89453125 MB free)
  Uptime: 52204.0 sec
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
       #1  1498 MHz    7755359            0      3593031     40800531      1008187  ticks
       #2  1498 MHz    4910468            0      1967812     45270421       274656  ticks
       #3  1498 MHz    7777562            0      1745562     42625578        70015  ticks
       #4  1498 MHz    3496968            0       939515     47712203        19734  ticks
       #5  1498 MHz    5268125            0      1529156     45351421        44078  ticks
       #6  1498 MHz   11898390            0      1420703     38829593        29140  ticks
       #7  1498 MHz   10071968            0      1981000     40095734        53109  ticks
       #8  1498 MHz    4089437            0       890250     47169000        21500  ticks
       
  Memory: 31.775043487548828 GB (14962.8125 MB free)
  Uptime: 52148.0 sec
  Load Avg:  0.0  0.0  0.0
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, cannonlake)
```