# Benchmark Report for *PartiallySeparableNLPModels*

## Job Properties
* Time of benchmarks:
    - Target: 21 Jul 2022 - 19:06
    - Baseline: 21 Jul 2022 - 19:05
* Package commits:
    - Target: 80e3f0
    - Baseline: 80e3f0
* Julia commits:
    - Target: ac5cc9
    - Baseline: ac5cc9
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

| ID                                 | time ratio    | memory ratio |
|------------------------------------|---------------|--------------|
| `["PSNLPS", "GRAD, arwhead 10"]`   | 1.17 (5%) :x: |   1.00 (1%)  |
| `["PSNLPS", "GRAD, arwhead 100"]`  | 1.14 (5%) :x: |   1.00 (1%)  |
| `["PSNLPS", "OBJ, arwhead 10"]`    | 1.08 (5%) :x: |   1.00 (1%)  |
| `["PSNLPS", "OBJ, arwhead 100"]`   | 1.13 (5%) :x: |   1.00 (1%)  |
| `["PSNLPS", "OBJ, arwhead 1000"]`  | 1.12 (5%) :x: |   1.00 (1%)  |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["PSNLPS"]`

## Julia versioninfo

### Target
```
Julia Version 1.7.1
Commit ac5cc99908 (2021-12-22 19:35 UTC)
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
      Microsoft Windows [version 10.0.18362.476]
  CPU: Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz: 
              speed         user         nice          sys         idle          irq
       #1  1498 MHz    4025437            0      2789500     29542734      1120703  ticks
       #2  1498 MHz    2942140            0      1204765     32210546       163000  ticks
       #3  1498 MHz    4566375            0      1659281     30131796       245703  ticks
       #4  1498 MHz    5181328            0      1428843     29747281        54531  ticks
       #5  1498 MHz    4992000            0      1674156     29691296       170578  ticks
       #6  1498 MHz    3163781            0       848500     32345171        67781  ticks
       #7  1498 MHz    4520578            0      1252812     30584046       117125  ticks
       #8  1498 MHz    4080000            0      1165625     31111812       119859  ticks
       
  Memory: 31.775043487548828 GB (15769.29296875 MB free)
  Uptime: 36357.0 sec
  Load Avg:  0.0  0.0  0.0
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, icelake-client)
```

### Baseline
```
Julia Version 1.7.1
Commit ac5cc99908 (2021-12-22 19:35 UTC)
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
      Microsoft Windows [version 10.0.18362.476]
  CPU: Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz: 
              speed         user         nice          sys         idle          irq
       #1  1498 MHz    4012078            0      2783359     29462093      1120312  ticks
       #2  1498 MHz    2927859            0      1201125     32128328       162984  ticks
       #3  1498 MHz    4549359            0      1653312     30054640       245671  ticks
       #4  1498 MHz    5159890            0      1424671     29672750        54437  ticks
       #5  1498 MHz    4975281            0      1668828     29613203       170515  ticks
       #6  1498 MHz    3147906            0       844921     32264484        67781  ticks
       #7  1498 MHz    4496406            0      1247390     30513500       117062  ticks
       #8  1498 MHz    4044937            0      1161640     31050718       119796  ticks
       
  Memory: 31.775043487548828 GB (14595.515625 MB free)
  Uptime: 36257.0 sec
  Load Avg:  0.0  0.0  0.0
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, icelake-client)
```