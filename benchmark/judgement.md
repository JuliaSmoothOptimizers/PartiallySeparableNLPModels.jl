# Benchmark Report for *PartiallySeparableNLPModels*

## Job Properties
* Time of benchmarks:
    - Target: 21 Jul 2022 - 18:49
    - Baseline: 21 Jul 2022 - 18:47
* Package commits:
    - Target: cd0ac6
    - Baseline: cd0ac6
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

| ID                         | time ratio                   | memory ratio |
|----------------------------|------------------------------|--------------|
| `["GRAD", "arwhead 1000"]` | 0.93 (5%) :white_check_mark: |   1.00 (1%)  |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["GRAD"]`
- `["OBJ"]`

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
       #1  1498 MHz    3966953            0      2743656     28590671      1108031  ticks
       #2  1498 MHz    2894390            0      1185718     31220953       162796  ticks
       #3  1498 MHz    4469656            0      1620703     29210703       245250  ticks
       #4  1498 MHz    5101125            0      1409968     28789968        54218  ticks
       #5  1498 MHz    4915000            0      1644031     28742031       170171  ticks
       #6  1498 MHz    3092359            0       830140     31378562        67578  ticks
       #7  1498 MHz    4407125            0      1227671     29666250       116781  ticks
       #8  1498 MHz    3916687            0      1139828     30244531       119593  ticks
       
  Memory: 31.775043487548828 GB (15834.48828125 MB free)
  Uptime: 35301.0 sec
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
       #1  1498 MHz    3961156            0      2740453     28510343      1107484  ticks
       #2  1498 MHz    2887796            0      1184000     31139937       162796  ticks
       #3  1498 MHz    4456390            0      1617421     29137921       245218  ticks
       #4  1498 MHz    5083187            0      1407234     28721312        54218  ticks
       #5  1498 MHz    4906687            0      1639406     28665640       170140  ticks
       #6  1498 MHz    3080906            0       828656     31302171        67578  ticks
       #7  1498 MHz    4395000            0      1223609     29593109       116765  ticks
       #8  1498 MHz    3889703            0      1136203     30185812       119578  ticks
       
  Memory: 31.775043487548828 GB (15745.2734375 MB free)
  Uptime: 35211.0 sec
  Load Avg:  0.0  0.0  0.0
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, icelake-client)
```