# Benchmark Report for *PartiallySeparableNLPModels*

## Job Properties
* Time of benchmarks:
    - Target: 21 Jul 2022 - 19:01
    - Baseline: 21 Jul 2022 - 19:00
* Package commits:
    - Target: 35a622
    - Baseline: 35a622
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

| ID                                   | time ratio    | memory ratio |
|--------------------------------------|---------------|--------------|
| `["PSNLPS", "GRAD", "arwhead 1000"]` | 1.08 (5%) :x: |   1.00 (1%)  |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["PSNLPS", "GRAD"]`
- `["PSNLPS", "OBJ"]`

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
       #1  1498 MHz    3989171            0      2769312     29300218      1118000  ticks
       #2  1498 MHz    2908156            0      1194359     31955968       162890  ticks
       #3  1498 MHz    4513312            0      1641390     29903781       245468  ticks
       #4  1498 MHz    5123937            0      1416812     29517734        54359  ticks
       #5  1498 MHz    4947015            0      1656703     29454765       170375  ticks
       #6  1498 MHz    3116375            0       836781     32105328        67703  ticks
       #7  1498 MHz    4457468            0      1237437     30363562       116968  ticks
       #8  1498 MHz    3986046            0      1152078     30920343       119750  ticks
       
  Memory: 31.775043487548828 GB (15908.2734375 MB free)
  Uptime: 36058.0 sec
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
       #1  1498 MHz    3983093            0      2764875     29197703      1117078  ticks
       #2  1498 MHz    2903203            0      1192109     31850140       162859  ticks
       #3  1498 MHz    4500718            0      1635968     29808765       245421  ticks
       #4  1498 MHz    5115265            0      1413718     29416468        54343  ticks
       #5  1498 MHz    4937765            0      1653656     29354031       170359  ticks
       #6  1498 MHz    3107281            0       835406     32002765        67671  ticks
       #7  1498 MHz    4438359            0      1235296     30271781       116906  ticks
       #8  1498 MHz    3951359            0      1148765     30845312       119718  ticks
       
  Memory: 31.775043487548828 GB (15878.171875 MB free)
  Uptime: 35945.0 sec
  Load Avg:  0.0  0.0  0.0
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, icelake-client)
```