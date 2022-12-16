# PartiallySeparableNLPModels.jl

## Philosophy
The purpose of PartiallySeparableNLPModels.jl is to define automatically partially-separable [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) and facilitate the implementation of partitioned quasi-Newton methods.

## Compatibility
Julia ≥ 1.6.

## How to install
```
pkg> add PartiallySeparableNLPModels
pkg> test PartiallySeparableNLPModels
```

## How to use 
See the [tutorial](https://JuliaSmoothOptimizers.github.io/PartiallySeparableNLPModels.jl/dev/tutorial/).

## Dependencies
The module uses [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl) to detect the partially-separable structure and [PartitionedStructures.jl](https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl) to allocate partitioned quasi-Newton approximations.

## Application
[PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl) implements partitioned quasi-Newton trust-region methods from `PartitionedDataTRPQN` and the `PartiallySeparableNLPModels.jl` methods.

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/PartiallySeparableNLPModels.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
