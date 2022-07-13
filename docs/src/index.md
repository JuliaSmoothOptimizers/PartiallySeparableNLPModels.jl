# PartiallySeparableNLPModels.jl

## Philosophy
The purpose of PartiallySeparableNLPModels.jl is to define automatically partially-separable [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) and facilitate the implementation of partitioned quasi-Newton methods.

## Compatibility
Julia ≥ 1.6.

## How to install
```
pkg> add https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl, https://github.com/paraynaud/PartiallySeparableNLPModels.jl
pkg> test PartiallySeparableNLPModels
```

## How to use 
See the [tutorial](https://paraynaud.github.io/PartiallySeparableNLPModels.jl/dev/tutorial/).

## Dependencies
The module uses [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl) to detect the partially-separable structure and [PartitionedStructures.jl](https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl) to allocate partitioned quasi-Newton approximations.

## Application
[PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl) implements partitioned quasi-Newton trust-region methods from `PartitionedData_TR_PQN` and the `PartiallySeparableNLPModels.jl` methods.