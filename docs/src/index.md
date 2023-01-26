# PartiallySeparableNLPModels: a NLPModel exploiting automatically its partially-separable structure (to define partitioned quasi-Newton models)

## How to cite
If you use PartiallySeparableNLPModels.jl in your work, please cite using the format given in [CITATION.bib](CITATION.bib).

## Philosophy
The purpose of PartiallySeparableNLPModels.jl is to define automatically partially-separable [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
Moreover, it defines several partitioned quasi-Newton models which are meant to be minimized through solvers from [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl)

## Compatibility
Julia ≥ 1.6.

## How to install
```
pkg> add PartiallySeparableNLPModels
pkg> test PartiallySeparableNLPModels
```

## How to use 
See the [tutorial](https://JuliaSmoothOptimizers.github.io/PartiallySeparableNLPModels.jl/main/tutorial/).

## Dependencies
The module uses [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl) to detect the partially-separable structure, [PartitionedStructures.jl](https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl) to allocate partitioned quasi-Newton approximations and [PartitionedVectors.jl](https://github.com/JuliaSmoothOptimizers/PartitionedVectors.jl) to fit the `AbstractVector` interface mandatory for AbstractNLPModel methods.

# Bug reports and discussions
If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/PartiallySeparableNLPModels.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
