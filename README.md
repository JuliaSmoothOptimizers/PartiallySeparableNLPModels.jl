# PartiallySeparableNLPModels : A three-way bridge between [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl), [PartitionedStructures.jl](https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl) and [PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl).

| **Documentation** | **Linux/macOS/Windows/FreeBSD** | **Coverage** | **DOI** |
|:-----------------:|:-------------------------------:|:------------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] [![build-cirrus][build-cirrus-img]][build-cirrus-url] | [![codecov][codecov-img]][codecov-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaSmoothOptimizers.github.io/PartiallySeparableNLPModels.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://JuliaSmoothOptimizers.github.io/PartiallySeparableNLPModels.jl/dev
[build-gh-img]: https://github.com/JuliaSmoothOptimizers/PartiallySeparableNLPModels.jl/workflows/CI/badge.svg?branch=master
[build-gh-url]: https://github.com/JuliaSmoothOptimizers/PartiallySeparableNLPModels.jl/actions
[build-cirrus-img]: https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/PartiallySeparableNLPModels.jl?logo=Cirrus%20CI
[build-cirrus-url]: https://cirrus-ci.com/github/JuliaSmoothOptimizers/PartiallySeparableNLPModels.jl
[codecov-img]: https://codecov.io/gh/JuliaSmoothOptimizers/PartiallySeparableNLPModels.jl/branch/master/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/JuliaSmoothOptimizers/PartiallySeparableNLPModels.jl
[doi-img]: https://zenodo.org/badge/267062779.svg
[doi-url]: https://zenodo.org/badge/latestdoi/267062779

## How to cite

If you use PartiallySeparableNLPModels.jl in your work, please cite using the format given in [CITATION.bib](CITATION.bib).

## Philosophy
The purpose of PartiallySeparableNLPModels.jl is to define automatically partially-separable [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) and facilitate the implementation of partitioned quasi-Newton methods.

## Compatibility
Julia ≥ 1.6.

## How to install
```
pkg> add https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl, https://github.com/JuliaSmoothOptimizers/PartiallySeparableNLPModels.jl
pkg> test PartiallySeparableNLPModels
```

## How to use 
See the [tutorial](https://JuliaSmoothOptimizers.github.io/PartiallySeparableNLPModels.jl/dev/tutorial/).

## Dependencies
The module uses [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl) to detect the partially-separable structure and [PartitionedStructures.jl](https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl) to allocate partitioned quasi-Newton approximations.

## Application
[PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl) implements partitioned quasi-Newton trust-region methods from `PartitionedDataTRPQN` and the `PartiallySeparableNLPModels.jl` methods.
