# PartiallySeparableNLPModels : A three-ways bridge between [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl), [PartitionedStructures.jl](https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl) and [PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl).

| **Documentation** | **Linux/macOS/Windows/FreeBSD** | **Coverage** | **DOI** |
|:-----------------:|:-------------------------------:|:------------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] [![build-cirrus][build-cirrus-img]][build-cirrus-url] | [![codecov][codecov-img]][codecov-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://paraynaud.github.io/PartiallySeparableNLPModels.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://paraynaud.github.io/PartiallySeparableNLPModels.jl/dev
[build-gh-img]: https://github.com/paraynaud/PartiallySeparableNLPModels.jl/workflows/CI/badge.svg?branch=master
[build-gh-url]: https://github.com/paraynaud/PartiallySeparableNLPModels.jl/actions
[build-cirrus-img]: https://img.shields.io/cirrus/github/paraynaud/PartiallySeparableNLPModels.jl?logo=Cirrus%20CI
[build-cirrus-url]: https://cirrus-ci.com/github/paraynaud/PartiallySeparableNLPModels.jl
[codecov-img]: https://codecov.io/gh/paraynaud/PartiallySeparableNLPModels.jl/branch/master/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/paraynaud/PartiallySeparableNLPModels.jl
[doi-img]: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.822073-blue.svg
[doi-url]: https://doi.org/10.5281/zenodo.822073

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