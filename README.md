# PartiallySeparableNLPModels: Exploiting the partially-separable structure to define partitioned quasi-Newton NLPModels

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
See the [tutorial](https://JuliaSmoothOptimizers.github.io/PartiallySeparableNLPModels.jl/stable/tutorial/).

## Dependencies
The module uses [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl) to detect the partially-separable structure, [PartitionedStructures.jl](https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl) to allocate partitioned quasi-Newton approximations and [PartitionedVectors.jl](https://github.com/JuliaSmoothOptimizers/PartitionedVectors.jl) to fit the `AbstractVector` interface mandatory for AbstractNLPModel methods.

# Bug reports and discussions
If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/PartiallySeparableNLPModels.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
