# PartiallySeparableNLPModels : A bridge between [CalculusTreeTools.jl](https://github.com/paraynaud/CalculusTreeTools.jl), [PartitionedStructures.jl](https://github.com/paraynaud/PartitionedStructures.jl) and [PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl).

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

## Motivation
The module [PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl) seeks to minimize the partially separable functions
$$
f(x) = \sum_{=1}^N \hat{f}_i (U_i x), \quad f \in \R^n \to \R, \quad \hat f_i:\R^{n_i} \to \R, \quad U_i \in \R^{n_i \times n}.
$$
$f$ is a sum of element functions $\hat{f}_i$, and usually $n_i \ll n$. $U_i$ is a linear operator, it selects the variables that parametrizes $\hat{f}_i$.

PartiallySeparableNLPModels.jl define and manage the structures required by [PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl) to run a trust-region method that exploits the partially separable structure of $f$.
Mainly, it manages the partitioned derivatives of $f$, such as the gradient 
$$
\nabla f(x) = \sum_{i=1}^N U_i^\top \nabla \hat{f}_i (U_i x),
$$
and the hessian 
$$
\nabla^2 f(x) = \sum_{i=1}^N U_i^\top \nabla^2 \hat{f_i} (U_i x) U_i,
$$
 where both are the sum of the element derivatives $\nabla \hat{f}_i,  \nabla^2\hat{f}_i$.
Moreover, this structure allows to define a partitioned quasi-Newton approximation of $\nabla^2 f$
$$
B = \sum_{i=1}^N U_i^\top \hat{B}_{i} U_i
$$
where each $\hat{B}_i \approx \nabla^2 \hat{f}_i$.

#### Reference
* A. Griewank and P. Toint, [*Partitioned variable metric updates for large structured optimization problems*](10.1007/BF01399316), Numerische Mathematik volume, 39, pp. 119--137, 1982.

## Content
PartiallySeparableNLPModels.jl use the module [CalculusTreeTools.jl](https://github.com/paraynaud/CalculusTreeTools.jl) to detect automatically the partially separable structure of $f$.
Once it is done, it defines the partitioned structures of $\nabla f$ and $B \approx \nabla^2 f$ with [PartitionedStructures.jl](https://github.com/paraynaud/PartitionedStructures.jl).

Considering the following `ADNLPModel`
```julia
using PartiallySeparableNLPModels, ADNLPModels

  function example(x)
    n = length(x)
    n < 2 && @error("length of x must be >= 2")
  return sum( sum( x[j] for j=1:i)^2 for i=2:n)
end 
start_example(n :: Int) = ones(n)
example_ADNLPModel(n :: Int=100) = ADNLPModel(example, start_example(n), name="Example "*string(n) * " variables")

n = 50 # size of the problem
nlp_example = example_ADNLPModel(n) # example model of 
```

You can either define the structures required for a trust region method using partitioned quasi-Newton update or define an `NLPModel` to evaluate $f, \nabla f, \nabla^2 f$ by exploiting the partial separabiliy of $f$
```julia
using NLPModels

partially_separable_nlp = PQNNLPModel(nlp_example)
x = rand(n)
NLPModels.obj(partially_separable_nlp, x) # compute the obective function
NLPModels.grad(partially_separable_nlp, x) # compute the gradient
```
To define the structure for a trust region method, you have to extract `ex` the `Expr` from `nlp_example` and `n`. You can use
```julia
(ex, n, x0) = get_expr_tree(nlp_example)
```
after that you can call `PartitionedData_TR_PQN`
```
part_data_pqn = build_PartitionedData_TR_PQN(ex, n; name=:pbfgs, x0=x0)
```
The sole purpose of this module is to simplify how the module [PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl) handle the partial separability.

## Dependencies
The module depends of : [CalculusTreeTools.jl](https://github.com/paraynaud/CalculusTreeTools.jl) to detects the partially separable structure and [PartitionedStructures.jl](https://github.com/paraynaud/PartitionedStructures.jl) to produce the partitioned quasi-Newton approximation.

## How to install
```
julia> ]
pkg> add https://github.com/paraynaud/PartitionedStructures.jl, https://github.com/paraynaud/CalculusTreeTools.jl, 
pkg> test PartiallySeparableNLPModels
```