# PartiallySeparableNLPModels.jl Tutorial

## An `NLPModel` exploiting partial separability
PartiallySeparableNLPModels.jl defines a subtype of `AbstractNLPModel` to exploit automatically the partially-separable structure of $f:\R^n \to \R$
```math
 f(x) = \sum_{i=1}^N f_i (U_i x) , \; f_i : \R^{n_i} \to \R, \; U_i \in \R^{n_i \times n},\; n_i \ll n,
```
as the sum of element functions $f_i$.

PartiallySeparableNLPModels.jl relies on [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl) to detect the partially-separable structure and defines the suitable partitioned structures, required by the partitioned derivatives, using [PartitionedStructures.jl](https://github.com/paraynaud/PartitionedStructures.jl).

As a user, you need only define your `ADNLPModel`:
```@example PSNLP
using PartiallySeparableNLPModels, ADNLPModels

function example(x)
  n = length(x)
  n < 2 && @error("length of x must be >= 2")
  return sum((x[i] + x[i+1])^2 for i=1:n-1)
end 
start_example(n :: Int) = ones(n)
example_model(n :: Int) = ADNLPModel(example, start_example(n), name="Example " * string(n) * " variables")

n = 4 # size of the problem
model = example_model(n)
```
and call `PartiallySeparableNLPModel` to define a partitioned `NLPModel`:
```@example PSNLP
pqn_adnlp = PartiallySeparableNLPModel(model)
```

Then, you can apply the usual methods `obj` and `grad` from [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl):
```@example PSNLP
using NLPModels
x = ones(n)
fx = NLPModels.obj(pqn_adnlp, x) # compute the obective function
```

```@example PSNLP
gx = NLPModels.grad(pqn_adnlp, x) # compute the gradient
```
`fx` and `gx` compute and accumulate the element functions $f_i$ and the element gradients $\nabla f_i$, respectively.
In addition, a `PartiallySeparableNLPModel` stores the value of each element gradient to perform partitioned quasi-Newton updates afterward. 

```@example PSNLP
gx == NLPModels.grad(model, x)
```

The same procedure can be applied to `MathOptNLPModel`s:
```@example PSNLP
using JuMP, MathOptInterface, NLPModelsJuMP

function jump_example(n::Int)
  m = Model()
  @variable(m, x[1:n])
  @NLobjective(m, Min, sum((x[i] + x[i+1])^2 for i = 1:n-1))
  evaluator = JuMP.NLPEvaluator(m)
  MathOptInterface.initialize(evaluator, [:ExprGraph])
  variables = JuMP.all_variables(m)
  x0 = ones(n)
  JuMP.set_start_value.(variables, x0)
  nlp = MathOptNLPModel(m)
  return nlp
end

jumpnlp_example = jump_example(n)
pqn_jumpnlp = PartiallySeparableNLPModel(jumpnlp_example)

fx = NLPModels.obj(pqn_jumpnlp, x) # compute the obective function
gx = NLPModels.grad(pqn_jumpnlp, x) # compute the gradient
```

## A partitioned quasi-Newton `NLPModel`
When defining a `PartiallySeparableNLPModel`, you allocate storage for partitioned quasi-Newton updates, which are implemented in `PartitionedStructures.jl`.

The Hessian approximation of each element function $f_i (y) = (y_1 + y_2)^2$ is initially set to an identity matrix. 
The contribution of every element Hessian approximation is accumulated as
```math
\left [
\begin{array}{ccc}
  \left ( \begin{array}{cc}
    1 & \\
    & 1 \\ 
  \end{array} \right ) & & \\
  & 0 & \\
  & & 0 \\
\end{array}
\right ] 
+ 
\left [
\begin{array}{ccc}
  0 & & \\
  & \left ( \begin{array}{cc}
    1 & \\
    & 1 \\ 
  \end{array} \right ) & \\
  & & 0 \\
\end{array}
\right ]
+ 
\left [
\begin{array}{ccc}
  0 & & \\
  & 0 & \\
  & & \left ( \begin{array}{cc}
    1 & \\
    & 1 \\ 
  \end{array} \right )\\
\end{array}
\right ]
```
The accumulated matrix can be visualized with:
```@example PSNLP
Matrix(hess_approx(pqn_jumpnlp))
```

You can specify the partitioned quasi-Newton update with the optional argument `name`:
```julia
PartiallySeparableNLPModel(jumpnlp_example; name=:plse) # by default
```
The possible variants are: `:pbfgs, :psr1, :pse, :plbfgs, :plsr1` and `:plse`, see the [PartitionedStructures.jl tutorial](https://juliasmoothoptimizers.github.io/PartitionedStructures.jl/dev/tutorial/) for more details.

Then, you can update the partitioned quasi-Newton approximation with the pair `x,s`:
```@example PSNLP
s = rand(n)
update_nlp(pqn_adnlp, x, s)
```
and you can perform a partitioned-matrix-vector product with:
```@example PSNLP
v = ones(n)
Bv = hprod(pqn_adnlp, x, v)
```

An in-place variant helps define a `LinearOperator` (see [LinearOperators](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)) from a `PartiallySeparableNLPModel`:
```@example PSNLP
using LinearOperators
T = eltype(x)
B = LinearOperator(T, n, n, true, true, ((Hv, v) -> hprod!(pqn_adnlp, x, v, Hv)))
B*v
```
which can be paired with iterative solvers (see [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl)).