# PartiallySeparableNLPModels.jl Tutorial

## An `NLPModel` exploiting partial separability
PartiallySeparableNLPModels.jl defines a subtype of `AbstractNLPModel` to exploit automatically the partially-separable structure of $f:\R^n \to \R$
```math
 f(x) = \sum_{i=1}^N f_i (U_i x) , \; f_i : \R^{n_i} \to \R, \; U_i \in \R^{n_i \times n},\; n_i \ll n,
```
as the sum of element functions $f_i$.

PartiallySeparableNLPModels.jl relies on [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl) to detect the partially-separable structure and defines the suitable partitioned structures, required by the partitioned derivatives, using [PartitionedStructures.jl](https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl).

As a user, you need only define an `NLPModel` with an objective function implemented in pure Julia.
For instance, one may use an `ADNLPModel`:
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
and call `PSNLPModel<:AbstractPartiallySeparableNLPModel` to define a partitioned `NLPModel`:
```@example PSNLP
pqn_adnlp = PSNLPModel(model)
```

Then, you can apply the usual methods `obj` and `grad`, `hprod` from [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl):
```@example PSNLP
using NLPModels
x = ones(n)
fx = NLPModels.obj(pqn_adnlp, x) # compute the obective function
```

```@example PSNLP
gx = NLPModels.grad(pqn_adnlp, x) # compute the gradient
```

```@example PSNLP
gx == NLPModels.grad(model, x)
```

```@example PSNLP
v = ones(n)
hv = NLPModels.hprod(model, x, v)
```
`fx`, `gx` and `hv` accumulate respectively the element functions $f_i$, the element gradients $\nabla f_i$, respectively and element Hessian-vector $\nabla^2 f_i(U_i x) U_i v$ contributions.
In addition, a `PSNLPModel` stores the value of each element gradient and element Hessian-vector product.

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
pqn_jumpnlp = PSNLPModel(jumpnlp_example)

fx = NLPModels.obj(pqn_jumpnlp, x) # compute the obective function
gx = NLPModels.grad(pqn_jumpnlp, x) # compute the gradient
```
In version v0.2.0, [`ManualNLPModel`](https://github.com/JuliaSmoothOptimizers/ManualNLPModels.jl)s will be supported.

## Partitioned quasi-Newton `NLPModel`s
A model deriving from `AbstractPQNNLPModel<:AbstractPartiallySeparableNLPModel` allocates storage required for partitioned quasi-Newton updates, which are implemented in `PartitionedStructures.jl` (see the [PartitionedStructures.jl tutorial](https://juliasmoothoptimizers.github.io/PartitionedStructures.jl/dev/tutorial/) for more details).
There are several variants:
* 'PBFGSNLPModel': every element-Hessian approximation is updated with BFGS;
* 'PSR1NLPModel': every element-Hessian approximation is updated with SR1;
* 'PSENLPModel': every element-Hessian approximation is updated with BFGS if the curvature condition holds, or with SR1 otherwise;
* 'PCSNLPModel': each element-Hessian approximation with BFGS if it is classified as `convex`, or with SR1 otherwise;
* 'PLBFGSNLP': every element-Hessian approximations is a LBFGS operator;
* 'PLSR1NLPModel': every element-Hessian approximations is a LSR1 operator;
* 'PLSENLPModel': by default, every element-Hessian approximations is a LBFGS operator as long as the curvature condition holds, otherwise it becomes a LSR1 operator.

```@example PSNLP
pbfgsnlp = PBFGSNLPModel(jumpnlp_example)
```

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
\right ].
```
The accumulated matrix can be visualized with:
```@example PSNLP
Matrix(hess_approx(pbfgsnlp))
```

Then, you can update the partitioned quasi-Newton approximation with the pair `x,s`:
```@example PSNLP
s = rand(n)
update_nlp(pbfgsnlp, x, s)
```
and you can perform a partitioned-matrix-vector product with:
```@example PSNLP
v = ones(n)
Bv = hprod(pbfgsnlp, x, v)
```

Moreover, there is an interface to `LinearOperator` (see [LinearOperators](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)) from any `AbstractPQNNLPModel`:
```@example PSNLP
using LinearOperators
B = LinearOperator(pbfgsnlp)
B*v
```
which can be paired with iterative solvers (see [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl)).
