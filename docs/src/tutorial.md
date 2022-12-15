# PartiallySeparableNLPModels.jl Tutorial

## An `NLPModel` exploiting partial separability
PartiallySeparableNLPModels.jl defines a subtype of `AbstractNLPModel` to exploit automatically the partially-separable structure of $f:\R^n \to \R$
```math
 f(x) = \sum_{i=1}^N f_i (U_i x) , \; f_i : \R^{n_i} \to \R, \; U_i \in \R^{n_i \times n},\; n_i \ll n,
```
as a sum of element functions $f_i$.

PartiallySeparableNLPModels.jl relies on [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl) to detect the partially-separable structure. Then, it defines suitable partitioned structures using [PartitionedStructures.jl](https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl) and [PartitionedVectors.jl](https://github.com/paraynaud/PartitionedVectors.jl).
Any `NLPModels` from PartiallySeparableNLPModels.jl rely on `PartitionedVector <: AbstractVector` instead of `Vector`.
Any model from PartiallySeparableNLPModels.jl may be defined either from a `ADNLPModel` or a `MathOptNLPModel`.
Let starts with an example using an `ADNLPModel` (`MathOptNLPModel` will follow):
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
and call `PSNLPModel <: AbstractPartiallySeparableNLPModel` to define a partitioned `NLPModel` using exact second derivatives:
```@example PSNLP
psnlp = PSNLPModel(model)
```
where `psnlp.meta.x0` is a `PartitionedVector`:
```@example PSNLP
psnlp.meta.x0
```

Then, you can apply the usual methods `obj` and `grad`, `hprod` from [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl):
```@example PSNLP
using NLPModels
x = similar(psnlp.meta.x0)
x .= 1
fx = NLPModels.obj(psnlp, x) # compute the objective function
```

```@example PSNLP
gx = NLPModels.grad(psnlp, x) # compute the gradient
```

```@example PSNLP
v = similar(x)
v .= 1
hv = NLPModels.hprod(psnlp, x, v)
```
`fx`, `gx` and `hv` accumulate contributions from element functions, either its evaluation $f_i(U_ix)$, its gradient $\nabla f_i(U_ix)$ or its element Hessian-vector $\nabla^2 f_i(U_i x) U_i v$.
You can get the `Vector` value of `gx` and `hv` with 
```@example PSNLP
Vector(hv)
Vector(gx)
```
and you can find more detail about `PartitionedVector`s in [PartitionedVectors.jl tutorial](https://paraynaud.github.io/PartitionedVectors.jl/stable/).

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

jumpnlp = jump_example(n)
psnlp = PSNLPModel(jumpnlp)

fx = NLPModels.obj(psnlp, x) # compute the objective function
gx = NLPModels.grad(psnlp, x) # compute the gradient
```

## Partitioned quasi-Newton `NLPModel`s
A model deriving from `AbstractPQNNLPModel<:QuasiNewtonModel` allocates storage required for partitioned quasi-Newton updates, which are implemented in `PartitionedStructures.jl` (see the [PartitionedStructures.jl tutorial](https://juliasmoothoptimizers.github.io/PartitionedStructures.jl/stable/tutorial/) for more details).
There are several variants:
* `PBFGSNLPModel`: every element-Hessian approximation is updated with BFGS;
* `PSR1NLPModel`: every element-Hessian approximation is updated with SR1;
* `PSENLPModel`: every element-Hessian approximation is updated with BFGS if the curvature condition holds, or with SR1 otherwise;
* `PCSNLPModel`: each element-Hessian approximation with BFGS if it is classified as convex, or with SR1 otherwise;
* `PLBFGSNLPModel`: every element-Hessian approximations is a LBFGS operator;
* `PLSR1NLPModel`: every element-Hessian approximations is a LSR1 operator;
* `PLSENLPModel`: by default, every element-Hessian approximations is a LBFGS operator as long as the curvature condition holds, otherwise it becomes a LSR1 operator.

```@example PSNLP
pbfgsnlp = PBFGSNLPModel(jumpnlp)
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
Matrix(pbfgsnlp.op)
```

Then, you can update the partitioned quasi-Newton approximation with the pair `y,s`:
```@example PSNLP
s = similar(x)
s .= 0.5
y = grad(pbfgsnlp, x+s) - grad(pbfgsnlp, x)
push!(pbfgsnlp, y, s)
```
and you can perform a partitioned-matrix-vector product with:
```@example PSNLP
Bv = hprod(pbfgsnlp, x, s)
Vector(Bv)
```

Finally, you can build a `TrunkSolver` (from [JSOSolvers](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl)) from a `PartiallySeparableNLPModel`:
```@example PSNLP
using JSOSolvers

trunk_solver = TrunkSolver(pbfgsnlp)
```
which define properly the PartitionedVectors mandatory for running `trunk`
`turnk_solver` can be `solve` afterward with:
```@example PSNLP
solve!(trunk_solver, pbfgsnlp)
```

For now, `TrunkSolver` is the sole `Solver` defined for `PartiallySeparableNLPModel`s, if you want to add another `Solver`, you should define it similarly to `TrunkSolver` in `src/trunk.jl`.