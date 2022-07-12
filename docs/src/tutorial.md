# PartiallySeparableNLPModels.jl Tutorial

## A `NLPModel` exploiting the partial separability
PartiallySeparableNLPModels.jl define `NLPModel` to implement partitioned quasi-Newton methods exploiting automatically the partially-separable structure of $f:\R^n \to \R$
```math
 f(x) = \sum_{i=1}^N f_i (U_i x) , \; f_i : \R^{n_i} \to \R, \; U_i \in \R^{n_i \times n},\; n_i \ll n,
```
as the sum of element function $f_i$.

PartiallySeparableNLPModels.jl relies on [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl) to detect the partially-separable structure and define the suitable partitioned structures, required by the partitioned derivatives, using [PartitionedStructures.jl](https://github.com/paraynaud/PartitionedStructures.jl).

As a user, you only need to define your `ADNLPModel`:
```@example PSNLP
using PartiallySeparableNLPModels, ADNLPModels

function example(x)
  n = length(x)
  n < 2 && @error("length of x must be >= 2")
return sum((x[j]+x[j+1])^2 for i=1:n+1)
end 
start_example(n :: Int) = ones(n)
example_ADNLPModel(n :: Int) = ADNLPModel(example, start_example(n), name="Example " * string(n) * " variables")

n = 4 # size of the problem
adnlp_example = example_ADNLPModel(n)
```
and call `PQNNLPModel` to define a partitioned quasi-Newton `NLPModel`:
```@example PSNLP
pqn_adnlp = PQNNLPModel(adnlp_example)
```

Then, you can apply the usual methods `obj` and `grad`, exploiting the partial separability, from [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl):
```@example PSNLP
using NLPModels
x = rand(n)
fx = NLPModels.obj(pqn_adnlp, x) # compute the obective function
gx = NLPModels.grad(pqn_adnlp, x) # compute the gradient
```

The same procedure can be applied to `MathNLPModel`s:
```@example PSNLP
using JuMP, MathOptInterface, NLPModelsJuMP

function jump_example(n::Int)
  m = Model()
  @variable(m, x[1:n])
  @NLobjective(m, Min, sum((x[i]^2 + x[i+1])^2 for i = 1:n-1))
  evaluator = JuMP.NLPEvaluator(m)
  MathOptInterface.initialize(evaluator, [:ExprGraph])
  variables = JuMP.all_variables(m)
  x0 = ones(n)
  JuMP.set_start_value.(variables, x0)
  nlp = MathOptNLPModel(m)
  return nlp
end

jumpnlp_example = jump_example(n)
pqn_jumpnlp = PQNNLPModel(jumpnlp_example)

fx = NLPModels.obj(pqn_jumpnlp, x) # compute the obective function
gx = NLPModels.grad(pqn_jumpnlp, x) # compute the gradient
```

## A partitioned quasi-Newton `NLPModel`
By defining a `PQNNLPModel` you allocate a partitioned quasi-Newton update, which is implemented in `PartitionedStructures.jl`.
You can visualize the matrix with:
```@example PSNLP
B = Matrix(get_pB(pqn_jumpnlp))
```
where each element Hessian approximation is instantiated with an identity matrix.

You can specify the partitioned quasi-Newton update with the optional argument `name`:
```@example PSNLP
PQNNLPModel(jumpnlp_example; name=:plse) # by default
```
The possible variants are: `:pbfg, :psr1, :pse, :plbfgs, :plsr1` and `:plse`, see the [PartitionedStructures.jl tutorial](https://juliasmoothoptimizers.github.io/PartitionedStructures.jl/dev/tutorial/) for more details.

Then, you can update the partitioned quasi-Newton approximation with the pair `x,s`:
```@example PSNLP
s = rand(n)
update_B = update_nlp(pqn_adnlp, x, s)
```
and you can perform the partitioned matrix-vector product with:
```@example PSNLP
v = ones(n)
Bv = product_part_data_x(pqn_adnlp, v)
```

A variant allocating in place the result helps to define a `LinearOperator` (see [LinearOperators](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)) from a `PQNNLPModel`:
```@example PSNLP
using LinearOperators
T = eltype(x)
linear_operator = LinearOperators.LinearOperator(T, n, n, true, true, ((res, v) -> product_part_data_x!(res, pqn_adnlp, v)))
linear_operator*v
```
which can be used later with iterative methods (see [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl)).