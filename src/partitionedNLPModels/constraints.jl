"""
    trees = get_constraint_expression_trees(nlp::ADNLPModel)
    trees = get_constraint_expression_trees(nlp::MathOptNLPModel)

Return a `Vector` of `ExpressionTreeForge.Type_expr_tree`, one per row of the constraint
function `c(x)` of `nlp`, i.e. `trees[j]` represents `cⱼ(x)`. `trees` is empty whenever
`nlp.meta.ncon == 0`.

Mirrors `ExpressionTreeForge.get_expression_tree`, which only extracts the objective.

Current limitations:
* for an `ADNLPModel`, every constraint must be defined inside the nonlinear constraint
  function (`nlp.c!`); models built with a separate linear-constraint block
  (`clinrows`/`clincols`/`clinvals`) are not supported;
* for a `MathOptNLPModel`, only constraints declared with the (legacy) `@NLconstraint`
  macro are supported. `MathOptInterface` only records an expression graph for constraints
  registered in its `MOI.Nonlinear.Model`, which is not populated by constraints declared
  with the modern `@constraint` macro.
"""
function get_constraint_expression_trees(nlp::ADNLPModel)
  ncon = nlp.meta.ncon
  ncon == 0 && return ExpressionTreeForge.Type_expr_tree[]
  isempty(nlp.clinrows) || error(
    "get_constraint_expression_trees does not support an ADNLPModel with a separate linear-constraint block (clinrows/clincols/clinvals); encode every constraint inside the nonlinear constraint function c!",
  )
  n = nlp.meta.nvar
  # the variable must be named `x`: ExpressionTreeForge parses the unicode-subscripted
  # symbol names produced by `Symbolics._toexpr` (e.g. `x₁`) by splitting on the literal
  # character 'x', exactly as `ExpressionTreeForge.get_expression_tree(::ADNLPModel)` does
  # for the objective.
  Symbolics.@variables x[1:n]
  cx = Vector{Symbolics.Num}(undef, ncon)
  nlp.c!(cx, x)
  return map(j -> ExpressionTreeForge.transform_to_expr_tree(Symbolics._toexpr(cx[j])), 1:ncon)
end

function get_constraint_expression_trees(nlp::MathOptNLPModel)
  ncon = nlp.meta.ncon
  ncon == 0 && return ExpressionTreeForge.Type_expr_tree[]
  evaluator = nlp.eval
  MathOptInterface.initialize(evaluator, [:ExprGraph])
  return map(1:ncon) do j
    expr = try
      MathOptInterface.constraint_expr(evaluator, j)
    catch e
      error(
        "get_constraint_expression_trees could not retrieve the expression graph of constraint $j; only constraints declared with @NLconstraint are supported (original error: $e)",
      )
    end
    (expr.head == :call && length(expr.args) == 3) || error(
      "get_constraint_expression_trees does not know how to parse the expression of constraint $j: $expr",
    )
    func_expr = expr.args[2]
    ExpressionTreeForge.transform_to_expr_tree(func_expr)
  end
end

"""
    PartitionedConstraint{T, OB, GB}

Bundles the partitioned structure of a single partially-separable constraint function
`cⱼ(x) = ∑ᵢ ĉⱼᵢ(Uⱼᵢx)`. Fields:
* `j`: the row index of the constraint (1 ≤ `j` ≤ `ncon`);
* `variable_indices`: union of the elemental variables of every element function of `cⱼ`,
  i.e. the sparsity pattern of the `j`-th row of the Jacobian;
* `vec_elt_fun`: the `ElementFunction`s composing `cⱼ`;
* `objective_backend`: evaluates `cⱼ(x)`;
* `gradient_backend`: evaluates `∇cⱼ(x)`;
* `local_x`: a `PartitionedVector` scratch buffer over the `n` variables of the original
  problem, partitioned according to `cⱼ`'s own element functions. `gradient_backend`
  requires its input to share this exact partition (unlike `objective_backend`, which
  internally rebuilds a dense vector from whatever `PartitionedVector` it is given), so
  `local_x` must be `set!` from the evaluation point before calling `gradient_backend` on
  it; it is *not* the model's own `x` (built from the objective's element functions).
"""
struct PartitionedConstraint{T, OB <: PartitionedBackend{T}, GB <: PartitionedBackend{T}}
  j::Int
  variable_indices::Vector{Int}
  vec_elt_fun::Vector{ElementFunction}
  objective_backend::OB
  gradient_backend::GB
  local_x::PartitionedVector{T}
end

"""
    vec_cons = partitioned_constraints_structure(nlp::SupportedNLPModel, n::Int; type=Float64, merging=true, kwargs...)

Return a `Vector{PartitionedConstraint}`, decomposing every constraint function `cⱼ(x)` of
`nlp` as a partially-separable function, reusing the same element-function extraction
machinery as the objective (see [`partitioned_structure`](@ref)). Empty whenever
`nlp.meta.ncon == 0`.
"""
function partitioned_constraints_structure(
  nlp::SupportedNLPModel,
  n::Int;
  type::DataType = Float64,
  merging::Bool = true,
  kwargs...,
)
  ncon = nlp.meta.ncon
  ncon == 0 && return PartitionedConstraint[]
  cons_trees = get_constraint_expression_trees(nlp)
  kwargs =
    NamedTuple(k => v for (k, v) in pairs(kwargs) if k ∉ (:objectivebackend, :gradientbackend))
  return map(1:ncon) do j
    result = partitioned_structure(
      nlp,
      cons_trees[j],
      n;
      type,
      name = :phv,
      merging,
      objectivebackend = :moiobj,
      gradientbackend = :reverseelt,
      kwargs...,
    )
    vec_elt_fun_j = result[3]
    objective_backend_j = result[8]
    gradient_backend_j = result[9]
    local_x_j = result[10]
    variable_indices =
      sort!(unique!(vcat((elt_fun -> elt_fun.variable_indices).(vec_elt_fun_j)...)))
    PartitionedConstraint(
      j,
      variable_indices,
      vec_elt_fun_j,
      objective_backend_j,
      gradient_backend_j,
      local_x_j,
    )
  end
end
