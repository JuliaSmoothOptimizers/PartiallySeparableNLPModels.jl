module ModPSNLPModels

using ..Utils, ..Meta
using ..ModAbstractPSNLPModels
using ExpressionTreeForge, PartitionedStructures, PartitionedVectors
using NLPModels, MathOptInterface
using ReverseDiff

export PSNLPModel

"""
    PSNLPModel{G, P, T, S, M <: AbstractNLPModel{T, S}, Meta <: AbstractNLPModelMeta{T, S},} <: AbstractPQNNLPModel{T,S}

Deduct and allocate the partitioned structures of a NLPModel using partitioned hessian-vector product.
`PSNLPModel` has fields:

* `model`: the original model;
* `meta`: gather information about the `PSNLPModel`;
* `counters`: count how many standards methods of `NLPModels` are called;
* `n`: the size of the problem;
* `N`: the number of element functions;
* `vec_elt_fun`: a `ElementFunction` vector, of size `N`;
* `M`: the number of distinct element-function expression trees;
* `vec_elt_complete_expr_tree`: a `Complete_expr_tree` vector, of size `M`;
* `element_expr_tree_table`: a vector of size `M`, the i-th element `element_expr_tree_table[i]::Vector{Int}` informs which element functions use the `vec_elt_complete_expr_tree[i]` expression tree;
* `index_element_tree`: a vector of size `N` where each component indicates which `Complete_expr_tree` from `vec_elt_complete_expr_tree` is used for the corresponding element;
* `vec_compiled_element_gradients`: the vector gathering the compiled tapes for every element gradient evaluations;
* `op`: the partitioned matrix (main memory cost);
* `name`: the name of partitioned quasi-Newton update performed
"""
mutable struct PSNLPModel{
  G,
  P,
  T,
  S,
  M <: AbstractNLPModel{T, Vector{T}},
  Meta <: AbstractNLPModelMeta{T, S},
} <: AbstractPartiallySeparableNLPModel{T, S}
  nlp::M
  meta::Meta
  counters::NLPModels.Counters

  n::Int
  N::Int
  vec_elt_fun::Vector{ElementFunction} #length(vec_elt_fun) == N
  # Vector composed by the expression trees of element functions .
  # Warning: Several element functions may have the same expression tree
  M::Int
  vec_elt_complete_expr_tree::Vector{G} # length(element_expr_tree) == M < N
  # element_expr_tree_table store the indices of every element function using each element_expr_tree, ∀i,j, 1 ≤ element_expr_tree_table[i][j] \leq N
  element_expr_tree_table::Vector{Vector{Int}} # length(element_expr_tree_table) == M
  index_element_tree::Vector{Int} # length(index_element_tree) == N, index_element_tree[i] ≤ M

  objective_evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}
  modified_objective_evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}
  x_modified::Vector{T}
  v_modified::Vector{T}

  # g is build directly from pg
  # the result of pB*v will be store and build from pv
  # name is the name of the partitioned quasi-Newton applied on pB
  name::Symbol
end

function PSNLPModel(nlp::SupportedNLPModel; type::DataType = Float64, merging::Bool = true)
  n = nlp.meta.nvar
  ex = get_expression_tree(nlp)

  (
    n,
    N,
    vec_elt_fun,
    M,
    vec_elt_complete_expr_tree,
    element_expr_tree_table,
    index_element_tree,
    objective_evaluator,
    modified_objective_evaluator,
    x_modified,
    v_modified,
    x,
    pB,
    fx,
    name,
  ) = partitioned_structure(ex, n; type, name = :phv, merging)
  P = typeof(pB)

  meta = partitioned_meta(nlp.meta, x)
  Meta = typeof(meta)
  Model = typeof(nlp)
  S = typeof(x)

  counters = NLPModels.Counters()
  pvqnlp = PSNLPModel{ExpressionTreeForge.Complete_expr_tree, P, type, S, Model, Meta}(
    nlp,
    meta,
    counters,
    n,
    N,
    vec_elt_fun,
    M,
    vec_elt_complete_expr_tree,
    element_expr_tree_table,
    index_element_tree,
    objective_evaluator,
    modified_objective_evaluator,
    x_modified,
    v_modified,
    name,
  )
  return pvqnlp
end

end
