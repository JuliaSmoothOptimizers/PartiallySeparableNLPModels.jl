module ModPCSNLPModels

using ..Utils, ..Meta
using ..ModAbstractPSNLPModels
using ExpressionTreeForge, PartitionedStructures, PartitionedVectors
using ..PartitionedBackends
using NLPModels
using ReverseDiff

export PCSNLPModel

"""
    PCSNLPModel{G, T, S, M <: AbstractNLPModel{T, S}, Meta <: AbstractNLPModelMeta{T, S},} <: AbstractPQNNLPModel{T,S}

Deduct and allocate the partitioned structures of a NLPModel using a PCS Hessian approximation.
`PCSNLPModel` has fields:

* `model`: the original model;
* `meta`: gather information about the `PCSNLPModel`;
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
mutable struct PCSNLPModel{
  G,
  T,
  S,
  M <: AbstractNLPModel{T, Vector{T}},
  Meta <: AbstractNLPModelMeta{T, S},
  OB <: PartitionedBackend{T},
  GB <: PartitionedBackend{T},
} <: AbstractPQNNLPModel{T, S}
  model::M
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

  objective_backend::OB
  gradient_backend::GB

  op::PartitionedStructures.Elemental_pm{T} # partitioned quasi-Newton approximation

  fx::T
  name::Symbol
end

function PCSNLPModel(nlp::SupportedNLPModel; type::DataType = eltype(nlp.meta.x0), merging::Bool = true, kwargs...)
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
    objective_backend,
    gradient_backend,
    x,
    op,
    fx,
    name,
  ) = partitioned_structure(nlp, ex, n; type, name = :pcs, merging, kwargs...)

  meta = partitioned_meta(nlp.meta, x)
  Meta = typeof(meta)
  Model = typeof(nlp)
  S = typeof(x)
  OB = typeof(objective_backend)
  GB = typeof(gradient_backend)

  counters = NLPModels.Counters()
  pvqnlp = PCSNLPModel{ExpressionTreeForge.Complete_expr_tree, type, S, Model, Meta, OB, GB}(
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
    objective_backend,
    gradient_backend,
    op,
    fx,
    name,
  )
  return pvqnlp
end

end
