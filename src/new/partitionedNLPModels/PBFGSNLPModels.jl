module ModPBFGSNLPModels

using ..Utils
using ..ModAbstractPSNLPModels
using ExpressionTreeForge, PartitionedStructures
using NLPModels
using ReverseDiff

export PBFGSNLPModel

"""
    PBFGSNLPModel{G, P, T, S, M <: AbstractNLPModel{T, S}, Meta <: AbstractNLPModelMeta{T, S},} <: AbstractPQNNLPModel{T,S}

Deduct and allocate the partitioned structures of a NLPModel using partitioned BFGS Hessian approximation.
`PBFGSNLPModel` has fields:

* `nlp`: the original model;
* `meta`: gather information about the `PartiallySeparableNLPModel`;
* `counters`: count how many standards methods of `NLPModels` are called;
* `n`: the size of the problem;
* `N`: the number of element functions;
* `vec_elt_fun`: a `ElementFunction` vector, of size `N`;
* `M`: the number of distinct element-function expression trees;
* `vec_elt_complete_expr_tree`: a `Complete_expr_tree` vector, of size `M`;
* `element_expr_tree_table`: a vector of size `M`, the i-th element `element_expr_tree_table[i]::Vector{Int}` informs which element functions use the `vec_elt_complete_expr_tree[i]` expression tree;
* `index_element_tree`: a vector of size `N` where each component indicates which `Complete_expr_tree` from `vec_elt_complete_expr_tree` is used for the corresponding element;
* `vec_compiled_element_gradients`: the vector gathering the compiled tapes for every element gradient evaluations;
* `x`: the current point;
* `v`: a temporary vector;
* `s`: the current step;
* `pg`: the partitioned gradient;
* `pv`: a temporary partitioned vector;
* `py`: the partitioned gradient difference;
* `ps`: the partitioned step;
* `phv`: the partitioned Hessian-vector product;
* `pB`: the partitioned matrix (main memory cost);
* `fx`: the current value of the objective function;
* `name`: the name of partitioned quasi-Newton update performed
"""
mutable struct PBFGSNLPModel{G, P, T, S, M <: AbstractNLPModel{T, S}, Meta <: AbstractNLPModelMeta{T, S},} <: AbstractPQNNLPModel{T,S}
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

  vec_compiled_element_gradients::Vector{ReverseDiff.CompiledTape}

  x::Vector{T} # length(x)==n
  v::Vector{T} # length(v)==n
  s::Vector{T} # length(v)==n
  pg::PartitionedStructures.Elemental_pv{T} # partitioned gradient
  pv::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  py::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  ps::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  phv::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  pB::P # partitioned B

  fx::T
  # g is build directly from pg
  # the result of pB*v will be store and build from pv
  # name is the name of the partitioned quasi-Newton applied on pB
  name::Symbol
end 


function PBFGSNLPModel(nlp::SupportedNLPModel)
  n = nlp.meta.nvar
  x0 = nlp.meta.x0
  ex = get_expression_tree(nlp)
  T = eltype(x0)

  (n, N, vec_elt_fun, M, vec_elt_complete_expr_tree, element_expr_tree_table, index_element_tree, vec_compiled_element_gradients, x, v, s, pg, pv, py, ps, phv, pB, fx, name) = partially_separable_structure(ex, n; name=:pbfgs, x0)
  P = typeof(pB)

  meta = nlp.meta
  Meta = typeof(meta)
  Model = typeof(nlp)

  counters = NLPModels.Counters()
  S = typeof(x)
  pbfgsnlp = PBFGSNLPModel{ExpressionTreeForge.Complete_expr_tree, P, T, S, Model, Meta}(nlp, meta, counters, n, N, vec_elt_fun, M, vec_elt_complete_expr_tree, element_expr_tree_table, index_element_tree, vec_compiled_element_gradients, x, v, s, pg, pv, py, ps, phv, pB, fx, name)
  return pbfgsnlp
end
  
end