module ModPVQNPModels

using ..Utils, ..Meta
using ..ModAbstractPSNLPModels
using ExpressionTreeForge, PartitionedStructures, PartitionedVectors
using NLPModels
using ReverseDiff

export PVQNPModel

"""
    PVQNPModel{G, P, T, S, M <: AbstractNLPModel{T, S}, Meta <: AbstractNLPModelMeta{T, S},} <: AbstractPQNNLPModel{T,S}

Deduct and allocate the partitioned structures of a NLPModel using partitioned BFGS Hessian approximation.
`PVQNPModel` has fields:

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
mutable struct PVQNPModel{G, P, T, S, M <: AbstractNLPModel{T, Vector{T}}, Meta <: AbstractNLPModelMeta{T, S},} <: AbstractPQNNLPModel{T,S}
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

  # x::Vector{T} # length(x)==n
  # v::Vector{T} # length(v)==n
  # s::Vector{T} # length(v)==n
  x::PartitionedVector{T}
  
  # pg::PartitionedStructures.Elemental_pv{T} # partitioned gradient
  # pv::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  # py::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  # ps::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  # phv::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  pB::P # partitioned B

  fx::T
  # g is build directly from pg
  # the result of pB*v will be store and build from pv
  # name is the name of the partitioned quasi-Newton applied on pB
  name::Symbol
end 

function PVQNPModel(nlp::SupportedNLPModel; type::DataType=Float64)
  n = nlp.meta.nvar
  ex = get_expression_tree(nlp)

  (n, N, vec_elt_fun, M, vec_elt_complete_expr_tree, element_expr_tree_table, index_element_tree, vec_compiled_element_gradients, x, pB, fx, name) = partitioned_structure(ex, n; type, name=:pbfgs)
  P = typeof(pB)

  meta = partitioned_meta(nlp.meta, x)
  Meta = typeof(meta)
  Model = typeof(nlp)
  S = typeof(x)

  counters = NLPModels.Counters()
  pvqnlp = PVQNPModel{ExpressionTreeForge.Complete_expr_tree, P, type, S, Model, Meta}(nlp, meta, counters, n, N, vec_elt_fun, M, vec_elt_complete_expr_tree, element_expr_tree_table, index_element_tree, vec_compiled_element_gradients, x, pB, fx, name)
  return pvqnlp
end
  
end


# mutable struct TrunkSolver{
#   T,
#   V <: AbstractVector{T},
#   Sub <: KrylovSolver{T, T, V},
#   Op <: AbstractLinearOperator{T},
# } <: AbstractOptimizationSolver
#   x::V
#   xt::V
#   gx::V
#   gt::V
#   gn::V
#   Hs::V
#   subsolver::Sub
#   H::Op
#   tr::TrustRegion{T, V}
# end

# function TrunkSolver(
#   nlp::AbstractNLPModel{T, V};
#   subsolver_type::Type{<:KrylovSolver} = CgSolver,
# ) where {T, V <: AbstractVector{T}}
#   nvar = nlp.meta.nvar
#   x = V(undef, nvar)
#   xt = V(undef, nvar)
#   gx = V(undef, nvar)
#   gt = V(undef, nvar)
#   gn = isa(nlp, QuasiNewtonModel) ? V(undef, nvar) : V(undef, 0)
#   Hs = V(undef, nvar)
#   subsolver = subsolver_type(nvar, nvar, V)
#   Sub = typeof(subsolver)
#   H = hess_op!(nlp, x, Hs)
#   Op = typeof(H)
#   tr = TrustRegion(gt, one(T))
#   return TrunkSolver{T, V, Sub, Op}(x, xt, gx, gt, gn, Hs, subsolver, H, tr)
# end


# function NLPModelMeta{T, S}(
#   nvar::Int;
#   x0::S = fill!(S(undef, nvar), zero(T)),
#   lvar::S = fill!(S(undef, nvar), T(-Inf)),
#   uvar::S = fill!(S(undef, nvar), T(Inf)), 
#   nlvb = nvar,
#   nlvo = nvar,
#   nlvc = nvar,
#   ncon = 0,
#   y0::S = fill!(S(undef, ncon), zero(T)),
#   lcon::S = fill!(S(undef, nstcon), T(-Inf)),
#   ucon::S = fill!(S(undef, ncon), T(Inf)),
#   nnzo = nvar,
#   nnzj = nvar * ncon,
#   lin_nnzj = 0,
#   nln_nnzj = nvar * ncon,
#   nnzh = nvar * (nvar + 1) / 2,
#   lin = Int[],
#   minimize = true,
#   islp = false,
#   name = "Generic",
# ) where {T, S}