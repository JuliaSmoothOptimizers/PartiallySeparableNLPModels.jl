module TrunkInterface

using JSOSolvers, Krylov, NLPModels, SolverTools
using PartitionedVectors
using ..ModAbstractPSNLPModels

function JSOSolvers.TrunkSolver(
  nlp::AbstractPartiallySeparableNLPModel{T, S};
  subsolver_type::Type{<:KrylovSolver} = CgSolver,
) where {T, S <: AbstractVector{T}}
  nvar = nlp.meta.nvar
  x = similar(nlp.meta.x0)
  x .= 0
  xt = similar(x)
  xt .= 0
  gx = similar(x; simulate_vector = false)
  gx .= 0
  gt = similar(x; simulate_vector = false)
  gt .= 0
  gn =
    isa(nlp, AbstractPQNNLPModel) ? similar(x; simulate_vector = false) : PartitionedVector([Int[]])
  gn .= 0
  Hs = similar(x; simulate_vector = false)
  Hs .= 0
  subsolver = subsolver_type(x)
  Sub = typeof(subsolver)
  H = NLPModels.hess_op!(nlp, x, Hs)
  Op = typeof(H)
  tr = TrustRegion(gt, one(T))
  return TrunkSolver{T, S, Sub, Op}(x, xt, gx, gt, gn, Hs, subsolver, H, tr)
end

function JSOSolvers.TrunkSolver(
  nlp::AbstractPQNNLPModel{T, S};
  subsolver_type::Type{<:KrylovSolver} = CgSolver,
) where {T, S <: AbstractVector{T}}
  nvar = nlp.meta.nvar
  x = similar(nlp.meta.x0)
  x .= 0
  xt = similar(x)
  xt .= 0
  gx = similar(x; simulate_vector = false)
  gx .= 0
  gt = similar(x; simulate_vector = false)
  gt .= 0
  gn =
    isa(nlp, AbstractPQNNLPModel) ? similar(x; simulate_vector = false) : PartitionedVector([Int[]])
  gn .= 0
  Hs = similar(x; simulate_vector = false)
  Hs .= 0
  subsolver = subsolver_type(x)
  Sub = typeof(subsolver)
  H = NLPModels.hess_op!(nlp, x, Hs)
  Op = typeof(H)
  tr = TrustRegion(gt, one(T))
  return TrunkSolver{T, S, Sub, Op}(x, xt, gx, gt, gn, Hs, subsolver, H, tr)
end

end
