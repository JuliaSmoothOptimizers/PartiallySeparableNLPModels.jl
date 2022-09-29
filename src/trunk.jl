module TrunkInterface

using JSOSolvers, Krylov, NLPModels, SolverTools
using ..ModAbstractPSNLPModels

function JSOSolvers.TrunkSolver(
  nlp::AbstractPartiallySeparableNLPModel{T, V};
  subsolver_type::Type{<:KrylovSolver} = CgSolver,
) where {T, V <: AbstractVector{T}}
  nvar = nlp.meta.nvar
  x = similar(nlp.meta.x0)
  xt = similar(nlp.meta.x0)
  gx = similar(nlp.meta.x0; simulate_vector=false)
  gt = similar(nlp.meta.x0; simulate_vector=false)
  gn = isa(nlp, AbstractPQNNLPModel) ? similar(nlp.meta.x0; simulate_vector=false) : PartitionedVector([Int[]])
  Hs = similar(nlp.meta.x0; simulate_vector=false)
  subsolver = subsolver_type(x)
  Sub = typeof(subsolver)
  H = NLPModels.hess_op!(nlp, x, Hs)
  Op = typeof(H)
  tr = TrustRegion(gt, one(T))
  return TrunkSolver{T, V, Sub, Op}(x, xt, gx, gt, gn, Hs, subsolver, H, tr)
end

end