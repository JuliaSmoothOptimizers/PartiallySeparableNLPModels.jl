module Mod_partitionedNLPModel

using ExpressionTreeForge
using LinearOperators
using ADNLPModels, NLPModels, NLPModelsJuMP
using ..Mod_ab_partitioned_data
using ..Mod_PQN

export PartiallySeparableNLPModel
export update_nlp, hess_approx

abstract type AbstractPartiallySeparableNLPModel{T, S} <: AbstractNLPModel{T, S} end
abstract type AbstractPQNNLPModel{T,S} <: AbstractPartiallySeparableNLPModel{T, S} end

""" Accumulate the supported NLPModels. """
SupportedNLPModel = Union{ADNLPModel, MathOptNLPModel}

include("PartiallySeparableNLPModel.jl")

"""
    f = obj(nlp, x)

Evaluate `f(x)`, the objective function of `nlp` at `x`.
"""
function NLPModels.obj(
  nlp::P,
  x::AbstractVector{T},
) where {P <: AbstractPartiallySeparableNLPModel{T, S}} where {T, S}
  increment!(nlp, :neval_obj)
  evaluate_obj_part_data(nlp.part_data, x)
end

"""
    g = grad!(nlp, x, g)

Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
"""
function NLPModels.grad!(
  nlp::P,
  x::AbstractVector{T},
  g::AbstractVector{T},
) where {P <: AbstractPartiallySeparableNLPModel{T, S}} where {T, S}
  increment!(nlp, :neval_grad)
  evaluate_grad_part_data!(g, nlp.part_data, x)
  return g
end

"""
    B = hess_approx(nlp::PartiallySeparableNLPModel)

Return the Hessian approximation of `nlp`.
"""
hess_approx(nlp::PartiallySeparableNLPModel) = get_pB(nlp)

"""
    hprod!(nlp::PartiallySeparableNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight=1.)

Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function NLPModels.hprod!(
  nlp::PartiallySeparableNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = 1.0,
)
  increment!(nlp, :neval_hprod)
  hprod!(nlp.part_data, x, obj_weight .* v, Hv)
end

LinearOperators.LinearOperator(nlp::PartiallySeparableNLPModel) = LinearOperator(nlp.part_data)

show(psnlp::PartiallySeparableNLPModel) = show(stdout, psnlp)

function show(io::IO, psnlp::PartiallySeparableNLPModel)
  show(io, psnlp.nlp)
  show(io, psnlp.part_data)
  return nothing
end

end
