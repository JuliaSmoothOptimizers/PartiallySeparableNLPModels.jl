module PartitionedBackends

using ReverseDiff, ForwardDiff
using MathOptInterface
using NLPModels
using ExpressionTreeForge, PartitionedVectors, PartitionedStructures
using ..ModAbstractPSNLPModels

export PartitionedBackend, AbstractObjectiveBackend, AbstractGradientBackend, AbstractHprodBackend
export objective, partitioned_gradient!, partitioned_hessian_prod!

const MOI = MathOptInterface

abstract type PartitionedBackend{T} end

abstract type AbstractObjectiveBackend{T} <: PartitionedBackend{T} end
abstract type AbstractGradientBackend{T} <: PartitionedBackend{T} end
abstract type AbstractHprodBackend{T} <: PartitionedBackend{T} end

"""
    fx = objective(backend::AbstractObjectiveBackend{T}, x::AbstractVector{T})

Compute the objective value from `backend` at the point `x`.
"""
objective(backend::PartitionedBackend{T}, x::AbstractVector{T}) where {T} =
  error("Objective interface not properly set: $(typeof(backend))")

"""
    partitioned_gradient!(backend::AbstractObjectiveBackend{T}, x::AbstractVector{T}, g::AbstractVector{T})

Compute the partitioned gradient from `backend` at the point `x` in place of `g`.
This method is designed for `PartitionedVector{T}<:AbstractVector{T}` (for now, both `x` and `g`).
"""
partitioned_gradient!(
  backend::PartitionedBackend{T},
  x::AbstractVector{T},
  g::AbstractVector{T},
) where {T} = error("Gradient interface not properly set: $(typeof(backend))")

"""
    partitioned_hessian_prod!(backend::AbstractHprodBackend{T}, x::AbstractVector{T}, v::AbstractVector{T}, Hv::AbstractVector{T})

Compute the partitioned Hessian-vector product ∇² f(x) v from `backend` in place of `Hv`.
This method is designed for `PartitionedVector{T}<:AbstractVector{T}` (`x`, `v` and `Hv`).
"""
partitioned_hessian_prod!(
  backend::PartitionedBackend{T},
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T},
) where {T} = error("Hessian-product interface not properly set: $(typeof(backend))")

include("ObjectiveBackends/NLPObjectiveBackend.jl")
include("ObjectiveBackends/MOIObjectiveBackend.jl")

include("GradientBackends/ElementReverseDiffGradient.jl")

include("GeneralBackends/ElementNonlinearModel.jl")
include("GeneralBackends/ModifiedNonlinearBackends.jl")
include("GeneralBackends/SparseJacobianBackend.jl")

include("HprodBackends/ElementReverseForwardHprod.jl")

end
