module PartitionedBackends

using ReverseDiff, ForwardDiff
using NLPModels
using ExpressionTreeForge, PartitionedVectors, PartitionedStructures
using ..ModAbstractPSNLPModels

export AbstractObjectiveBackend, AbstractGradientBackend, AbstractHprodBackend
export objective, partitioned_gradient, partitioned_hessian_prod!

abstract type PartitionedBackend{T} end

abstract type AbstractObjectiveBackend{T} <: PartitionedBackend{T} end
abstract type AbstractGradientBackend{T} <: PartitionedBackend{T} end
abstract type AbstractHprodBackend{T} <: PartitionedBackend{T} end

objective(backend::AbstractObjectiveBackend{T}, x::AbstractVector{T}) where T = @error "Objective interface not properly set: $(typeof(backend))"
partitioned_gradient(backend::AbstractObjectiveBackend{T}, x::AbstractVector{T}, g::AbstractVector{T}) where T = @error "Gradient interface not properly set: $(typeof(backend))"
partitioned_hessian_prod!(backend::AbstractObjectiveBackend{T}, x::AbstractVector{T}, v::AbstractVector{T}, Hv::AbstractVector{T}) where T = @error "Hessian-product interface not properly set: $(typeof(backend))"

include("ObjectiveBackends/NLPObjectiveBackend.jl")
include("GradientBackends/ElementReverseDiffGradient.jl")
include("HprodBackends/ElementReverseForwardHprod.jl")

end