module PartitionedBackends

using NLPModels, PartitionedVectors
using ..ModAbstractPSNLPModels

export AbstractObjectiveBackend, AbstractGradientBackend, AbstractHprodBackend
export objective, gradient!, hessianprod!

abstract type PartitionedBackend{T} end

abstract type AbstractObjectiveBackend{T} <: PartitionedBackend{T} end
abstract type AbstractGradientBackend{T} <: PartitionedBackend{T} end
abstract type AbstractHprodBackend{T} <: PartitionedBackend{T} end

objective(backend::AbstractObjectiveBackend{T}, x::AbstractVector{T}) where T = @error "Objective interface not properly set: $(typeof(backend))"
gradient!(backend::AbstractObjectiveBackend{T}, x::AbstractVector{T}) where T = @error "Gradient interface not properly set: $(typeof(backend))"
hessianprod!(backend::AbstractObjectiveBackend{T}, x::AbstractVector{T}) where T = @error "Hessian-product interface not properly set: $(typeof(backend))"

include("ObjectiveBackends/NLPObjectiveBackend.jl")

end