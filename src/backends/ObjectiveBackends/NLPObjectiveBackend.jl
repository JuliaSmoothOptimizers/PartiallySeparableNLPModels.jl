export NLPObjectiveBackend

"""
    NLPObjectiveBackend{T, Model}

Composed of `nlp::Model`, it evaluates the objective function from `NLPModels.obj(nlp, x::AbstractVector{T})`.
The user has to make sure `nlp` can evaluate `x::AbstractVector{T}` with a suitable type `T`.
"""
mutable struct NLPObjectiveBackend{T, Model} <: AbstractObjectiveBackend{T}
  nlp::Model # ADNLPModel or MathOptNLPModel
end

"""
    NLPObjectiveBackend(nlp::SupportedNLPModel; type=eltype(nlp.meta.x0))

Create an objective backend from `nlp`.
"""
NLPObjectiveBackend(
  nlp::SupportedNLPModel;
  type::Type{T} = eltype(nlp.meta.x0),
  kwargs...,
) where {T} = NLPObjectiveBackend{type, typeof(nlp)}(nlp)

objective(backend::NLPObjectiveBackend{T, Model}, x::Vector{T}) where {T, Model} =
  NLPModels.obj(backend.nlp, x)

function objective(backend::NLPObjectiveBackend{T, Model}, x::PartitionedVector{T}) where {T, Model}
  PartitionedVectors.build!(x)
  NLPModels.obj(backend.nlp, x.epv.v)
end
