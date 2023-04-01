export MOIObjectiveBackend

"""
    MOIObjectiveBackend{T, Model}

Composed of `nlp::Model`, it evaluates the objective function from `NLPModels.obj(nlp, x::AbstractVector{T})`.
The user has to make sure `nlp` is can evaluate `x::AbstractVector{T}` with a suitable type `T`.
"""
mutable struct MOIObjectiveBackend{T} <: AbstractObjectiveBackend{T}
  evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}
end

"""
    MOIObjectiveBackend(nlp::SupportedNLPModel; type=eltype(nlp.meta.x0))

Create an objective backend from `nlp`.
"""
function MOIObjectiveBackend(expr_tree::G; type=Float64, kwargs...) where G
  evaluator = ExpressionTreeForge.non_linear_JuMP_model_evaluator(expr_tree)
  MOIObjectiveBackend{type}(evaluator) 
end

objective(backend::MOIObjectiveBackend{T}, x::Vector{T}) where {T} = MOI.eval_objective(backend.evaluator, x)

function objective(backend::MOIObjectiveBackend{T}, x::PartitionedVector{T}) where {T}
  PartitionedVectors.build!(x)
  objective(backend, x.epv.v)
end