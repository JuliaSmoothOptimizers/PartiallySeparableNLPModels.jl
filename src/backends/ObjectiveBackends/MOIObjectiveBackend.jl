export MOIObjectiveBackend

"""
    MOIObjectiveBackend{T, Model}

Composed of `nlp::Model`, it evaluates the objective function from `NLPModels.obj(nlp, x::AbstractVector{T})`.
The user has to make sure `nlp` is can evaluate `x::AbstractVector{T}` with a suitable type `T`.
"""
mutable struct MOIObjectiveBackend{T} <: AbstractObjectiveBackend{T}
  evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}
  translated_x::PartitionedVector{T}
end

"""
    MOIObjectiveBackend(nlp::SupportedNLPModel; type=eltype(nlp.meta.x0))

Create an objective backend from `nlp`.
"""
function MOIObjectiveBackend(expr_tree::G,
  n::Int;
  elemental_variables = ExpressionTreeForge.get_elemental_variables(expr_tree),
  type=Type{T},
  kwargs...) where {T,G}
  _elemental_variables = reduce((x,y)-> unique!(sort!(vcat(x,y))), elemental_variables)
  translated_x = PartitionedVector([_elemental_variables]; n, simulate_vector=true)
  evaluator = ExpressionTreeForge.non_linear_JuMP_model_evaluator(expr_tree)
  MOIObjectiveBackend{type}(evaluator, translated_x) 
end

objective(backend::MOIObjectiveBackend{T}, x::Vector{T}) where {T} = MOI.eval_objective(backend.evaluator, x)

function objective(backend::MOIObjectiveBackend{T}, x::PartitionedVector{T}) where {T}
  PartitionedVectors.build!(x)
  PartitionedVectors.set!(backend.translated_x, x.epv.v)
  real_size_x = backend.translated_x[1].vec
  objective(backend, real_size_x)
end