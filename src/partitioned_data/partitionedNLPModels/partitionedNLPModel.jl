module Mod_partitionedNLPModel

using ExpressionTreeForge
using ADNLPModels, NLPModels, NLPModelsJuMP
using JuMP, MathOptInterface, ModelingToolkit
using ..Mod_ab_partitioned_data
using ..Mod_PQN

export PQNNLPModel

abstract type PartitionedNLPModel{T, S} <: AbstractNLPModel{T, S} end
abstract type AbstractPQNNLPModel{T, S} <: PartitionedNLPModel{T, S} end

""" Accumulate the supported NLPModels. """
SupportedNLPModel = Union{ADNLPModel, MathOptNLPModel}

"""
    expr, n, x0 = get_expr_tree(adnlp::MathOptNLPModel; x0::Vector{T} = copy(adnlp.meta.x0), kwargs...) where {T <: Number}
    expr, n, x0 = get_expr_tree(adnlp::ADNLPModel; x0::Vector{T} = copy(adnlp.meta.x0), kwargs...) where {T <: Number}

Return for a `MathOptNLPModel` or a `ADNLPModel`: the expression tree of the objective function `expr::Expr`, the size of the problem `n` and the initial point `x0`.
"""
function get_expr_tree(
  nlp::MathOptNLPModel;
  x0::Vector{T} = copy(nlp.meta.x0),
  kwargs...,
) where {T <: Number}
  model = nlp.eval.m
  n = nlp.meta.nvar
  evaluator = JuMP.NLPEvaluator(model)
  MathOptInterface.initialize(evaluator, [:ExprGraph])
  obj_Expr = MathOptInterface.objective_expr(evaluator)::Expr
  ex = ExpressionTreeForge.transform_to_expr_tree(obj_Expr)::ExpressionTreeForge.Type_expr_tree
  return ex, n, x0
end

function get_expr_tree(
  adnlp::ADNLPModel;
  x0::Vector{T} = copy(adnlp.meta.x0),
  kwargs...,
) where {T <: Number}
  n = adnlp.meta.nvar
  ModelingToolkit.@variables x[1:n]
  fun = adnlp.f(x)
  ex = ExpressionTreeForge.transform_to_expr_tree(fun)::ExpressionTreeForge.Type_expr_tree
  return ex, n, x0
end

include("pqnNLPModel.jl")

"""
    f = obj(nlp, x)

Evaluate `f(x)`, the objective function of `nlp` at `x`.
"""
NLPModels.obj(nlp::P, x::AbstractVector{T}) where {P <: AbstractPQNNLPModel{T, S}} where {T, S} =
  evaluate_obj_part_data(nlp.part_data, x)

"""
    g = grad!(nlp, x, g)

Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
"""
function NLPModels.grad!(
  nlp::P,
  x::AbstractVector{T},
  g::AbstractVector{T},
) where {P <: AbstractPQNNLPModel{T, S}} where {T, S}
  evaluate_grad_part_data!(g, nlp.part_data, x)
  return g
end

end
