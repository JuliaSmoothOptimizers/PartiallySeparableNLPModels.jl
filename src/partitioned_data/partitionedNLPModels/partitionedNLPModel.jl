module Mod_partitionedNLPModel

using ExpressionTreeForge
using ADNLPModels, NLPModels, NLPModelsJuMP
using JuMP, MathOptInterface, ModelingToolkit
using ..Mod_ab_partitioned_data
using ..Mod_PQN

export PartiallySeparableNLPModel
export update_nlp, hess_approx

abstract type AbstractPartiallySeparableNLPModel{T, S} <: AbstractNLPModel{T, S} end

""" Accumulate the supported NLPModels. """
SupportedNLPModel = Union{ADNLPModel, MathOptNLPModel}

"""
    expr, n, x0 = get_expr_tree(adnlp::MathOptNLPModel; x0::Vector{T} = copy(adnlp.meta.x0), kwargs...) where {T <: Number}
    expr, n, x0 = get_expr_tree(adnlp::ADNLPModel; x0::Vector{T} = copy(adnlp.meta.x0), kwargs...) where {T <: Number}

Return for a `MathOptNLPModel` or a `ADNLPModel`: the expression tree `expr::Expr` of the objective function, the size of the problem `n` and the initial point `x0`.
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
    Hx = hess(nlp, x; obj_weight=1.0)

Evaluate the objective Hessian at `x` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,
"""
function NLPModels.hess(
  nlp::PartiallySeparableNLPModel,
  x::AbstractVector;
  obj_weight=1.
)
  increment!(nlp, :neval_hess)
  sp_hess = hess(nlp.part_data, x)
  return obj_weight .* sp_hess
end

"""
    hprod(nlp, x, v; obj_weight=1.0)

Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function NLPModels.hprod!(
  nlp::PartiallySeparableNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight=1.
)
  increment!(nlp, :neval_hprod)
  hprod!(nlp.part_data, x, obj_weight .* v, Hv)
end

end
