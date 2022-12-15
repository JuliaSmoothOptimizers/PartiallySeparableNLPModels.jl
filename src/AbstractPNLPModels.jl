module ModAbstractPSNLPModels

using Printf, Statistics, LinearAlgebra, FastClosures
using ADNLPModels, LinearOperators, NLPModels, NLPModelsJuMP, NLPModelsModifiers
using ExpressionTreeForge, PartitionedVectors

import Base.show

export AbstractPartiallySeparableNLPModel, AbstractPQNNLPModel, SupportedNLPModel
export ElementFunction
export update_nlp, hess_approx, update_nlp!

abstract type AbstractPartiallySeparableNLPModel{T, S} <: AbstractNLPModel{T, S} end
abstract type AbstractPQNNLPModel{T, S} <: QuasiNewtonModel{T, S} end

""" Accumulate the supported NLPModels. """
SupportedNLPModel = Union{ADNLPModel, MathOptNLPModel}

"""
    ElementFunction

A type that gathers the information indentifying an element function in a `PartiallySeparableNLPModel`, and its properties.
`ElementFunction` has fields:

* `i`: the index of the element function;
* `index_element_tree`: the index occupied in the element-function vector after the deletion of redundant element functions;
* `variable_indices`: list of elemental variables of `ElementFunction`;
* `type`: `constant`, `linear`, `quadratic`, `cubic` or `general`;
* `convexity_status`: `constant`, `linear`, `convex`, `concave` or `unknown`.
"""
mutable struct ElementFunction
  i::Int # the index of the function 1 ≤ i ≤ N
  index_element_tree::Int # 1 ≤ index_element_tree ≤ M
  variable_indices::Vector{Int} # ≈ Uᵢᴱ
  type::ExpressionTreeForge.Type_calculus_tree
  convexity_status::ExpressionTreeForge.M_implementation_convexity_type.Convexity_wrapper
end

include("common_methods.jl")
include("PQNNLPmethods.jl")
include("PSNLPmethods.jl")

show(psnlp::AbstractPartiallySeparableNLPModel) = show(stdout, psnlp)

function show(io::IO, psnlp::AbstractPartiallySeparableNLPModel)
  show(io, psnlp.nlp)
  println(io, "\nPartitioned structure summary:")
  n = get_n(psnlp)
  N = get_N(psnlp)
  M = get_M(psnlp)
  S = ["           element functions", "  distinct element functions"]
  V = [N, M]
  print(io, join(NLPModels.lines_of_hist(S, V), "\n"))

  @printf(io, "\n %20s:\n", "Element statistics")
  element_functions = psnlp.vec_elt_fun

  element_function_types = (elt_fun -> elt_fun.type).(element_functions)
  constant = count(is_constant, element_function_types)
  linear = count(is_linear, element_function_types)
  quadratic = count(is_quadratic, element_function_types)
  cubic = count(is_cubic, element_function_types)
  general = count(is_more, element_function_types)

  S1 = ["constant", "linear", "quadratic", "cubic", "general"]
  V1 = [constant, linear, quadratic, cubic, general]
  LH1 = NLPModels.lines_of_hist(S1, V1)

  element_function_convexity_status = (elt_fun -> elt_fun.convexity_status).(element_functions)
  convex = count(is_convex, element_function_convexity_status) - constant - linear
  concave = count(is_concave, element_function_convexity_status) - constant - linear
  general = count(is_unknown, element_function_convexity_status)

  S2 = ["convex", "concave", "general"]
  V2 = [convex, concave, general]
  LH2 = NLPModels.lines_of_hist(S2, V2)

  LH = map((i) -> LH1[i] * LH2[i], 1:3)
  push!(LH, LH1[4])
  push!(LH, LH1[5])
  print(io, join(LH, "\n"))

  @printf(io, "\n %28s: %s %28s: \n", "Element function dimensions", " "^12, "Variable overlaps")
  length_element_functions = (elt_fun -> length(elt_fun.variable_indices)).(element_functions)
  mean_length_element_functions = round(mean(length_element_functions), digits = 4)
  min_length_element_functions = minimum(length_element_functions)
  max_length_element_functions = maximum(length_element_functions)

  S1 = ["min", "mean", "max"]
  V1 = [min_length_element_functions, mean_length_element_functions, max_length_element_functions]
  LH1 = NLPModels.lines_of_hist(S1, V1)

  pv = psnlp.meta.x0.epv
  component_list = PartitionedStructures.get_component_list(pv)
  length_by_variable = (elt_list_var -> length(elt_list_var)).(component_list)
  mean_length_variable = round(mean(length_by_variable), digits = 4)
  min_length_variable = minimum(length_by_variable)
  max_length_variable = maximum(length_by_variable)
  S2 = ["min", "mean", "max"]
  V2 = [min_length_variable, mean_length_variable, max_length_variable]
  LH2 = NLPModels.lines_of_hist(S2, V2)

  LH = map((i, j) -> i * j, LH1, LH2)
  print(io, join(LH, "\n"))

  return nothing
end

show(psnlp::AbstractPQNNLPModel) = show(stdout, psnlp)

function show(io::IO, psnlp::AbstractPQNNLPModel)
  show(io, psnlp.model)
  println(io, "\nPartitioned structure summary:")
  n = get_n(psnlp)
  N = get_N(psnlp)
  M = get_M(psnlp)
  S = ["           element functions", "  distinct element functions"]
  V = [N, M]
  print(io, join(NLPModels.lines_of_hist(S, V), "\n"))

  @printf(io, "\n %20s:\n", "Element statistics")
  element_functions = psnlp.vec_elt_fun

  element_function_types = (elt_fun -> elt_fun.type).(element_functions)
  constant = count(is_constant, element_function_types)
  linear = count(is_linear, element_function_types)
  quadratic = count(is_quadratic, element_function_types)
  cubic = count(is_cubic, element_function_types)
  general = count(is_more, element_function_types)

  S1 = ["constant", "linear", "quadratic", "cubic", "general"]
  V1 = [constant, linear, quadratic, cubic, general]
  LH1 = NLPModels.lines_of_hist(S1, V1)

  element_function_convexity_status = (elt_fun -> elt_fun.convexity_status).(element_functions)
  convex = count(is_convex, element_function_convexity_status) - constant - linear
  concave = count(is_concave, element_function_convexity_status) - constant - linear
  general = count(is_unknown, element_function_convexity_status)

  S2 = ["convex", "concave", "general"]
  V2 = [convex, concave, general]
  LH2 = NLPModels.lines_of_hist(S2, V2)

  LH = map((i) -> LH1[i] * LH2[i], 1:3)
  push!(LH, LH1[4])
  push!(LH, LH1[5])
  print(io, join(LH, "\n"))

  @printf(io, "\n %28s: %s %28s: \n", "Element function dimensions", " "^12, "Variable overlaps")
  length_element_functions = (elt_fun -> length(elt_fun.variable_indices)).(element_functions)
  mean_length_element_functions = round(mean(length_element_functions), digits = 4)
  min_length_element_functions = minimum(length_element_functions)
  max_length_element_functions = maximum(length_element_functions)

  S1 = ["min", "mean", "max"]
  V1 = [min_length_element_functions, mean_length_element_functions, max_length_element_functions]
  LH1 = NLPModels.lines_of_hist(S1, V1)

  pv = psnlp.meta.x0.epv
  component_list = PartitionedStructures.get_component_list(pv)
  length_by_variable = (elt_list_var -> length(elt_list_var)).(component_list)
  mean_length_variable = round(mean(length_by_variable), digits = 4)
  min_length_variable = minimum(length_by_variable)
  max_length_variable = maximum(length_by_variable)
  S2 = ["min", "mean", "max"]
  V2 = [min_length_variable, mean_length_variable, max_length_variable]
  LH2 = NLPModels.lines_of_hist(S2, V2)

  LH = map((i, j) -> i * j, LH1, LH2)
  print(io, join(LH, "\n"))

  return nothing
end

end
