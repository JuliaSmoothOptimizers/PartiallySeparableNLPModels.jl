module ModAbstractPSNLPModels

using Printf, Statistics, LinearAlgebra, FastClosures
using ADNLPModels, LinearOperators, NLPModels, NLPModelsJuMP
using ExpressionTreeForge

import Base.show

export AbstractPartiallySeparableNLPModel, AbstractPQNNLPModel, SupportedNLPModel
export ElementFunction
export update_nlp, hess_approx, update_nlp!

abstract type AbstractPartiallySeparableNLPModel{T, S} <: AbstractNLPModel{T, S} end
abstract type AbstractPQNNLPModel{T,S} <: AbstractPartiallySeparableNLPModel{T, S} end

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

# NLPModels interface for AbstractPartiallySeparableNLPModel
"""
    f = obj(nlp, x)

Evaluate `f(x)`, the objective function of `nlp` at `x`.
"""
function NLPModels.obj(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S, # PartitionedVector
) where {T, S<:AbstractVector{T}} 
  increment!(psnlp, :neval_obj)
  epv = x.epv
  index_element_tree = get_index_element_tree(psnlp)
  N = get_N(psnlp)
  f = 0
  for i = 1:N
    elt_expr_tree = get_vec_elt_complete_expr_tree(psnlp, index_element_tree[i])
    fᵢx = ExpressionTreeForge.evaluate_expr_tree( 
      elt_expr_tree,
      PartitionedStructures.get_eev_value(epv, i), # i-th element
    )
    f += fᵢx
  end  
  return f
end

"""
    g = grad(nlp, x)

Evaluate `∇f(x)`, the gradient of the objective function at `x`.
"""
function NLPModels.grad(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S, # PartitionedVector  
) where {T, S<:AbstractVector{T}} 
  g = similar(x; simulate_vector=false)
  grad!(psnlp, x, g)
  return g
end

"""
    g = grad!(nlp, x, g)

Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
"""
function NLPModels.grad!(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S, # PartitionedVector
  g::S, # PartitionedVector
) where {T, S<:AbstractVector{T}} 
  increment!(psnlp, :neval_grad)  
  epv_x = x.epv
  epv_g = g.epv
  index_element_tree = get_index_element_tree(psnlp)
  N = get_N(psnlp)
  for i = 1:N
    compiled_tape = get_vec_compiled_element_gradients(psnlp, index_element_tree[i])
    Uix = PartitionedStructures.get_eev_value(epv_x, i)
    gi = PartitionedStructures.get_eev_value(epv_g, i)
    ReverseDiff.gradient!(gi, compiled_tape, Uix)
  end
  return g
end

"""
    hprod!(nlp::AbstractPartiallySeparableNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight=1.)

Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function NLPModels.hprod(
  psnlp::AbstractPartiallySeparableNLPModel{T,S},
  x::S,
  v::S;
  obj_weight = 1.0,
  β = 0.0,
) where {T, S<:AbstractVector{T}} 
  Hv = similar(x; simulate_vector=false)
  NLPModels.hprod!(psnlp, x, v, Hv; obj_weight, β)
  return Hv
end 

"""
    hprod!(nlp::AbstractPartiallySeparableNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight=1.)

Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function NLPModels.hprod!(
  psnlp::AbstractPartiallySeparableNLPModel{T,S},
  x::S,
  v::S,
  Hv::S;
  obj_weight = 1.0,
  β = 0.0,
) where {T, S<:AbstractVector{T}} 
  increment!(psnlp, :neval_hprod)
  epv_x = x.epv
  epv_v = v.epv
  epv_Hv = Hv.epv

  index_element_tree = get_index_element_tree(psnlp)
  N = get_N(psnlp)
  ∇f(x; f) = ReverseDiff.gradient(f, x)
  ∇²fv!(x, v, Hv; f) = ForwardDiff.derivative!(Hv, t -> ∇f(x + t * v; f), 0)

  for i = 1:N
    complete_tree = get_vec_elt_complete_expr_tree(psnlp, index_element_tree[i])
    elf_fun = ExpressionTreeForge.evaluate_expr_tree(complete_tree)

    Uix = PartitionedStructures.get_eev_value(epv_x, i)
    Uiv = PartitionedStructures.get_eev_value(epv_v, i)
    Hvi = PartitionedStructures.get_eev_value(epv_Hv, i)
    ∇²fv!(Uix, Uiv, Hvi; f = elf_fun)
  end
  return Hv 
end

function NLPModels.hprod!(
  pqnnlp::AbstractPQNNLPModel{T,S},
  x::S,
  v::S,
  Hv::S;
  obj_weight = 1.0,
) where {T, S<:AbstractVector{T}} 
  increment!(pqnnlp, :neval_hprod)
  epv_Hv = Hv.epv
  epv_v = v.epv
  pB = get_pB(pqnnlp)
  mul_epm_epv!(epv_Hv, pB, epv_v)
  Hv .*= obj_weight
  return Hv
end

# function NLPModels.hess_op(ps_nlp::AbstractPartiallySeparableNLPModel{T,S}, x::AbstractVector; kwargs...) where {T, S}
#   n = get_n(ps_nlp)
#   B = LinearOperator(T, n, n, true, true, (res, v, α, β) -> NLPModels.hprod!(ps_nlp, x, v, res; obj_weight=α, β))
#   return B
# end
 
function NLPModels.hess_op(
  pqnnlp::AbstractPartiallySeparableNLPModel{T,S},
  x::S;
  obj_weight = 1.0,
) where {T, S<:AbstractVector{T}} 
  Hv = similar(x; simulate_vector=false)
  return hess_op!(pqnnlp, x, Hv; obj_weight)
end

function NLPModels.hess_op!(
  pqnnlp::AbstractPartiallySeparableNLPModel{T,S},
  x::S,
  Hv::S;
  obj_weight = 1.0,
) where {T, S<:AbstractVector{T}} 
  n = get_n(pqnnlp)
  prod! = @closure (res, v, α, β) -> begin
    # res.simulate_vector = false
    hprod!(pqnnlp, x, v, Hv; obj_weight = obj_weight)
    if β == 0
      @. res = α * Hv
    else
      @. res = α * Hv + β * res
    end
  end
  B = LinearOperator(T, n, n, true, true, prod!)
  return B
end

# NLPModels interface for AbstractPQNNLPModel

"""
    B = hess_approx(nlp::AbstractPQNNLPModel)

Return the Hessian approximation of `nlp`.
"""
hess_approx(pqn_nlp::AbstractPQNNLPModel) = get_pB(pqn_nlp)

# NLPModels.hess_op(pqnnlp::AbstractPQNNLPModel{T,S}, x::S; kwargs...) = LinearOperator(pqnnlp)

# """
#     partitionedMulOp!(pqn_nlp::AbstractPQNNLPModel, res, v, α, β)

# Partitioned 5-arg `mul!` for `pqn_nlp` using the partitioned matrix and partitioned vectors to destribute and collect the result of element matrix-vector products.
# """
# function partitionedMulOp!(pqnnlp::AbstractPQNNLPModel{T,S}, res::S, v::S, α, β) where {T, S<:AbstractVector{T}} 
#   epv = v.epv
#   epv_res = res.epv
#   pB = get_pB(pqnnlp)
#   mul_epm_epv!(epv_res, pB, epv)
#   if β == 0
#     @. res = α * res
#   else
#     @. res = α * Hv + β * res
#   end
#   return res
#   mul!(res, I, PartitionedStructures.get_v(epv_res), α, β)
#   return epv_res
# end

# function LinearOperators.LinearOperator(pqn_nlp::AbstractPQNNLPModel{T,S}) where {T, S}
#   n = get_n(pqn_nlp)
#   B = LinearOperator(T, n, n, true, true, (res, v, α, β) -> partitionedMulOp!(pqn_nlp, res, v, α, β))
#   return B
# end

function Base.push!(
  pqn_nlp::AbstractPQNNLPModel{T,S},
  s::S,
  y::S;
  kwargs...,
) where {T, S<:AbstractVector{T}} 
  epv_s = s.epv
  epv_y = y.epv
  pB = get_pB(pqn_nlp)
  PartitionedStructures.update!(pB, epv_y, epv_s; name = pqn_nlp.name, kwargs...)
  return pB
end

# """
#     update_nlp(pqn_nlp::AbstractPQNNLPModel{T,S}, x::Vector{T}, s::Vector{T})

# Perform the partitioned quasi-Newton update given the current point `x` and the step `s`.
# When `x` is omitted, `update_PQN!` consider that `pqn_nlp` has the current point in pqn_nlp.x`.
# Moreover, it assumes that the partitioned gradient at `x` is already computed in `pqn_nlp.pg`.
# Will be replace by `push!` when `PartitionedVector` are implemented.
# """
# function update_nlp(
#   pqn_nlp::AbstractPQNNLPModel{T,S},
#   x::Vector{T},
#   s::Vector{T};
#   kwargs...,
# ) where {T <: Number, S}
#   update_nlp!(pqn_nlp, x, s; kwargs...)
#   return Matrix(get_pB(pqn_nlp))
# end

# """
#     update_nlp!(pqn_nlp::AbstractPQNNLPModel{T,S}, s::Vector{T})
#     update_nlp!(pqn_nlp::AbstractPQNNLPModel{T,S}, x::Vector{T}, s::Vector{T})

# Perform the partitioned quasi-Newton update given the current point `x` and the step `s`.
# When `x` is omitted, `update_PQN!` consider that `pqn_nlp` has the current point in pqn_nlp.x`.
# Moreover, it assumes that the partitioned gradient at `x` is already computed in `pqn_nlp.pg`.
# Will be replace by `push!` when `PartitionedVector` are implemented.
# """
# function update_nlp!(
#   pqn_nlp::AbstractPQNNLPModel{T,S},
#   x::Vector{T},
#   s::Vector{T};
#   kwargs...,
# ) where {T <: Number, S}
#   set_x!(pqn_nlp, x)
#   evaluate_grad_part_data!(pqn_nlp)
#   update_nlp!(pqn_nlp, s; kwargs...)
# end

# function update_nlp!(
#   pqn_nlp::AbstractPQNNLPModel{T,S},
#   s::Vector{T};
#   reset = 0,
#   kwargs...,
# ) where {T <: Number, S}
#   evaluate_y_part_data!(pqn_nlp, s)
#   py = get_py(pqn_nlp)
#   set_ps!(pqn_nlp, s)
#   ps = get_ps(pqn_nlp)
#   pB = get_pB(pqn_nlp)
#   PartitionedStructures.update!(pB, py, ps; name = pqn_nlp.name, kwargs...)
#   return pqn_nlp
# end


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

end
