export ElementReverseDiffGradient

_ni(element_variables::Vector{Int}) = isempty(element_variables) ? 0 : maximum(element_variables)

"""
    element_gradient_tape = compiled_grad_element_function(element_function::T; ni::Int = length(ExpressionTreeForge.get_elemental_variables(element_function)), type = Float64) where {T}

Return the `element_gradient_tape::GradientTape` to compute the gradient computation of an `element_function` with `ReverseDiff`.
"""
function compiled_grad_element_function(
  element_function::G;
  element_variables::Vector{Int} = ExpressionTreeForge.get_elemental_variables(element_function),
  ni::Int = _ni(element_variables),
  type::Type{T} = Float64,
) where {T, G}
  f = ExpressionTreeForge.evaluate_expr_tree(element_function)
  f_tape = ReverseDiff.GradientTape(f, rand(type, ni))
  compiled_f_tape = ReverseDiff.compile(f_tape)
  return compiled_f_tape
end

"""
    ElementReverseDiffGradient{T}

Composed of:
- `vec_element_gradient_tapes::Vector{ReverseDiff.CompiledTape}`: M distinct element function tapes;
- `index_element_tree::Vector{Int}`: from which any of the N element function may associate a gradient tape from `vec_element_gradient_tapes`.
Each `ReverseDiff.CompiledTape` accumulates the element-function's contribution in a element-vector of a `PartitionedVector`.
"""
mutable struct ElementReverseDiffGradient{T} <: AbstractGradientBackend{T}
  vec_element_gradient_tapes::Vector{ReverseDiff.CompiledTape}
  index_element_tree::Vector{Int}
end

"""
    gradient_brackend = ElementReverseDiffGradient(vec_elt_expr_tree::Vector, index_element_tree::Vector{Int}; type::Type{T}=Float64) where {T}

Return an `ElementReverseDiffGradient` from a `Vector` of expression trees
(supported by [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl))
of size `length(vec_elt_expr_tree)=M` and `index_element_tree` which redirects each element function `i`
 to its corresponding expression tree (1 ≤ `index_element_tree[i]` ≤ M, ∀ 1 ≤ i ≤ N).
"""
function ElementReverseDiffGradient(
  vec_elt_expr_tree::Vector,
  index_element_tree::Vector{Int};
  type::Type{T} = Float64,
) where {T}
  vec_compiled_element_gradients =
    map(element_tree -> compiled_grad_element_function(element_tree; type), vec_elt_expr_tree)
  return ElementReverseDiffGradient{type}(vec_compiled_element_gradients, index_element_tree)
end

function partitioned_gradient!(
  backend::ElementReverseDiffGradient{T},
  x::PartitionedVector{T},
  g::PartitionedVector{T},
) where {T}
  epv_x = x.epv
  epv_g = g.epv
  index_element_tree = backend.index_element_tree
  N = length(index_element_tree)
  for i = 1:N
    compiled_tape = backend.vec_element_gradient_tapes[index_element_tree[i]]
    Uix = PartitionedStructures.get_eev_value(epv_x, i)
    gi = PartitionedStructures.get_eev_value(epv_g, i)
    ReverseDiff.gradient!(gi, compiled_tape, Uix)
  end
  return g
end
