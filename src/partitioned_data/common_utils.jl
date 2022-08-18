module Mod_common

using ReverseDiff, LinearAlgebra
using ExpressionTreeForge, PartitionedStructures
using ExpressionTreeForge.M_implementation_convexity_type

export ElementFunction
export distinct_element_expr_tree, compiled_grad_element_function

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

"""
    (element_expr_trees, indices_element_tree) = distinct_element_expr_tree(vec_element_expr_tree::Vector{T}, vec_element_variables::Vector{Vector{Int}}; N::Int = length(vec_element_expr_tree)) where {T}

In practice, there may have several element functions having the same expression tree.
`distinct_element_expr_tree` filters the vector `vec_element_expr_tree` to return `element_expr_trees` the distincts element functions.
`length(element_expr_trees) == M < N == length(vec_element_expr_tree)`.
In addition it returns `indices_element_tree`, who records the index (1 <= i <= M) related ot the expression tree of each element function.
"""
function distinct_element_expr_tree(
  vec_element_expr_tree::Vector{T},
  vec_element_variables::Vector{Vector{Int}};
  N::Int = length(vec_element_expr_tree),
) where {T}
  N == length(vec_element_variables) ||
    @error("The sizes vec_element_expr_tree and vec_element_variables are differents")
  indices_element_tree = (xi -> -xi).(ones(Int, N))
  element_expr_trees = Vector{T}(undef, 0)
  vec_val_elt_fun_ones = map(
    (elt_fun, elt_vars) ->
      ExpressionTreeForge.evaluate_expr_tree(elt_fun, ones(length(elt_vars))),
    vec_element_expr_tree,
    vec_element_variables,
  ) # evaluate as first equality test
  working_array = map((val_elt_fun_ones, i) -> (val_elt_fun_ones, i), vec_val_elt_fun_ones, 1:N)
  current_expr_tree_index = 1
  # Filter working_array with its current first element tree (val).
  # After an iterate, working_array doesn't possess anymore expression tree similarto val. 
  while isempty(working_array) == false
    val = working_array[1][1]
    comparator_value_elt_fun(val_elt_fun) = val_elt_fun[1] == val
    current_indices_similar_element_functions =
      findall(comparator_value_elt_fun, working_array[:, 1])
    real_indices_similar_element_functions =
      (tup -> tup[2]).(working_array[current_indices_similar_element_functions])
    current_expr_tree = vec_element_expr_tree[working_array[1][2]]
    push!(element_expr_trees, current_expr_tree)
    comparator_elt_expr_tree(expr_tree) = expr_tree == current_expr_tree
    current_indices_equal_element_function = findall(
      comparator_elt_expr_tree,
      vec_element_expr_tree[real_indices_similar_element_functions],
    )
    real_indices_equal_element_function =
      (
        tup -> tup[2]
      ).(
        working_array[current_indices_similar_element_functions[current_indices_equal_element_function]]
      )
    indices_element_tree[real_indices_equal_element_function] .= current_expr_tree_index
    deleteat!(
      working_array,
      current_indices_similar_element_functions[current_indices_equal_element_function],
    )
    current_expr_tree_index += 1
  end
  minimum(indices_element_tree) == -1 && @error("Not every element function is attributed")
  return element_expr_trees, indices_element_tree
end

"""
    element_gradient_tape = compiled_grad_element_function(element_function::T; ni::Int = length(ExpressionTreeForge.get_elemental_variables(element_function)), type = Float64) where {T}

Return the `elment_gradient_tape::GradientTape` which speed up the gradient computation of `element_function` with `ReverseDiff`.
"""
function compiled_grad_element_function(
  element_function::T;
  ni::Int = length(ExpressionTreeForge.get_elemental_variables(element_function)),
  type = Float64,
) where {T}
  f = ExpressionTreeForge.evaluate_expr_tree(element_function)
  f_tape = ReverseDiff.GradientTape(f, rand(type, ni))
  compiled_f_tape = ReverseDiff.compile(f_tape)
  return compiled_f_tape
end

end
