module Utils

using ReverseDiff, LinearAlgebra
using ExpressionTreeForge, PartitionedStructures
using ExpressionTreeForge.M_implementation_convexity_type

using ..ModAbstractPSNLPModels

export distinct_element_expr_tree, compiled_grad_element_function
export partially_separable_structure

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

"""
    partitioneddata_tr_pqn = build_PartitionedDataTRPQN(expr_tree, n)

Return the structure required to run a partitioned quasi-Newton trust-region method. 
It finds the partially-separable structure of an expression tree `expr_tree` representing f(x) = ∑fᵢ(xᵢ).
Then it allocates the partitioned structures required.
To define properly the sparse matrix of the partitioned matrix we need the size of the problem: `n`.
"""
function partially_separable_structure(
  tree::G,
  n::Int;
  x0::Vector{T} = rand(Float64, n),
  name = :plse,
  kwargs...,
) where {G, T <: Number}

  # Transform the expression tree of type G into an expression tree of type Type_expr_tree (the standard type used by my algorithms)
  expr_tree = ExpressionTreeForge.transform_to_expr_tree(tree)::ExpressionTreeForge.Type_expr_tree

  # Get the element functions
  vec_element_function = ExpressionTreeForge.extract_element_functions(
    expr_tree,
  )::Vector{ExpressionTreeForge.Type_expr_tree}
  N = length(vec_element_function)

  # Retrieve elemental variables
  element_variables = map(
    (i -> ExpressionTreeForge.get_elemental_variables(vec_element_function[i])),
    1:N,
  )::Vector{Vector{Int}}

  # IMPORTANT line, sort the elemental variables. Mandatory for normalize_indices! and the partitioned structures
  sort!.(element_variables)

  # Change the indices of the element-function expression trees.
  map(
    ((elt_fun, elt_var) -> ExpressionTreeForge.normalize_indices!(elt_fun, elt_var)),
    vec_element_function,
    element_variables,
  )

  # Filter the element expression tree to keep only the distinct expression trees
  (element_expr_tree, index_element_tree) =
    distinct_element_expr_tree(vec_element_function, element_variables)
  M = length(element_expr_tree)

  # Create a table giving for each distinct element expression tree, every element function using it
  element_expr_tree_table = map((i -> findall((x -> x == i), index_element_tree)), 1:M)

  # Create complete trees given the remaining expression trees
  vec_elt_complete_expr_tree = ExpressionTreeForge.complete_tree.(element_expr_tree)
  # Cast the constant of the complete trees
  vec_type_complete_element_tree =
    map(tree -> ExpressionTreeForge.cast_type_of_constant(tree, T), vec_elt_complete_expr_tree)

  ExpressionTreeForge.set_bounds!.(vec_type_complete_element_tree) # Propagate the bounds 
  ExpressionTreeForge.set_convexity!.(vec_type_complete_element_tree) # deduce the convexity status 

  # Get the convexity status of element functions
  convexity_wrapper = map(
    (
      complete_tree -> ExpressionTreeForge.M_implementation_convexity_type.Convexity_wrapper(
        ExpressionTreeForge.get_convexity_status(complete_tree),
      )
    ),
    vec_type_complete_element_tree,
  )

  # Get the type of element functions
  type_element_function =
    map(elt_fun -> ExpressionTreeForge.get_type_tree(elt_fun), vec_type_complete_element_tree)

  vec_elt_fun = Vector{ElementFunction}(undef, N)
  for i = 1:N  # Define the N element functions
    index_distinct_element_tree = index_element_tree[i]
    elt_fun = ElementFunction(
      i,
      index_distinct_element_tree,
      element_variables[i],
      type_element_function[index_distinct_element_tree],
      convexity_wrapper[index_distinct_element_tree],
    )
    vec_elt_fun[i] = elt_fun
  end

  vec_compiled_element_gradients =
    map((tree -> compiled_grad_element_function(tree; type = T)), element_expr_tree)

  x = copy(x0)
  v = similar(x)
  s = similar(x)

  pg = PartitionedStructures.create_epv(element_variables, n, type = T)
  pv = similar(pg)
  py = similar(pg)
  ps = similar(pg)
  phv = similar(pg)

  # convex_expr_tree = map(convexity_status -> is_convex(convexity_status), convexity_wrapper)
  convex_vector = zeros(Bool, N)
  for (index, list_element) in enumerate(element_expr_tree_table)
    map(
      index_element -> convex_vector[index_element] = is_convex(convexity_wrapper[index]),
      list_element,
    )
  end

  (name == :pbfgs) && (pB = epm_from_epv(pg))
  (name == :psr1) && (pB = epm_from_epv(pg))
  (name == :pse) && (pB = epm_from_epv(pg))
  (name == :pcs) && (pB = epm_from_epv(pg; convex_vector))
  (name == :plbfgs) && (pB = eplo_lbfgs_from_epv(pg; kwargs...))
  (name == :plsr1) && (pB = eplo_lsr1_from_epv(pg))
  (name == :plse) && (pB = eplo_lose_from_epv(pg; kwargs...))
  (name == :phv) && (pB = nothing)

  fx = (T)(-1)
  return (
    n,
    N,
    vec_elt_fun,
    M,
    vec_elt_complete_expr_tree,
    element_expr_tree_table,
    index_element_tree,
    vec_compiled_element_gradients,
    x,
    v,
    s,
    pg,
    pv,
    py,
    ps,
    phv,
    pB,
    fx,
    name,
  )
end


"""
    partitioneddata_tr_pqn = build_PartitionedDataTRPQN(expr_tree, n)

Return the structure required to run a partitioned quasi-Newton trust-region method. 
It finds the partially-separable structure of an expression tree `expr_tree` representing f(x) = ∑fᵢ(xᵢ).
Then it allocates the partitioned structures required.
To define properly the sparse matrix of the partitioned matrix we need the size of the problem: `n`.
"""
function partially_separable_structure(
  tree::G,
  n::Int;
  x0::Vector{T} = rand(Float64, n),
  name = :plse,
  kwargs...,
) where {G, T <: Number}

  # Transform the expression tree of type G into an expression tree of type Type_expr_tree (the standard type used by my algorithms)
  expr_tree = ExpressionTreeForge.transform_to_expr_tree(tree)::ExpressionTreeForge.Type_expr_tree

  # Get the element functions
  vec_element_function = ExpressionTreeForge.extract_element_functions(
    expr_tree,
  )::Vector{ExpressionTreeForge.Type_expr_tree}
  N = length(vec_element_function)

  # Retrieve elemental variables
  element_variables = map(
    (i -> ExpressionTreeForge.get_elemental_variables(vec_element_function[i])),
    1:N,
  )::Vector{Vector{Int}}

  # IMPORTANT line, sort the elemental variables. Mandatory for normalize_indices! and the partitioned structures
  sort!.(element_variables)

  # Change the indices of the element-function expression trees.
  map(
    ((elt_fun, elt_var) -> ExpressionTreeForge.normalize_indices!(elt_fun, elt_var)),
    vec_element_function,
    element_variables,
  )

  # Filter the element expression tree to keep only the distinct expression trees
  (element_expr_tree, index_element_tree) =
    distinct_element_expr_tree(vec_element_function, element_variables)
  M = length(element_expr_tree)

  # Create a table giving for each distinct element expression tree, every element function using it
  element_expr_tree_table = map((i -> findall((x -> x == i), index_element_tree)), 1:M)

  # Create complete trees given the remaining expression trees
  vec_elt_complete_expr_tree = ExpressionTreeForge.complete_tree.(element_expr_tree)
  # Cast the constant of the complete trees
  vec_type_complete_element_tree =
    map(tree -> ExpressionTreeForge.cast_type_of_constant(tree, T), vec_elt_complete_expr_tree)

  ExpressionTreeForge.set_bounds!.(vec_type_complete_element_tree) # Propagate the bounds 
  ExpressionTreeForge.set_convexity!.(vec_type_complete_element_tree) # deduce the convexity status 

  # Get the convexity status of element functions
  convexity_wrapper = map(
    (
      complete_tree -> ExpressionTreeForge.M_implementation_convexity_type.Convexity_wrapper(
        ExpressionTreeForge.get_convexity_status(complete_tree),
      )
    ),
    vec_type_complete_element_tree,
  )

  # Get the type of element functions
  type_element_function =
    map(elt_fun -> ExpressionTreeForge.get_type_tree(elt_fun), vec_type_complete_element_tree)

  vec_elt_fun = Vector{ElementFunction}(undef, N)
  for i = 1:N  # Define the N element functions
    index_distinct_element_tree = index_element_tree[i]
    elt_fun = ElementFunction(
      i,
      index_distinct_element_tree,
      element_variables[i],
      type_element_function[index_distinct_element_tree],
      convexity_wrapper[index_distinct_element_tree],
    )
    vec_elt_fun[i] = elt_fun
  end

  vec_compiled_element_gradients =
    map((tree -> compiled_grad_element_function(tree; type = T)), element_expr_tree)

    epv = PartitionedStructures.create_epv(element_variables, n, type = T)
  
    # x = copy(x0)
    x = PartitionedVector(epv; T, simulate_vector=true)

  # convex_expr_tree = map(convexity_status -> is_convex(convexity_status), convexity_wrapper)
  convex_vector = zeros(Bool, N)
  for (index, list_element) in enumerate(element_expr_tree_table)
    map(
      index_element -> convex_vector[index_element] = is_convex(convexity_wrapper[index]),
      list_element,
    )
  end

  (name == :pbfgs) && (pB = epm_from_epv(pg))
  (name == :psr1) && (pB = epm_from_epv(pg))
  (name == :pse) && (pB = epm_from_epv(pg))
  (name == :pcs) && (pB = epm_from_epv(pg; convex_vector))
  (name == :plbfgs) && (pB = eplo_lbfgs_from_epv(pg; kwargs...))
  (name == :plsr1) && (pB = eplo_lsr1_from_epv(pg))
  (name == :plse) && (pB = eplo_lose_from_epv(pg; kwargs...))
  (name == :phv) && (pB = nothing)

  fx = (T)(-1)
  return (
    n,
    N,
    vec_elt_fun,
    M,
    vec_elt_complete_expr_tree,
    element_expr_tree_table,
    index_element_tree,
    vec_compiled_element_gradients,
    x,
    v,
    s,
    pg,
    pv,
    py,
    ps,
    phv,
    pB,
    fx,
    name,
  )
end


end
