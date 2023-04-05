module Utils

using ReverseDiff, LinearAlgebra
using NLPModelsJuMP
using ExpressionTreeForge, PartitionedStructures, PartitionedVectors
using ExpressionTreeForge.M_implementation_convexity_type

using ..ModAbstractPSNLPModels, ..PartitionedBackends

export distinct_element_expr_tree
export partitioned_structure

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
  # evaluate every element functions to reduce the number of comparisons between element expression trees
  vec_val_elt_fun = map(
    (elt_fun, elt_vars) ->
      ExpressionTreeForge.evaluate_expr_tree(elt_fun, collect(Float64, 1:length(elt_vars))),
    vec_element_expr_tree,
    vec_element_variables,
  )

  # retain each element function value and its original index
  working_array = map((val_elt_fun, i) -> (val_elt_fun, i), vec_val_elt_fun, 1:N)
  current_expr_tree_index = 1
  # Filter working_array with its current first element tree (value).
  # After an iterate, working_array doesn't possess an expression tree similar to value. 
  while isempty(working_array) == false
    value = working_array[1][1]
    ni_current_tree = length(vec_element_variables[working_array[1][2]])

    # a predicate filtering the element expression trees that cannot be equal
    comparator_value_elt_fun(val_elt_fun) =
      (val_elt_fun[1] == value) &&
      (length(vec_element_variables[val_elt_fun[2]]) == ni_current_tree)
    current_indices_similar_element_functions = findall(comparator_value_elt_fun, working_array)

    # get the real indices of the element trees selected by comparator_value_elt_fun
    real_indices_similar_element_functions =
      (tup -> tup[2]).(working_array[current_indices_similar_element_functions])
    # get the current element expression tree associated to value
    current_expr_tree = vec_element_expr_tree[working_array[1][2]]
    push!(element_expr_trees, current_expr_tree)

    # a predicate comparing current_expr_tree to an other expression tree node by node
    comparator_elt_expr_tree(expr_tree) = expr_tree == current_expr_tree
    current_indices_equal_element_function = findall(
      comparator_elt_expr_tree,
      vec_element_expr_tree[real_indices_similar_element_functions],
    )

    # get the reals indices (from vec_element_expr_tree) of element expression trees equalt to current_expr_tree
    real_indices_equal_element_function =
      (
        tup -> tup[2]
      ).(
        working_array[current_indices_similar_element_functions[current_indices_equal_element_function]]
      )
    # set for any element expression tree (from 1 to N) its new index (from 1 to M)
    indices_element_tree[real_indices_equal_element_function] .= current_expr_tree_index
    # remove the element expression treee treated from the working array
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
    (vec_element_functions, N, linear_vector) = merge_linear_elements(vec_element_functions::Vector{ExpressionTreeForge.Type_expr_tree}, N::Int)

Merge every linear element function from `vec_element_functions` into a single one.
Return the new adequate `vec_element_functions`, `N` and `linear_vector::Vector{Bool}` of size `N` indicating with `true` which element is linear.
If the method runs correctly, only `linear_vector[N]` may be set to `true`.
"""
function merge_linear_elements(
  vec_element_functions::Vector{ExpressionTreeForge.Type_expr_tree},
  N::Int,
)
  type_element_functions = ExpressionTreeForge.get_type_tree.(vec_element_functions)
  linears =
    (
      elt_fun -> ExpressionTreeForge.is_linear(elt_fun) || ExpressionTreeForge.is_constant(elt_fun)
    ).(type_element_functions)
  indices_linear_elements = filter(i -> linears[i], 1:N)
  if !isempty(indices_linear_elements)
    gathering_linear_elements =
      ExpressionTreeForge.sum_expr_trees(vec_element_functions[indices_linear_elements])
    indices_nonlinear_elements = filter(i -> !linears[i], 1:N)
    vec_element_functions =
      vcat(vec_element_functions[indices_nonlinear_elements], gathering_linear_elements)
    N = length(vec_element_functions)
    linear_vector = zeros(Bool, N) # every element function except the last one is nonlinear
    linear_vector[N] = true # last element function is the only one linear
  else
    linear_vector = zeros(Bool, N) # every element function is nonlinear
  end
  return vec_element_functions, N, linear_vector
end

merge_element_heuristic(
  vec_element_functions::Vector{ExpressionTreeForge.Type_expr_tree},
  element_variables::Vector{Vector{Int}},
  expr_tree::ExpressionTreeForge.Type_expr_tree,
  linear_vector::Vector{Bool},
  N::Int,
  n::Int,
  ::Val{false};
  name = :plse,
) = ()

function merge_element_heuristic(
  vec_element_functions::Vector{ExpressionTreeForge.Type_expr_tree},
  element_variables::Vector{Vector{Int}},
  expr_tree::ExpressionTreeForge.Type_expr_tree,
  linear_vector::Vector{Bool},
  N::Int,
  n::Int,
  ::Val{true};
  name = :plse,
)
  effective_size_element_var = map(i -> !linear_vector[i] * length(element_variables[i]), 1:N)
  mem_dense_elements = sum((size_element -> size_element^2).(effective_size_element_var))
  mem_linear_operator_elements =
    sum((size_element -> size_element * 5 * 2).(effective_size_element_var))
  max_authorised_mem = n^3 / log(n) # mem limit
  if (mem_dense_elements > max_authorised_mem) && (name ∈ [:pbfgs, :pse, :psr1, :pcs])
    @warn "mem usage to important, reduction to an unstructured structure"
    N = 1
    vec_element_functions = [expr_tree]
    element_variables = [ExpressionTreeForge.get_elemental_variables(expr_tree)]
  elseif (mem_linear_operator_elements > max_authorised_mem) && (name ∈ [:plbfgs, :plse, :plsr1])
    @warn "mem usage to important, reduction to an unstructured structure"
    N = 1
    vec_element_functions = [expr_tree]
    element_variables = [ExpressionTreeForge.get_elemental_variables(expr_tree)]
  end
  return (vec_element_functions, element_variables, N)
end

function select_objective_gradient_backend(
  nlp,
  n::Int,
  expr_tree,
  vec_element_functions,
  unnormalize_element_expr_trees,
  vec_typed_complete_element_tree,
  index_element_tree::Vector{Int},
  elemental_variables::Vector{Vector{Int}};
  type = Float64,
  objectivebackend = :nlp,
  gradientbackend = :reverseelt,
  kwargs...,
)

  # objective backend selection
  if (objectivebackend == :moiobj) && (type == Float64)
    @warn "The objective function is computed by an Evaluator of an MathOptInterface.Nonlinear.Model"
    objective_backend = MOIObjectiveBackend(expr_tree, n; elemental_variables, type, kwargs...)
  elseif (objectivebackend == :moielt) && (type == Float64)
    @warn "The gradient computes each element contribution from the Evaluator of an MathOptInterface.Nonlinear.Model"
    objective_backend =
      ElementMOIModelBackend(vec_typed_complete_element_tree, index_element_tree; kwargs...)
  elseif (objectivebackend == :modifiedmoiobj) && (type == Float64)
    @warn "The objective function is computed by an Evaluator of an MathOptInterface.Nonlinear.Model representing a modifier obejctive function"
    objective_backend = ModifiedObjectiveMOIModelBackend(vec_element_functions; kwargs...)
  elseif (objectivebackend == :spjacmoi) && (type == Float64)
    @warn "The objective function is computed by an Evaluator of an MathOptInterface.Nonlinear.Model representing a partially-separable function for which each constraint is an element function"
    objective_backend = SparseJacobianMoiModelBackend(
      unnormalize_element_expr_trees,
      n;
      elemental_variables,
      kwargs...,
    )
  elseif typeof(nlp) == MathOptNLPModel && (type != Float64)
    @warn "Incompatible backend, MathOptNLPModel can't support type != Float64, both Float64 and $(type) will be consider during the execution"
    objective_backend = NLPObjectiveBackend(nlp; type, kwargs...)
  else # objectivebackend == :nlp
    @warn "The objective function is computed NLPModels.obj(nlp, x), nlp being the original NLPModel"
    objective_backend = NLPObjectiveBackend(nlp; type, kwargs...)
  end

  # gradient backend selection
  if objectivebackend == gradientbackend
    @warn "Common backend for the objective and the gradient"
    gradient_backend = objective_backend
  else
    if (gradientbackend == :moielt) && (type == Float64)
      @warn "The gradient computes each element contribution from the Evaluator of an MathOptInterface.Nonlinear.Model"
      gradient_backend =
        ElementMOIModelBackend(vec_typed_complete_element_tree, index_element_tree; kwargs...)
    elseif (gradientbackend == :modifiedmoiobj) && (type == Float64)
      @warn "The partitioned gradient is computed by an Evaluator of an MathOptInterface.Nonlinear.Model representing a modifier obejctive function"
      gradient_backend = ModifiedObjectiveMOIModelBackend(vec_element_functions; kwargs...)
    elseif (type != Float64) && (gradientbackend == :moielt)
      @warn "Incompatible backend, MathOptInterface.Nonlinear.Model can't support type != Float64, by default, gradient_backend = ElementReverseDiffGradient"
      gradient_backend = ElementReverseDiffGradient(
        vec_typed_complete_element_tree,
        index_element_tree;
        type,
        kwargs...,
      )
    elseif (objectivebackend == :spjacmoi) && (type == Float64)
      @warn "The objective function is computed by an Evaluator of an MathOptInterface.Nonlinear.Model representing a partially-separable function for which each constraint is an element function"
      gradient_backend = SparseJacobianMoiModelBackend(
        unnormalize_element_expr_trees,
        n;
        elemental_variables,
        kwargs...,
      )
    else
      @warn "The gradient computes each element contribution from a ReverseDiff.GradientTape"
      gradient_backend = ElementReverseDiffGradient(
        vec_typed_complete_element_tree,
        index_element_tree;
        type,
        kwargs...,
      )
    end
  end
  return (objective_backend, gradient_backend)
end

function select_hprod_backend(
  vec_typed_complete_element_tree,
  index_element_tree::Vector{Int};
  kwargs...,
)
  hprod_backend =
    ElementReverseForwardHprod(vec_typed_complete_element_tree, index_element_tree; kwargs...)
  return hprod_backend
end

"""
    partitioned_structure = build_PartitionedDataTRPQN(expr_tree, n)

Return the structure required to run a partitioned quasi-Newton trust-region method. 
It finds the partially-separable structure of an expression tree `expr_tree` representing f(x) = ∑fᵢ(xᵢ).
Then it allocates the partitioned structures required.
To define properly the sparse matrix of the partitioned matrix we need the size of the problem: `n`.
"""
function partitioned_structure(
  nlp::SupportedNLPModel,
  tree::G,
  n::Int;
  type::DataType = Float64,
  name = :plse,
  merging::Bool = true,
  objectivebackend = :nlp,
  gradientbackend = :reverseelt,
  kwargs...,
) where {G}

  # Transform the expression tree of type G into an expression tree of type Type_expr_tree (the standard type used by my algorithms)
  expr_tree = ExpressionTreeForge.transform_to_expr_tree(tree)::ExpressionTreeForge.Type_expr_tree

  # Get the element functions
  vec_element_functions = copy.(
    ExpressionTreeForge.extract_element_functions(expr_tree)
  )::Vector{ExpressionTreeForge.Type_expr_tree}
  N = length(vec_element_functions)

  # merge linear element functions
  (vec_element_functions, N, linear_vector) = merge_linear_elements(vec_element_functions, N)

  # Retrieve elemental variables
  element_variables = map(
    (i -> ExpressionTreeForge.get_elemental_variables(vec_element_functions[i])),
    1:N,
  )::Vector{Vector{Int}}

  # Basic heuristic checking the memory requirement of a partitioned structure,
  # if the memory needed is too large, merge every element into a single one.
  (vec_element_functions, element_variables, N) = merge_element_heuristic(
    vec_element_functions,
    element_variables,
    expr_tree,
    linear_vector,
    N,
    n,
    Val(merging);
    name,
  )

  # IMPORTANT line, sort the elemental variables. Mandatory for normalize_indices! and the partitioned structures
  sort!.(element_variables)

  unnormalize_element_expr_trees = copy.(vec_element_functions)

  # Change the indices of the element-function expression trees.  
  map(
    ((elt_fun, elt_var) -> ExpressionTreeForge.normalize_indices!(elt_fun, elt_var)),
    vec_element_functions,
    element_variables,
  )

  # Filter the element expression tree to keep only the distinct expression trees
  (element_expr_tree, index_element_tree) =
    distinct_element_expr_tree(vec_element_functions, element_variables)
  M = length(element_expr_tree)

  # Create a table giving for each distinct element expression tree, every element function using it
  element_expr_tree_table = map((i -> findall((x -> x == i), index_element_tree)), 1:M)

  # Create complete trees given the remaining expression trees
  vec_elt_complete_expr_tree = ExpressionTreeForge.complete_tree.(element_expr_tree)
  # Cast the constant of the complete trees
  vec_typed_complete_element_tree =
    map(tree -> ExpressionTreeForge.cast_type_of_constant(tree, type), vec_elt_complete_expr_tree)
  ExpressionTreeForge.set_bounds!.(vec_typed_complete_element_tree) # Propagate the bounds 
  ExpressionTreeForge.set_convexity!.(vec_typed_complete_element_tree) # deduce the convexity status 

  # Get the convexity status of element functions
  convexity_wrapper = map(
    (
      complete_tree -> ExpressionTreeForge.M_implementation_convexity_type.Convexity_wrapper(
        ExpressionTreeForge.get_convexity_status(complete_tree),
      )
    ),
    vec_typed_complete_element_tree,
  )

  # Get the type of element functions
  type_element_function =
    map(elt_fun -> ExpressionTreeForge.get_type_tree(elt_fun), vec_typed_complete_element_tree)

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

  (objective_backend, gradient_backend) = select_objective_gradient_backend(
    nlp,
    n,
    expr_tree,
    vec_element_functions,
    unnormalize_element_expr_trees,
    vec_typed_complete_element_tree,
    index_element_tree,
    element_variables;
    type,
    objectivebackend,
    gradientbackend,
    kwargs...,
  )

  x = PartitionedVector(element_variables; T = type, n, simulate_vector = true)

  # convex_expr_tree = map(convexity_status -> is_convex(convexity_status), convexity_wrapper)
  convex_vector = zeros(Bool, N)
  for (index, list_element) in enumerate(element_expr_tree_table)
    map(
      index_element -> convex_vector[index_element] = is_convex(convexity_wrapper[index]),
      list_element,
    )
  end

  epv = PartitionedStructures.create_epv(element_variables, n, type = type)

  (name == :pbfgs) && (pB = epm_from_epv(epv; linear_vector))
  (name == :psr1) && (pB = epm_from_epv(epv; linear_vector))
  (name == :pse) && (pB = epm_from_epv(epv; linear_vector))
  (name == :pcs) && (pB = epm_from_epv(epv; convex_vector, linear_vector))
  (name == :plbfgs) && (pB = eplo_lbfgs_from_epv(epv; linear_vector, kwargs...))
  (name == :plsr1) && (pB = eplo_lsr1_from_epv(epv; linear_vector))
  (name == :plse) && (pB = eplo_lose_from_epv(epv; linear_vector, kwargs...))
  (name == :phv) && (
    pB =
      select_hprod_backend(vec_typed_complete_element_tree, index_element_tree; type, kwargs...)
  )

  fx = (type)(-1)
  return (
    n,
    N,
    vec_elt_fun,
    M,
    vec_typed_complete_element_tree,
    element_expr_tree_table,
    index_element_tree,
    objective_backend,
    gradient_backend,
    x,
    pB,
    fx,
    name,
  )
end

end
