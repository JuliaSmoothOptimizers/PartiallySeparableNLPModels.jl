module ModPBFGSNLPModels

using ..Utils
using ..ModPSNLPModels
using ExpressionTreeForge, PartitionedStructures
using NLPModels
using ReverseDiff

export PBFGSNLPModel

mutable struct PBFGSNLPModel{G, P, T, S, M <: AbstractNLPModel{T, S}, Meta <: AbstractNLPModelMeta{T, S},} <: AbstractPQNNLPModel{T,S}
  nlp::M
  meta::Meta
  counters::NLPModels.Counters

  n::Int
  N::Int
  vec_elt_fun::Vector{ElementFunction} #length(vec_elt_fun) == N
  # Vector composed by the expression trees of element functions .
  # Warning: Several element functions may have the same expression tree
  M::Int
  vec_elt_complete_expr_tree::Vector{G} # length(element_expr_tree) == M < N
  # element_expr_tree_table store the indices of every element function using each element_expr_tree, ∀i,j, 1 ≤ element_expr_tree_table[i][j] \leq N
  element_expr_tree_table::Vector{Vector{Int}} # length(element_expr_tree_table) == M
  index_element_tree::Vector{Int} # length(index_element_tree) == N, index_element_tree[i] ≤ M

  vec_compiled_element_gradients::Vector{ReverseDiff.CompiledTape}

  x::Vector{T} # length(x)==n
  v::Vector{T} # length(v)==n
  s::Vector{T} # length(v)==n
  pg::PartitionedStructures.Elemental_pv{T} # partitioned gradient
  pv::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  py::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  ps::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  phv::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  pB::P # partitioned B

  fx::T
  # g is build directly from pg
  # the result of pB*v will be store and build from pv
  # name is the name of the partitioned quasi-Newton applied on pB
  name::Symbol
end 


function PBFGSNLPModel(nlp::SupportedNLPModel)
  n = nlp.meta.nvar
  x0 = nlp.meta.x0
  ex = get_expression_tree(nlp)
  T = eltype(x0)
  expr_tree = ExpressionTreeForge.transform_to_expr_tree(ex)::ExpressionTreeForge.Type_expr_tree
  # expr_tree = ex
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
    map((tree -> compiled_grad_element_function(tree; type = T)::ReverseDiff.CompiledTape), element_expr_tree)

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

  name = :pbfgs
  pB = epm_from_epv(pg)
  fx = (T)(-1)
  
  # (n, N, vec_elt_fun, M, vec_elt_complete_expr_tree, element_expr_tree_table, index_element_tree, vec_compiled_element_gradients, x, v, s, pg, pv, py, ps, phv, pB, fx, name) = partially_separable_structure(ex, n; name=:pbfgs, x0)
  # @show x
  # println(x)
  meta = nlp.meta
  counters = NLPModels.Counters()
  sleep(2)
  PBFGSNLPModel(nlp, meta, counters, n, N, vec_elt_fun, M, vec_elt_complete_expr_tree, element_expr_tree_table, index_element_tree, vec_compiled_element_gradients, x, v, s, pg, pv, py, ps, phv, pB, fx, name)
end
  
end


