module Mod_PQN
using ReverseDiff
using PartitionedStructures, ExpressionTreeForge
using ..Mod_ab_partitioned_data, ..Mod_common
using ..ExpressionTreeForge.M_implementation_convexity_type

import ..Mod_ab_partitioned_data.update_nlp!

export PartitionedDataTRPQN
export update_PQN, update_PQN!, build_PartitionedDataTRPQN

"""
    PartitionedDataTRPQN{G, T <: Number, P <: Part_mat{T}} <: PartitionedData

Gather the structures required to run a partitioned quasi-Newton trust-region method.
`PartitionedDataTRPQN` has fields:

* `n`: the size of the problem;
* `N`: the number of element functions;
* `vec_elt_fun`: a `ElementFunction` vector, of size `N`;
* `M`: the number of distinct element-function expression trees;
* `vec_elt_complete_expr_tree`: a `Complete_expr_tree` vector, of size `M`;
* `element_expr_tree_table`: a vector of size `M`, the i-th element `element_expr_tree_table[i]::Vector{Int}` informs which element functions use the `vec_elt_complete_expr_tree[i]` expression tree;
* `index_element_tree`: a vector of size `N` where each component indicates which `Complete_expr_tree` from `vec_elt_complete_expr_tree` is used for the corresponding element;
* `vec_compiled_element_gradients`: the vector gathering the compiled tapes for every element gradient evaluations;
* `x`: the current point;
* `v`: a temporary vector;
* `s`: the current step;
* `pg`: the partitioned gradient;
* `pv`: a temporary partitioned vector;
* `py`: the partitioned gradient difference;
* `ps`: the partitioned step;
* `pB`: the partitioned matrix (main memory cost);
* `fx`: the current value of the objective function;
* `name`: the name of partitioned quasi-Newton update performed at each iteration:

  * `name=:pbfgs`: every element-Hessian approximation is updated with BFGS;
  * `name=:psr1`: every element-Hessian approximation is updated with SR1;
  * `name=:pse`: every element-Hessian approximation is updated with BFGS if the curvature condition holds, or with SR1 otherwise;
  * `name=:pcs`: each element-Hessian approximation with BFGS if it is classified as `convex`, or with SR1 otherwise;
  * `name=:plbfgs`: every element-Hessian approximations is a LBFGS operator;
  * `name=:plsr1`: every element-Hessian approximations is a LSR1 operator;
  * `name=:plse`: by default, every element-Hessian approximations is a LBFGS operator as long as the curvature condition holds, otherwise it becomes a LSR1 operator.

"""
mutable struct PartitionedDataTRPQN{G, T <: Number, P <: Part_mat{T}} <:
               Mod_ab_partitioned_data.PartitionedData
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
  pB::P # partitioned B

  fx::T
  # g is build directly from pg
  # the result of pB*v will be store and build from pv
  # name is the name of the partitioned quasi-Newton applied on pB
  name::Symbol
end

"""
    update_nlp!(pd_pqn::PartitionedDataTRPQN{G, T, P}, s::Vector{T})
    update_nlp!(pd_pqn::PartitionedDataTRPQN{G, T, P}, x::Vector{T}, s::Vector{T})

Perform the partitioned quasi-Newton update given the current point `x` and the step `s`.
When `x` is omitted, `update_PQN!` consider that `pd_pqn` has the current point in pd_pqn.x`.
Moreover, it assumes that the partitioned gradient at `x` is already computed in `pd_pqn.pg`.
"""
update_nlp!(
  pd_pqn::PartitionedDataTRPQN{G, T, P},
  s::Vector{T};
  kwargs...,
) where {G, T <: Number, P <: Part_mat{T}} = update_PQN!(pd_pqn, s; kwargs...)

update_nlp!(
  pd_pqn::PartitionedDataTRPQN{G, T, P},
  x::Vector{T},
  s::Vector{T};
  kwargs...,
) where {G, T <: Number, P <: Part_mat{T}} = update_PQN!(pd_pqn, x, s; kwargs...)

"""
    B = update_PQN(pd_pqn::PartitionedDataTRPQN{G, T, P}, x::Vector{T}, s::Vector{T};

Perform the partitioned quasi-Newton update given the current point `x` and the step `s`.
"""
update_PQN(
  pd_pqn::PartitionedDataTRPQN{G, T, P},
  x::Vector{T},
  s::Vector{T};
  kwargs...,
) where {G, T <: Number, P <: Part_mat{T}} = begin
  update_PQN!(pd_pqn, x, s; kwargs...)
  return Matrix(get_pB(pd_pqn))
end

"""
    update_PQN!(pd_pqn::PartitionedDataTRPQN{G, T, P}, s::Vector{T})
    update_PQN!(pd_pqn::PartitionedDataTRPQN{G, T, P}, x::Vector{T}, s::Vector{T})

Perform the partitioned quasi-Newton update given the current point `x` and the step `s`.
When `x` is omitted, `update_PQN!` consider that `pd_pqn` has the current point in pd_pqn.x`.
Moreover, it assumes that the partitioned gradient at `x` is already computed in `pd_pqn.pg`.
"""
function update_PQN!(
  pd_pqn::PartitionedDataTRPQN{G, T, P},
  x::Vector{T},
  s::Vector{T};
  kwargs...,
) where {G, T <: Number, P <: Part_mat{T}}
  set_x!(pd_pqn, x)
  evaluate_grad_part_data!(pd_pqn)
  update_PQN!(pd_pqn, s; kwargs...)
end

function update_PQN!(
  pd_pqn::PartitionedDataTRPQN{G, T, P},
  s::Vector{T};
  reset = 0,
  kwargs...,
) where {G, T <: Number, P <: Part_mat{T}}
  evaluate_y_part_data!(pd_pqn, s)
  py = get_py(pd_pqn)
  set_ps!(pd_pqn, s)
  ps = get_ps(pd_pqn)
  pB = get_pB(pd_pqn)
  PartitionedStructures.update!(pB, py, ps; name = pd_pqn.name, kwargs...)
  return pd_pqn
end

"""
    partitioneddata_tr_pqn = build_PartitionedDataTRPQN(expr_tree, n)

Return the structure required to run a partitioned quasi-Newton trust-region method. 
It finds the partially-separable structure of an expression tree `expr_tree` representing f(x) = ∑fᵢ(xᵢ).
Then it allocates the partitioned structures required.
To define properly the sparse matrix of the partitioned matrix we need the size of the problem: `n`.
"""
function build_PartitionedDataTRPQN(
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

  # convex_expr_tree = map(convexity_status -> is_convex(convexity_status), convexity_wrapper)
  convex_vector = zeros(Bool, N)
  for (index, list_element) in enumerate(element_expr_tree_table)
    map(index_element -> convex_vector[index_element] = is_convex(convexity_wrapper[index]), list_element)    
  end

  (name == :pbfgs) && (pB = epm_from_epv(pg))
  (name == :psr1) && (pB = epm_from_epv(pg))
  (name == :pse) && (pB = epm_from_epv(pg))
  (name == :pcs) && (pB = epm_from_epv(pg; convex_vector))
  (name == :plbfgs) && (pB = eplo_lbfgs_from_epv(pg; kwargs...))
  (name == :plsr1) && (pB = eplo_lsr1_from_epv(pg))
  (name == :plse) && (pB = eplo_lose_from_epv(pg; kwargs...))
  P = typeof(pB)

  fx = (T)(-1)
  pd_pqn = PartitionedDataTRPQN{ExpressionTreeForge.Complete_expr_tree, T, P}(
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
    pB,
    fx,
    name,
  )

  return pd_pqn
end

end
