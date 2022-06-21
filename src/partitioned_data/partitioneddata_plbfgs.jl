module Mod_PLBFGS
using ReverseDiff
using PartitionedStructures, CalculusTreeTools
using ..Mod_ab_partitioned_data, ..Mod_common

import ..Mod_ab_partitioned_data.update_nlp!

export PartitionedData_TR_PLBFGS
export update_PLBFGS, update_PLBFGS!, build_PartitionedData_TR_PLBFGS

mutable struct PartitionedData_TR_PLBFGS{G, T <: Number} <: Mod_ab_partitioned_data.PartitionedData
  n::Int
  N::Int
  vec_elt_fun::Vector{Element_function} #length(vec_elt_fun) == N
  # Vector composed by the different expression graph of element function .
  # Several element function may have the same expression graph
  M::Int
  vec_elt_complete_expr_tree::Vector{G} # length(element_expr_tree) == M < N
  element_expr_tree_table::Vector{Vector{Int}} # length(element_expr_tree_table) == M
  # element_expr_tree_table store the indices of every element function using each element_expr_tree, ∀i,j, 1 ≤ element_expr_tree_table[i][j] \leq N
  index_element_tree::Vector{Int} # length(index_element_tree) == N, index_element_tree[i] ≤ M

  vec_compiled_element_gradients::Vector{ReverseDiff.CompiledTape}

  x::Vector{T} # length(x)==n
  v::Vector{T} # length(v)==n
  s::Vector{T} # length(v)==n
  pg::PartitionedStructures.Elemental_pv{T} # partitioned gradient
  pv::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  py::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  ps::PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
  pB::PartitionedStructures.Elemental_plom_bfgs{T} # partitioned B

  fx::T
  # g is build directly from pg
  # the result of pB*v will be store and build from pv

  name::Symbol
end

update_nlp!(
  pd_plbfgs::PartitionedData_TR_PLBFGS{G, T},
  s::Vector{T};
  kwargs...,
) where {G, T <: Number} = update_PLBFGS!(pd_plbfgs, s; kwargs...)
update_nlp!(
  pd_plbfgs::PartitionedData_TR_PLBFGS{G, T},
  x::Vector{T},
  s::Vector{T};
  kwargs...,
) where {G, T <: Number} = update_PLBFGS!(pd_plbfgs, x, s; kwargs...)

"""
		update_PLBFGS(pd_pblfgs,x,s)
Perform the PBFGS update givent the two iterate x and s
"""
update_PLBFGS(
  pd_pblfgs::PartitionedData_TR_PLBFGS{G, T},
  x::Vector{T},
  s::Vector{T};
  kwargs...,
) where {G, T <: Number} = begin
  update_PLBFGS!(pd_pblfgs, x, s; kwargs...)
  return Matrix(get_pB(pd_pblfgs))
end
function update_PLBFGS!(
  pd_pblfgs::PartitionedData_TR_PLBFGS{G, T},
  x::Vector{T},
  s::Vector{T};
  kwargs...,
) where {G, T <: Number}
  set_x!(pd_pblfgs, x)
  evaluate_grad_part_data!(pd_pblfgs)
  update_PLBFGS!(pd_pblfgs, s; kwargs...)
end

"""
		update_PLBFGS(pd_pblfgs,s)
Perform the PBFGS update givent the current iterate x and the next iterate s
"""
function update_PLBFGS!(
  pd_pblfgs::PartitionedData_TR_PLBFGS{G, T},
  s::Vector{T};
  kwargs...,
) where {G, T <: Number}
  evaluate_y_part_data!(pd_pblfgs, s)
  py = get_py(pd_pblfgs)
  set_ps!(pd_pblfgs, s)
  ps = get_ps(pd_pblfgs)
  pB = get_pB(pd_pblfgs)
  PartitionedStructures.PLBFGS_update!(pB, py, ps; kwargs...)
end

"""
    build_PartitionedData_TR_PLBFGS(expr_tree, n)
Find the partially separable structure of a function f stored as an expression tree expr_tree.
To define properly the size of sparse matrix we need the size of the problem : n.
At the end, we get the partially separable structure of f, f(x) = ∑fᵢ(xᵢ)
"""
function build_PartitionedData_TR_PLBFGS(
  tree::G,
  n::Int;
  x0::Vector{T} = rand(Float64, n),
) where {G, T <: Number}
  expr_tree = CalculusTreeTools.transform_to_expr_tree(tree)::CalculusTreeTools.t_expr_tree # transform the expression tree of type G into an expr tree of type t_expr_tree (the standard type used by my algorithms)
  vec_element_function =
    CalculusTreeTools.delete_imbricated_plus(expr_tree)::Vector{CalculusTreeTools.t_expr_tree} #séparation en fonction éléments
  N = length(vec_element_function)

  element_variables = map(
    (i -> CalculusTreeTools.get_elemental_variable(vec_element_function[i])),
    1:N,
  )::Vector{Vector{Int}}# retrieve elemental variables
  sort!.(element_variables) # important line, sort the elemental varaibles. Mandatory for N_to_Ni and the partitioned structures
  map(
    ((elt_fun, elt_var) -> CalculusTreeTools.element_fun_from_N_to_Ni!(elt_fun, elt_var)),
    vec_element_function,
    element_variables,
  ) # renumérotation des variables des fonctions éléments en variables internes

  (element_expr_tree, index_element_tree) =
    distinct_element_expr_tree(vec_element_function, element_variables) # Remains only the distinct expr graph functions
  M = length(element_expr_tree)
  element_expr_tree_table = map((i -> findall((x -> x == i), index_element_tree)), 1:M) # create a table that give for each distinct element expr grah, every element function using it

  vec_elt_complete_expr_tree = CalculusTreeTools.create_complete_tree.(element_expr_tree) # create complete tree given the remaining expr graph
  vec_type_complete_element_tree =
    map(tree -> CalculusTreeTools.cast_type_of_constant(tree, T), vec_elt_complete_expr_tree) # cast the constant of the complete trees

  CalculusTreeTools.set_bounds!.(vec_type_complete_element_tree) # deduce the bounds 
  CalculusTreeTools.set_convexity!.(vec_type_complete_element_tree) # deduce the bounds 
  # information concernant la convexité des arbres complets distincts

  convexity_wrapper = map(
    (
      complete_tree -> CalculusTreeTools.convexity_wrapper(
        CalculusTreeTools.get_convexity_status(complete_tree),
      )
    ),
    vec_type_complete_element_tree,
  ) # convexity of element function
  type_element_function =
    map(elt_fun -> CalculusTreeTools.get_type_tree(elt_fun), vec_type_complete_element_tree) # type of element function

  vec_elt_fun = Vector{Element_function}(undef, N)
  for i = 1:N  # Define the N element functions
    index_distinct_element_tree = index_element_tree[i]
    elt_fun = Element_function(
      i,
      index_distinct_element_tree,
      element_variables[i],
      type_element_function[index_distinct_element_tree],
      convexity_wrapper[index_distinct_element_tree],
    )
    vec_elt_fun[i] = elt_fun
  end

  vec_compiled_element_gradients =
    map((tree -> compiled_grad_elmt_fun(tree; type = T)), element_expr_tree)

  x = copy(x0)
  v = similar(x)
  s = similar(x)

  pg = PartitionedStructures.create_epv(element_variables, n, type = T)
  pv = similar(pg)
  py = similar(pg)
  ps = similar(pg)

  pB = identity_eplom_LBFGS(element_variables, N, n; T = T)
  fx = -1
  pd_pblfgs = PartitionedData_TR_PLBFGS{CalculusTreeTools.complete_expr_tree, T}(
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
    :plbfgs,
  )

  return pd_pblfgs
end

end
