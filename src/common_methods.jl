using SparseArrays
using ReverseDiff, ForwardDiff
using PartitionedStructures, ExpressionTreeForge

export get_n,
  get_N,
  get_vec_elt_fun,
  get_M,
  get_vec_elt_complete_expr_tree,
  get_element_expr_tree_table,
  get_index_element_tree,
  get_vec_compiled_element_gradients
export get_x, get_v, get_s, get_pg, get_pv, get_py, get_ps, get_phv, get_pB, get_fx
export set_n!,
  set_N!,
  set_vec_elt_fun!,
  set_M!,
  set_vec_elt_complete_expr_tree!,
  set_element_expr_tree_table!,
  set_index_element_tree!,
  set_vec_compiled_element_gradients!
export set_x!,
  set_v!,
  set_s!,
  set_pg!,
  set_pv!,
  set_ps!,
  set_pg!,
  set_pv!,
  set_py!,
  set_ps!,
  set_phv!,
  set_pB!,
  set_fx!

export product_part_data_x, evaluate_obj_part_data, evaluate_grad_part_data
export product_part_data_x!,
  evaluate_obj_part_data!, evaluate_y_part_data!, evaluate_grad_part_data!
export update_nlp!

@inline get_n(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.n

@inline get_N(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.N

@inline get_vec_elt_fun(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.vec_elt_fun

@inline get_M(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.M

@inline get_vec_elt_complete_expr_tree(psnlp::AbstractPartiallySeparableNLPModel) =
  psnlp.vec_elt_complete_expr_tree

@inline get_vec_elt_complete_expr_tree(psnlp::AbstractPartiallySeparableNLPModel, i::Int) =
  psnlp.vec_elt_complete_expr_tree[i]

@inline get_element_expr_tree_table(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.element_expr_tree_table

@inline get_index_element_tree(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.index_element_tree

@inline get_vec_compiled_element_gradients(psnlp::AbstractPartiallySeparableNLPModel) =
  psnlp.vec_compiled_element_gradients

@inline get_vec_compiled_element_gradients(psnlp::AbstractPartiallySeparableNLPModel, i::Int) =
  psnlp.vec_compiled_element_gradients[i]

@inline set_n!(psnlp::AbstractPartiallySeparableNLPModel, n::Int) = psnlp.n = n
@inline set_N!(psnlp::AbstractPartiallySeparableNLPModel, N::Int) = psnlp.N = N
@inline set_vec_elt_fun!(psnlp::AbstractPartiallySeparableNLPModel, vec_elt_fun::Vector{ElementFunction}) =
  psnlp.vec_elt_fun .= vec_elt_fun

@inline set_M!(psnlp::AbstractPartiallySeparableNLPModel, M::Int) = psnlp.M = M

@inline set_vec_elt_complete_expr_tree!(
  psnlp::AbstractPartiallySeparableNLPModel,
  vec_elt_complete_expr_tree::Vector{G},
) where {G} = psnlp.vec_elt_complete_expr_tree .= vec_elt_complete_expr_tree

@inline set_element_expr_tree_table!(
  psnlp::AbstractPartiallySeparableNLPModel,
  element_expr_tree_table::Vector{Vector{Int}},
) = psnlp.element_expr_tree_table .= element_expr_tree_table

@inline set_index_element_tree!(psnlp::AbstractPartiallySeparableNLPModel, index_element_tree::Vector{Int}) =
  psnlp.index_element_tree .= index_element_tree

@inline set_vec_compiled_element_gradients!(
  psnlp::AbstractPartiallySeparableNLPModel,
  vec_compiled_element_gradients::Vector{ReverseDiff.CompiledTape},
) = psnlp.vec_compiled_element_gradients = vec_compiled_element_gradients

#=-----------------------------------------------------------------------------------------
 ------------------------------------AbstractPQNNLPModel-----------------------------------
-----------------------------------------------------------------------------------------=#
@inline get_n(psnlp::AbstractPQNNLPModel) = psnlp.n

@inline get_N(psnlp::AbstractPQNNLPModel) = psnlp.N

@inline get_vec_elt_fun(psnlp::AbstractPQNNLPModel) = psnlp.vec_elt_fun

@inline get_M(psnlp::AbstractPQNNLPModel) = psnlp.M

@inline get_vec_elt_complete_expr_tree(psnlp::AbstractPQNNLPModel) =
  psnlp.vec_elt_complete_expr_tree

@inline get_vec_elt_complete_expr_tree(psnlp::AbstractPQNNLPModel, i::Int) =
  psnlp.vec_elt_complete_expr_tree[i]

@inline get_element_expr_tree_table(psnlp::AbstractPQNNLPModel) = psnlp.element_expr_tree_table

@inline get_index_element_tree(psnlp::AbstractPQNNLPModel) = psnlp.index_element_tree

@inline get_vec_compiled_element_gradients(psnlp::AbstractPQNNLPModel) =
  psnlp.vec_compiled_element_gradients

@inline get_vec_compiled_element_gradients(psnlp::AbstractPQNNLPModel, i::Int) =
  psnlp.vec_compiled_element_gradients[i]

@inline set_n!(psnlp::AbstractPQNNLPModel, n::Int) = psnlp.n = n
@inline set_N!(psnlp::AbstractPQNNLPModel, N::Int) = psnlp.N = N
@inline set_vec_elt_fun!(psnlp::AbstractPQNNLPModel, vec_elt_fun::Vector{ElementFunction}) =
  psnlp.vec_elt_fun .= vec_elt_fun

@inline set_M!(psnlp::AbstractPQNNLPModel, M::Int) = psnlp.M = M

@inline set_vec_elt_complete_expr_tree!(
  psnlp::AbstractPQNNLPModel,
  vec_elt_complete_expr_tree::Vector{G},
) where {G} = psnlp.vec_elt_complete_expr_tree .= vec_elt_complete_expr_tree

@inline set_element_expr_tree_table!(
  psnlp::AbstractPQNNLPModel,
  element_expr_tree_table::Vector{Vector{Int}},
) = psnlp.element_expr_tree_table .= element_expr_tree_table

@inline set_index_element_tree!(psnlp::AbstractPQNNLPModel, index_element_tree::Vector{Int}) =
  psnlp.index_element_tree .= index_element_tree

@inline set_vec_compiled_element_gradients!(
  psnlp::AbstractPQNNLPModel,
  vec_compiled_element_gradients::Vector{ReverseDiff.CompiledTape},
) = psnlp.vec_compiled_element_gradients = vec_compiled_element_gradients