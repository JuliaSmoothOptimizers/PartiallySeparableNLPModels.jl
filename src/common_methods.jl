using SparseArrays
using ReverseDiff, ForwardDiff
using PartitionedStructures, ExpressionTreeForge

export get_n,
  get_N,
  get_M,
  get_vec_elt_complete_expr_tree,
  get_index_element_tree,
  get_vec_compiled_element_gradients

@inline get_n(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.n
@inline get_n(psnlp::AbstractPQNNLPModel) = psnlp.n

@inline get_N(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.N
@inline get_N(psnlp::AbstractPQNNLPModel) = psnlp.N

@inline get_M(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.M
@inline get_M(psnlp::AbstractPQNNLPModel) = psnlp.M

@inline get_vec_elt_complete_expr_tree(psnlp::AbstractPartiallySeparableNLPModel) =
  psnlp.vec_elt_complete_expr_tree
@inline get_vec_elt_complete_expr_tree(psnlp::AbstractPQNNLPModel) =
  psnlp.vec_elt_complete_expr_tree

@inline get_vec_elt_complete_expr_tree(psnlp::AbstractPartiallySeparableNLPModel, i::Int) =
  psnlp.vec_elt_complete_expr_tree[i]
@inline get_vec_elt_complete_expr_tree(psnlp::AbstractPQNNLPModel, i::Int) =
  psnlp.vec_elt_complete_expr_tree[i]

@inline get_index_element_tree(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.index_element_tree
@inline get_index_element_tree(psnlp::AbstractPQNNLPModel) = psnlp.index_element_tree

@inline get_vec_compiled_element_gradients(psnlp::AbstractPartiallySeparableNLPModel) =
  psnlp.vec_compiled_element_gradients
@inline get_vec_compiled_element_gradients(psnlp::AbstractPQNNLPModel) =
  psnlp.vec_compiled_element_gradients

@inline get_vec_compiled_element_gradients(psnlp::AbstractPartiallySeparableNLPModel, i::Int) =
  psnlp.vec_compiled_element_gradients[i]
@inline get_vec_compiled_element_gradients(psnlp::AbstractPQNNLPModel, i::Int) =
  psnlp.vec_compiled_element_gradients[i]