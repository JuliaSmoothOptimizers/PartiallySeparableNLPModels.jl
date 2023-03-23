using SparseArrays
using ReverseDiff, ForwardDiff
using PartitionedStructures, ExpressionTreeForge

export get_n,
  get_N,
  get_M,
  get_vec_elt_complete_expr_tree,
  get_index_element_tree,
  get_objective_evaluator,
  get_modified_objective_evaluator,
  get_x_modified,
  get_v_modified,
  set_vector_from_pv!,
  set_pv_from_vector!

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

@inline get_objective_evaluator(psnlp::AbstractPartiallySeparableNLPModel) =
  psnlp.objective_evaluator
@inline get_objective_evaluator(pqnnlp::AbstractPQNNLPModel) =
  pqnnlp.objective_evaluator
@inline get_modified_objective_evaluator(psnlp::AbstractPartiallySeparableNLPModel) =
  psnlp.modified_objective_evaluator
@inline get_modified_objective_evaluator(pqnnlp::AbstractPQNNLPModel) =
  pqnnlp.modified_objective_evaluator

@inline get_x_modified(psnlp::AbstractPartiallySeparableNLPModel) =
  psnlp.x_modified
@inline get_x_modified(pqnnlp::AbstractPQNNLPModel) =
  pqnnlp.x_modified

@inline get_v_modified(psnlp::AbstractPartiallySeparableNLPModel) =
  psnlp.v_modified
@inline get_v_modified(pqnnlp::AbstractPQNNLPModel) =
  pqnnlp.v_modified



function set_vector_from_pv!(v::Vector{T}, pv::PartitionedVector{T}) where T
  cpt = 1
  for i in 1:size(pv,1)    
    nie = pv[i].nie
    range = cpt:cpt+nie-1
    view(v, range) .= pv[i].vec
    cpt += nie
  end
  return pv
end

function set_pv_from_vector!(pv::PartitionedVector{T}, v::Vector{T}) where T
  cpt = 1
  for i in 1:size(pv,1)    
    nie = pv[i].nie
    range = cpt:cpt+nie-1
    pv[i].vec .= view(v, range)
    cpt += nie
  end
  return pv
end