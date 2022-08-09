module PartiallySeparableNLPModels

using ExpressionTreeForge
using PartitionedStructures

include("partitioned_data/_include.jl")

using .Mod_ab_partitioned_data, .Mod_PQN
using .Mod_partitionedNLPModel

export PartitionedData, PartitionedDataTRPQN, build_PartitionedDataTRPQN
export PartiallySeparableNLPModel

export element_function
export product_part_data_x, evaluate_obj_part_data, evaluate_grad_part_data
export product_part_data_x!,
  evaluate_obj_part_data!, evaluate_y_part_data!, evaluate_grad_part_data!
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
export update_nlp, update_nlp!, update_PQN, update_PQN!
export hess_approx

end
