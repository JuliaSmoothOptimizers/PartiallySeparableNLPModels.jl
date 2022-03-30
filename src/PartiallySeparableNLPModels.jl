module PartiallySeparableNLPModels

	using CalculusTreeTools
	using PartitionedStructures

	include("old_version/PartiallySeparableStructure.jl")
	include("partitioned_data/_include.jl")

	using .Mod_ab_partitioned_data, .Mod_PBFGS, .Mod_PLBFGS, .Mod_PQN
	using .Mod_partitionedNLPModel

	export deduct_partially_separable_structure, evaluate_SPS, untype_evaluate_SPS, evaluate_SPS2, evaluate_SPS_gradient!, build_gradient!, minus_grad_vec!, id_hessian!, product_matrix_sps
	export element_function, SPS, element_hessian, Hess_matrix, element_gradient, grad_vector #types

	export PartitionedData
	export PartitionedData_TR_PBFGS, PartitionedData_TR_PLBFGS
	export build_PartitionedData_TR_PBFGS, build_PartitionedData_TR_PLBFGS, build_PartitionedData_TR_PQN

	export PartitionedNLPModel
	export PBFGSNLPModel, PLBFGSNLPModel, PQNNLPModel

	export product_part_data_x, evaluate_obj_part_data, evaluate_grad_part_data
	export product_part_data_x!, evaluate_obj_part_data!, evaluate_y_part_data!, evaluate_grad_part_data!
	export get_n, get_N, get_vec_elt_fun, get_M, get_vec_elt_complete_expr_tree, get_element_expr_tree_table, get_index_element_tree, get_vec_compiled_element_gradients
	export get_x, get_v, get_s, get_pg, get_pv, get_ps, get_pB, get_fx
	export set_n!, set_N!, set_vec_elt_fun!, set_M!, set_vec_elt_complete_expr_tree!, set_element_expr_tree_table!, set_index_element_tree!, set_vec_compiled_element_gradients!
	export set_x!, set_v!, set_s!, set_pg!, set_pv!, set_ps!, set_pg!, set_pv!, set_ps!, set_pB!, set_fx!
	export update_nlp!

	export update_PBFGS, update_PBFGS!, update_PLBFGS, update_PLBFGS!, update_PQN, update_PQN!
	
end # module
