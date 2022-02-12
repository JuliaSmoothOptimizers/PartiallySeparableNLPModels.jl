module PartiallySeparableNLPModels

	using CalculusTreeTools
	using PartitionedStructures

	include("old_version/PartiallySeparableStructure.jl")
	include("partitioned_data/_include.jl")

	using .Mod_ab_partitioned_data, .Mod_PBFGS, .Mod_PLBFGS

	export deduct_partially_separable_structure, evaluate_SPS, untype_evaluate_SPS, evaluate_SPS2, evaluate_SPS_gradient!, build_gradient!, minus_grad_vec!, id_hessian!, product_matrix_sps
	export element_function, SPS, element_hessian, Hess_matrix, element_gradient, grad_vector #types

	export PartitionedData_TR_PBFGS, PartitionedData_TR_PLBFGS
	export build_PartitionedData_TR_PBFGS, build_PartitionedData_TR_PLBFGS
	export product_part_data_x, product_part_data_x!, evaluate_obj_part_data, evaluate_obj_part_data!, evaluate_y_part_data!, evaluate_grad_part_data, evaluate_grad_part_data!
	export update_PBFGS, update_PBFGS!, update_PLBFGS, update_PLBFGS!
	
end # module
