module PartiallySeparableNLPModels

using CalculusTreeTools
using PartitionedStructures

include("PartiallySeparableStructure.jl")
include("partitioned_data/_include.jl")

export deduct_partially_separable_structure, evaluate_SPS, untype_evaluate_SPS, evaluate_SPS2, evaluate_SPS_gradient!, build_gradient!, minus_grad_vec!, id_hessian!, product_matrix_sps
export element_function, SPS, element_hessian, Hess_matrix, element_gradient, grad_vector #types

export PartitionedData_TR_BFGS
export build_PartitionedData_TR_BFGS, evaluate_obj_pd_pbfgs, evaluate_obj_pd_pbfgs!, evaluate_grad_pd_pbfgs!, evaluate_grad_pd_pbfgs

end # module
