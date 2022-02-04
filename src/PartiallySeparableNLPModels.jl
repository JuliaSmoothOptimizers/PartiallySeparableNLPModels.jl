module PartiallySeparableNLPModels

using CalculusTreeTools

export deduct_partially_separable_structure, evaluate_SPS, untype_evaluate_SPS, evaluate_SPS2, evaluate_SPS_gradient!, build_gradient!, minus_grad_vec!, id_hessian!, product_matrix_sps
export element_function, SPS, element_hessian, Hess_matrix, element_gradient, grad_vector #types

include("PartiallySeparableStructure.jl")

end # module
