module PartiallySeparableNLPModel

using CalculusTreeTools


include("PartiallySeparableStructure.jl")


export deduct_partially_separable_structure

export evaluate_SPS, evaluate_SPS2, evaluate_SPS_gradient!, build_gradient!, minus_grad_vec!, id_hessian!, product_matrix_sps

# export les types nécessaires
export element_function, SPS, element_hessian, Hess_matrix, element_gradient, grad_vector
end # module
