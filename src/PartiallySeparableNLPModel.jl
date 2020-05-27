module PartiallySeparableNLPModel

using CalculusTreeTools

fonction_test(e :: Any, x :: AbstractVector) = CalculusTreeTools.evaluate_expr_tree(e,x)
fonction_test2(e :: Any) = CalculusTreeTools.transform_to_expr_tree(e)

greet() = print("Hello World!")


include("PartiallySeparableStructure.jl")

export function_test, function_test2

export deduct_partially_separable_structure

export evaluate_SPS, evaluate_SPS_gradient!, build_gradient!, minus_grad_vec!, id_hessian!, product_matrix_sps

# export les types nécessaires
export element_function, SPS, element_hessian, Hess_matrix, element_gradient, grad_vector
end # module
