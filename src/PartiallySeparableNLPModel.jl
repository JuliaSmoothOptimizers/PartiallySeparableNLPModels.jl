module PartiallySeparableNLPModel

using CalculusTreeTools

fonction_test(e :: Any, x :: AbstractVector) = CalculusTreeTools.evaluate_expr_tree(e,x)
fonction_test2(e :: Any) = CalculusTreeTools.transform_to_expr_tree(e)

greet() = print("Hello World!")


include("PartiallySeparablestructure.jl")

export function_test, function_test2

export deduct_partially_separable_structure
end # module
