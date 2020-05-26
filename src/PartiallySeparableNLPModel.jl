module PartiallySeparableNLPModel

using CalculusTreeTools

fonction_test(e :: Any, x :: AbstractVector) = CalculusTreeTools.evaluate_expr_tree(e,x)
fonction_test2(e :: Any) = CalculusTreeTools.transform_to_expr_tree(e)

greet() = print("Hello World!")

export function_test, function_test2
end # module
