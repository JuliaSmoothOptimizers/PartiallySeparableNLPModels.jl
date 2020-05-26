using JuMP, MathOptInterface



m = Model()
n = 1000
@variable(m, x[1:n])
@NLobjective(m, Min, sum( x[j] * x[j+1] + tan(x[j+1]) for j in 1:n-1 ) + (sin(x[1]))^2 + x[n-1]^3 )
evaluator = JuMP.NLPEvaluator(m)
MathOptInterface.initialize(evaluator, [:ExprGraph])
Expr_j = MathOptInterface.objective_expr(evaluator)
expr_tree_j = PartiallySeparableNLPModel.fonction_test2(Expr_j)


x = ones(n)
res_Expr = PartiallySeparableNLPModel.fonction_test(Expr_j, x)
res_expr_tree = PartiallySeparableNLPModel.fonction_test(expr_tree_j, x)


sps1 = PartiallySeparableNLPModel.deduct_partially_separable_structure(Expr_j, n)
sps2 = PartiallySeparableNLPModel.deduct_partially_separable_structure(expr_tree_j, n)
θ = 1e-6
@testset "premiers tests" begin
    @test res_expr_tree == res_Expr
    @test PartiallySeparableNLPModel.evaluate_SPS(sps1, x) == PartiallySeparableNLPModel.evaluate_SPS(sps2, x)
    @test PartiallySeparableNLPModel.evaluate_SPS(sps2, x) - MathOptInterface.eval_objective( evaluator, x) < θ
end
# Juno.@run PartiallySeparableNLPModel.deduct_partially_separable_structure(Expr_j, n)
