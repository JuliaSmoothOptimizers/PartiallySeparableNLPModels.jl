using JuMP, MathOptInterface, LinearAlgebra
using CalculusTreeTools


m = Model()
n = 5
@variable(m, x[1:n])
# @NLobjective(m, Min, sum( x[j] * x[j+1] + tan(x[j+1]) for j in 1:n-1 ) + (sin(x[1]))^2 + x[n-1]^3 )
@NLobjective(m, Min, sum( x[j] * x[j+1] + (x[j+1])^2 for j in 1:n-1 ) + (sin(x[1]))^2 + x[n-1]^3 )
evaluator = JuMP.NLPEvaluator(m)
MathOptInterface.initialize(evaluator, [:ExprGraph])
Expr_j = MathOptInterface.objective_expr(evaluator)
expr_tree_j = CalculusTreeTools.transform_to_expr_tree(Expr_j)
complete_expr_tree = CalculusTreeTools.create_complete_tree(expr_tree_j)
x = ones(n)



sps1 = PartiallySeparableNLPModel.deduct_partially_separable_structure(Expr_j, n)
sps2 = PartiallySeparableNLPModel.deduct_partially_separable_structure(expr_tree_j, n)
sps3 = PartiallySeparableNLPModel.deduct_partially_separable_structure(complete_expr_tree, n)
θ = 1e-6
PartiallySeparableNLPModel.evaluate_SPS(sps3, x)
PartiallySeparableNLPModel.evaluate_SPS(sps2, x)
# PartiallySeparableNLPModel.evaluate_SPS(sps1, x)
@testset "tests évaluation" begin
    # @test PartiallySeparableNLPModel.evaluate_SPS(sps1, x) == PartiallySeparableNLPModel.evaluate_SPS(sps2, x)
    @test PartiallySeparableNLPModel.evaluate_SPS(sps2, x) == PartiallySeparableNLPModel.evaluate_SPS(sps3, x)
    @test PartiallySeparableNLPModel.evaluate_SPS(sps2, x) - MathOptInterface.eval_objective( evaluator, x) < θ
end

@testset "tests gradient" begin
    MOI_gradient = Vector{ typeof(x[1]) }(undef,n)
    MathOptInterface.eval_objective_gradient(evaluator, MOI_gradient, x)

    # @test PartiallySeparableNLPModel.evaluate_gradient(sps1, x) == PartiallySeparableNLPModel.evaluate_gradient(sps2, x)
    @test PartiallySeparableNLPModel.evaluate_gradient(sps2, x) == PartiallySeparableNLPModel.evaluate_gradient(sps3, x)
    @test norm(PartiallySeparableNLPModel.evaluate_gradient(sps2, x) - MOI_gradient) < θ
end

# Juno.@run PartiallySeparableNLPModel.deduct_partially_separable_structure(Expr_j, n)
