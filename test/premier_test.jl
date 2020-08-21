using JuMP, MathOptInterface, LinearAlgebra
using CalculusTreeTools

using PartiallySeparableNLPModel
using BenchmarkTools, ProfileView, Test

m = Model()
n = 500
@variable(m, x[1:n])
# @NLobjective(m, Min, sum( x[j] * x[j+1] + tan(x[j+1]) for j in 1:n-1 ) + (sin(x[1]))^2 + x[n-1]^3 )
@NLobjective(m, Min, sum( x[j] * x[j+1] + (x[j+1]+5*x[j])^2 for j in 1:n-1 ) + (sin(x[1]))^2 + x[n-1]^3 )
evaluator = JuMP.NLPEvaluator(m)
MathOptInterface.initialize(evaluator, [:ExprGraph])
Expr_ = MathOptInterface.objective_expr(evaluator)
expr_tree = CalculusTreeTools.transform_to_expr_tree(Expr_)
complete_expr_tree = CalculusTreeTools.create_complete_tree(expr_tree)
x = ones(n)
x = Vector{Float64}([1.0:n;])


sps1 = PartiallySeparableNLPModel.deduct_partially_separable_structure(Expr_, n)
sps2 = PartiallySeparableNLPModel.deduct_partially_separable_structure(expr_tree, n)
sps3 = PartiallySeparableNLPModel.deduct_partially_separable_structure(complete_expr_tree, n)


# error("test")
θ = 1e-6
obj1 = PartiallySeparableNLPModel.evaluate_SPS(sps1, x)
obj2 = PartiallySeparableNLPModel.evaluate_SPS(sps2, x)
obj3 = PartiallySeparableNLPModel.evaluate_SPS(sps3, x)
obj4 = PartiallySeparableNLPModel.evaluate_obj_pre_compiled(sps1, x)
# Juno.@run PartiallySeparableNLPModel.evaluate_obj_pre_compiled(sps1, x)
@show obj1, obj4


    # dump(sps1.obj_pre_compiled_trees[1].racine.children[1])
    # sps1.obj_pre_compiled_trees[1].racine.children[1].field.op.multiple_x_view
    # sps1.obj_pre_compiled_trees[1].racine.children[1].field.op.multiple_x_view[1].parent
    #
    # sps1.obj_pre_compiled_trees[1].multiple_x
    #
    # sps1.obj_pre_compiled_trees[1].racine.children[1].field.op.multiple_x_view[1].parent === sps1.obj_pre_compiled_trees[1].multiple_x[1]
    #
    # sps1.obj_pre_compiled_trees[1].multiple_x[1] .= [1.0,2.0]


# bench_new = @benchmark PartiallySeparableNLPModel.evaluate_SPS(sps1, x)
# bench_old = @benchmark PartiallySeparableNLPModel.evaluate_SPS2(sps1, x)
# @profview (@benchmark PartiallySeparableNLPModel.evaluate_obj_pre_compiled(sps1, x))
# error("stop")

grad1 =  PartiallySeparableNLPModel.evaluate_gradient(sps1, x)
grad2 =  PartiallySeparableNLPModel.evaluate_gradient(sps2, x)
grad3 = PartiallySeparableNLPModel.evaluate_gradient(sps3, x)
moi_grad = similar(x)
MathOptInterface.eval_objective_gradient(evaluator, moi_grad, x)





@testset "tests évaluation" begin
    @test PartiallySeparableNLPModel.evaluate_SPS(sps1, x) == PartiallySeparableNLPModel.evaluate_SPS(sps2, x)
    @test PartiallySeparableNLPModel.evaluate_SPS(sps2, x) == PartiallySeparableNLPModel.evaluate_SPS(sps3, x)
    @test PartiallySeparableNLPModel.evaluate_SPS(sps2, x) - MathOptInterface.eval_objective( evaluator, x) < θ
end

@testset "tests gradient" begin
    MOI_gradient = Vector{ eltype(x) }(undef,n)
    MathOptInterface.eval_objective_gradient(evaluator, MOI_gradient, x)

    @test PartiallySeparableNLPModel.evaluate_gradient(sps1, x) == PartiallySeparableNLPModel.evaluate_gradient(sps2, x)
    @test PartiallySeparableNLPModel.evaluate_gradient(sps2, x) == PartiallySeparableNLPModel.evaluate_gradient(sps3, x)
    @test norm(PartiallySeparableNLPModel.evaluate_gradient(sps2, x) - MOI_gradient) < θ
end
