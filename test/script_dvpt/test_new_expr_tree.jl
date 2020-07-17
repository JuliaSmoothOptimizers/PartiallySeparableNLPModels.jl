using JuMP, MathOptInterface, LinearAlgebra
using CalculusTreeTools, PartiallySeparableNLPModel

using BenchmarkTools

using Base.Threads

m = Model()
n = 10
@variable(m, x[1:n])
@NLobjective(m, Min, sum( 100 * (x[j-1]^2 - x[j])^2 + (x[j-1] - 1)^2  for j in 2:n)) #rosenbrock function
evaluator = JuMP.NLPEvaluator(m)
MathOptInterface.initialize(evaluator, [:ExprGraph])
Expr_j = MathOptInterface.objective_expr(evaluator)
expr_tree_j = CalculusTreeTools.transform_to_expr_tree(Expr_j)
complete_expr_tree = CalculusTreeTools.create_complete_tree(expr_tree_j)
x = ones(n)

arbre_test_cast = copy(complete_expr_tree)
@show @elapsed  CalculusTreeTools.cast_type_of_constant(arbre_test_cast, Float64)


# @code_warntype PartiallySeparableNLPModel.deduct_partially_separable_structure(complete_expr_tree, n)



expr_Expr = :(x[1] + x[2] *5 + x[7]/10)
res = CalculusTreeTools.cast_type_of_constant(expr_Expr, Float32)
@show res, expr_Expr


# sps1 = PartiallySeparableNLPModel.deduct_partially_separable_structure(Expr_j, n)
# sps2 = PartiallySeparableNLPModel.deduct_partially_separable_structure(expr_tree_j, n)
sps3 = PartiallySeparableNLPModel.deduct_partially_separable_structure(complete_expr_tree, n)
# time_deduct = @elapsed (sps3 = PartiallySeparableNLPModel.deduct_partially_separable_structure(complete_expr_tree, n))
# @show time_deduct
error("fin")
res_original = PartiallySeparableNLPModel.evaluate_SPS(sps3, x)
res_new = PartiallySeparableNLPModel.evaluate_SPS2(sps3, x)
# res_view = PartiallySeparableNLPModel.evaluate_SPS3(sps3, x)
bench_deduct = @benchmark PartiallySeparableNLPModel.deduct_partially_separable_structure(complete_expr_tree, n)
bench_original = @benchmark PartiallySeparableNLPModel.evaluate_SPS(sps3, x)
bench_new = @benchmark PartiallySeparableNLPModel.evaluate_SPS2(sps3, x)
# bench_view = @benchmark PartiallySeparableNLPModel.evaluate_SPS3(sps3, x)

# @code_warntype PartiallySeparableNLPModel.evaluate_SPS3(sps3, x)
# @code_warntype PartiallySeparableNLPModel.evaluate_SPS2(sps3, x)
# @code_warntype PartiallySeparableNLPModel.evaluate_SPS(sps3, x)

# time_eval = @elapsed PartiallySeparableNLPModel.evaluate_SPS(sps3, x)
# @show time_eval
# time_eval2 = @elapsed PartiallySeparableNLPModel.evaluate_SPS2(sps3, x)
# @show time_eval2
# time_eval3 = @elapsed PartiallySeparableNLPModel.evaluate_SPS3(sps3, x)
# @show time_eval3


@show res_new - res_original
@show res_new, res_original
# @show Base.summarysize(sps3)
# bench_memory = @benchmark (sps3.x = ones(n))

# test = ones(100000)
# view_test = view(test, [1:10000;])
# view_test2 = view(test, [25555:25558;])
# @benchmark Array(view_test)
# @benchmark Array(view_test2)


# @show (Array{SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true},1} where T <: Number <: Array{SubArray{T,1,Array{T,1},N,true},1} where N where T <: Number)
# @show Array{SubArray{T,1,Array{T,1},Tuple{Array{Int64,1}},false},1} <: Array{SubArray{T,1,Array{T,1},N,true},1} where N
