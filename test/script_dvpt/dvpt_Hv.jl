using Revise
using JuMP, MathOptInterface, LinearAlgebra
using CalculusTreeTools, PartiallySeparableNLPModel
using Test, BenchmarkTools

using ReverseDiff

using BenchmarkTools

using Base.Threads

m = Model()
n = 10000
@variable(m, x[1:n])
@NLobjective(m, Min, sum( 100 * (x[j-1]^2 - x[j])^2 + (x[j-1] - 1)^2  for j in 2:n)) #rosenbrock function
evaluator = JuMP.NLPEvaluator(m)
MathOptInterface.initialize(evaluator, [:ExprGraph, :Hess])
Expr_j = MathOptInterface.objective_expr(evaluator)
expr_tree_j = CalculusTreeTools.transform_to_expr_tree(Expr_j)
complete_expr_tree = CalculusTreeTools.create_complete_tree(expr_tree_j)
x = ones(n)

sps = PartiallySeparableNLPModel.deduct_partially_separable_structure(complete_expr_tree, n)





println("\n\n\n\n\n")
x = (x -> 3*x).(ones(n))
f = (y :: PartiallySeparableNLPModel.element_function -> PartiallySeparableNLPModel.element_gradient{typeof(x[1])}(Vector{typeof(x[1])}(zeros(typeof(x[1]), length(y.used_variable)) )) )
p_grad = PartiallySeparableNLPModel.grad_vector{typeof(x[1])}( f.(sps.structure) )
v = (x -> 2*x).(ones(n))
v = zeros(n)
# v = (y -> 5 * y).(ones(n))
v[1] = 1




res_total = PartiallySeparableNLPModel.Hv(sps,x,v)
x_MOI_Hessian_y = similar(x)
MathOptInterface.eval_hessian_lagrangian_product(evaluator, x_MOI_Hessian_y, x, v, 1.0, zeros(0))

@show norm( res_total - x_MOI_Hessian_y)

bench_jump = @benchmark MathOptInterface.eval_hessian_lagrangian_product(evaluator, x_MOI_Hessian_y, x, v, 1.0, zeros(0))
hv = similar(x)
bench_sps =  @benchmark PartiallySeparableNLPModel.Hv!(hv, sps, x, v)
# # bench_elemental = @benchmark ∇²fv!(tree1, x, v, res)
#
error("pause")
#
# hess_mat = PartiallySeparableNLPModel.hess(sps, x)
# @show hess_mat
# res = similar(x)
# PartiallySeparableNLPModel.product_matrix_sps!(sps, hess_mat, v, res)


# PartiallySeparableNLPModel.Hv!(sps, x, p_grad, v)
# hv = PartiallySeparableNLPModel.build_gradient(sps, p_grad)
#
#
#
# @show hv
# @show x_MOI_Hessian_y
# @show p_grad





# Hess_matrix_ones = PartiallySeparableNLPModel.struct_hessian(sps, x)
# storage_matrix = PartiallySeparableNLPModel.struct_hessian(sps, x)
# Bx1 = PartiallySeparableNLPModel.product_matrix_sps(sps, Hess_matrix_ones, x)
# Bx2 = PartiallySeparableNLPModel.Hv_only_product(sps, Hess_matrix_ones, x )
# res = similar(x)
# res2 = similar(x)
# bench1 = @benchmark PartiallySeparableNLPModel.product_matrix_sps!(sps, Hess_matrix_ones, x, res)
# bench2 = @benchmark PartiallySeparableNLPModel.Hv_only_product!(sps, Hess_matrix_ones, x, res2 )
# @test Bx1 == Bx2


#
# Juno.@run  PartiallySeparableNLPModel.Hv_only_product(sps, Hess_matrix_ones, x )
# structure = PartiallySeparableNLPModel.get_structure(sps)
# different_element_tree = PartiallySeparableNLPModel.get_different_element_tree(sps)
# vector_index = PartiallySeparableNLPModel.get_elmnt_fun_index_view(structure, different_element_tree)
# Juno.@run  PartiallySeparableNLPModel.get_elmnt_fun_index_view(structure, different_element_tree)
#
# # res = similar(x)
# bench1 = @benchmark PartiallySeparableNLPModel.product_matrix_sps!(sps, Hess_matrix_ones, x, res)
#
# res2 = similar(x)
# bench2 = @benchmark PartiallySeparableNLPModel.Hv_only_product!(sps, Hess_matrix_ones, x, res2 )
#
# PartiallySeparableNLPModel.test_parcours(sps, Hess_matrix_ones)
#
# bench6 = @benchmark PartiallySeparableNLPModel.test_parcours(sps, Hess_matrix_ones)
# bench1 = @benchmark PartiallySeparableNLPModel.product_matrix_sps(sps, Hess_matrix_ones, x)
# bench2 = @benchmark PartiallySeparableNLPModel.Hv_only_product(sps, Hess_matrix_ones, x)


# @code_warntype PartiallySeparableNLPModel.product_matrix_sps!(sps, Hess_matrix_ones, x, res)
# @code_warntype PartiallySeparableNLPModel.Hv_only_product2!(sps, Hess_matrix_ones, x, Bᵢxᵢ )
 # @code_warntype PartiallySeparableNLPModel.Hv_only_product2!(sps, Hess_matrix_ones, x, Bᵢxᵢ )
