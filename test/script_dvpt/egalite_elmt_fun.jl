using PartiallySeparableNLPModel
using JuMP, MathOptInterface, LinearAlgebra, SparseArrays

using CalculusTreeTools


function create_initial_point_Rosenbrock(n)
    point_initial = Vector{Float64}(undef, n)
    for i in 1:n
        if mod(i,2) == 1
            point_initial[i] = -1.2
        elseif mod(i,2) == 0
            point_initial[i] = 1.0
        else
            error("bizarre")
        end
    end
    return point_initial
end

function create_Rosenbrock_JuMP_Model(n :: Int)
    m = Model()
    @variable(m, x[1:n])
    @NLobjective(m, Min, sum( 100 * (x[j-1]^2 - x[j])^2 + (x[j-1] - 1)^2  for j in 2:n)) #rosenbrock function
    evaluator = JuMP.NLPEvaluator(m)
    MathOptInterface.initialize(evaluator, [:ExprGraph, :Hess])
    obj = MathOptInterface.objective_expr(evaluator)
    vec_var = JuMP.all_variables(m)
    vec_value = create_initial_point_Rosenbrock(n)
    JuMP.set_start_value.(vec_var, vec_value)
    return (m, evaluator,obj)
end

function create_chained_wood_JuMP_Model(n :: Int)
    σ = 10e-5
    m = Model()
    @variable(m, x[1:n])
    @NLobjective(m, Min, sum( 100 * (x[2*j-1]^2 - x[2*j])^2 + (x[2*j-1] - 1)^2 + 90 * (x[2*j+1]^2 - x[2*j+2])^2 + (x[2*j+1] - 1)^2 + 10 * (x[2*j] + x[2*j+2] - 2)^2 + (x[2*j] - x[2*j+2])^2 * 0.1  for j in 1:Integer((n-2)/2) )) #rosenbrock function
    evaluator = JuMP.NLPEvaluator(m)
    MathOptInterface.initialize(evaluator, [:ExprGraph, :Hess])
    obj = MathOptInterface.objective_expr(evaluator)
    vec_var = JuMP.all_variables(m)
    vec_value = create_initial_point_chained_wood(n)
    JuMP.set_start_value.(vec_var, vec_value)
    return (m, evaluator,obj)
end


function create_initial_point_chained_wood(n)
    point_initial = Vector{Float64}(undef, n)
    for i in 1:n
        if i <= 4 && mod(i,2) == 1
            point_initial[i] = -3
        elseif i <= 4 && mod(i,2) == 0
            point_initial[i] = -1
        elseif i > 4 && mod(i,2) == 1
            point_initial[i] = -2
        elseif i > 4 && mod(i,2) == 0
            point_initial[i] = 0
        else
            error("bizarre")
        end
    end
    return point_initial
end


# function get_different_CalculusTree(sps :: PartiallySeparableNLPModel.SPS{T}) where T
#     structure = sps.structure
#     elmt_fun = ( x -> PartiallySeparableNLPModel.get_fun(x)).(structure)
#     work_elmt_fun = copy(elmt_fun)
#     different_calculus_tree = Vector{T}(undef,0)
#
#     while isempty(work_elmt_fun) == false
#         current_tree = work_elmt_fun[1]
#         push!(different_calculus_tree, current_tree)
#         work_elmt_fun = filter( (x -> x != current_tree) , work_elmt_fun)
#     end
#
#     different_calculus_tree_index = Vector{Int}(undef, length(elmt_fun))
#     for i in 1:length(elmt_fun)
#         for j in 1:length(different_calculus_tree)
#             if elmt_fun[i] == different_calculus_tree[j]
#                 different_calculus_tree_index[i] = j
#                 break
#             end
#         end
#     end
#     @show work_elmt_fun, different_calculus_tree, length(different_calculus_tree)
#     @show different_calculus_tree_index
#     return (different_calculus_tree, different_calculus_tree_index)
# end

n = 100000
(m, evaluator,obj) = create_Rosenbrock_JuMP_Model(n)

x = ones(n)
y = zeros(n)
rdm = rand(n)

expr_tree_obj = CalculusTreeTools.transform_to_expr_tree(obj)
comp_ext = CalculusTreeTools.create_complete_tree(expr_tree_obj)

# détection de la structure partiellement séparable
SPS1 = PartiallySeparableNLPModel.deduct_partially_separable_structure(obj, n)

obj2 = CalculusTreeTools.transform_to_expr_tree(obj)
SPS2 = PartiallySeparableNLPModel.deduct_partially_separable_structure(obj2, n)

SPS3 = PartiallySeparableNLPModel.deduct_partially_separable_structure(comp_ext, n)


# comparison_tree1 = SPS2.structure[1].fun
# comparison_tree2 = SPS2.structure[2].fun
# global cpt1 = 0
# global cpt2 = 0
# for i in 3:length(SPS2.structure)
#     if SPS2.structure[i].fun == comparison_tree1
#         global cpt1 = cpt1 + 1
#     elseif SPS2.structure[i].fun == comparison_tree2
#         global cpt2 = cpt2 + 1
#     end
# end
# @show cpt1, cpt2
# @show sizeof(SPS1), sizeof(SPS2), sizeof(SPS3)
@show Base.summarysize(SPS1), Base.summarysize(SPS2), Base.summarysize(SPS3)
# @show Base.summarysize(SPS3)
# get_different_CalculusTree(SPS2)
# get_different_CalculusTree(SPS3)

#
# println("\n\n\n\n\n\n\n test sur chained_wood \n\n\n\n\n\n\n ")
#
#
# n = 10
# (m, evaluator,obj) = create_chained_wood_JuMP_Model(n)
#
# x = ones(n)
# y = zeros(n)
# rdm = rand(n)
#
# expr_tree_obj = CalculusTreeTools.transform_to_expr_tree(obj)
# comp_ext = CalculusTreeTools.create_complete_tree(expr_tree_obj)
#
# # détection de la structure partiellement séparable
# SPS1 = PartiallySeparableNLPModel.deduct_partially_separable_structure(obj, n)
#
# obj2 = CalculusTreeTools.transform_to_expr_tree(obj)
# SPS2 = PartiallySeparableNLPModel.deduct_partially_separable_structure(obj2, n)
#
# SPS3 = PartiallySeparableNLPModel.deduct_partially_separable_structure(comp_ext, n)
#
# # get_different_CalculusTree(SPS2)
# # get_different_CalculusTree(SPS3)
#
# @show Base.summarysize(SPS1), Base.summarysize(SPS2), Base.summarysize(SPS3)
@show Base.summarysize(SPS3)
@show Base.summarysize(SPS3.structure)
@show Base.summarysize(SPS3.structure[1])
@show Base.summarysize(SPS3.different_element_tree)
@show Base.summarysize(SPS3.compiled_gradients)
