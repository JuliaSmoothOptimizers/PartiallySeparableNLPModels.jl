using PartiallySeparableNLPModel
using JuMP, MathOptInterface, LinearAlgebra, SparseArrays
using BenchmarkTools, ProfileView

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



n = 100000
(m, evaluator,obj) = create_Rosenbrock_JuMP_Model(n)

x = ones(n)
y = zeros(n)
rdm = rand(n)

expr_tree_obj = CalculusTreeTools.transform_to_expr_tree(obj)
comp_ext = CalculusTreeTools.create_complete_tree(expr_tree_obj)

# détection de la structure partiellement séparable
SPS1 = PartiallySeparableNLPModel.deduct_partially_separable_structure(comp_ext, n)


obj_fun = PartiallySeparableNLPModel.evaluate_SPS(SPS1, y)
bench_obj = @benchmark PartiallySeparableNLPModel.evaluate_SPS(SPS1, y)
@profview PartiallySeparableNLPModel.evaluate_SPS(SPS1, y)
