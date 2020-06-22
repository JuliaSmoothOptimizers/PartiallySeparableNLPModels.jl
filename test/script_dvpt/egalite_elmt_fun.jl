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


n = 500
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


comparison_tree1 = SPS2.structure[1].fun
comparison_tree2 = SPS2.structure[2].fun
global cpt1 = 0
global cpt2 = 0
for i in 3:length(SPS2.structure)
    if SPS2.structure[i].fun == comparison_tree1
        global cpt1 = cpt1 + 1
    elseif SPS2.structure[i].fun == comparison_tree2
        global cpt2 = cpt2 + 1
    end
end
@show cpt1, cpt2
@show sizeof(SPS1), sizeof(SPS2), sizeof(SPS3)
@show Base.summarysize(SPS1), Base.summarysize(SPS2), Base.summarysize(SPS3)
