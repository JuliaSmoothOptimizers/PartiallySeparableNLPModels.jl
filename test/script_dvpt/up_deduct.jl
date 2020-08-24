using JuMP, MathOptInterface, NLPModelsJuMP, LinearAlgebra, SparseArrays, Test, JSOSolvers, BenchmarkTools,  OptimizationProblems, NLPModels
using CalculusTreeTools, PartiallySeparableNLPModel




# n = 1000
# jump_model = OptimizationProblems.dixmaane(n - (n %3))
#
# println("fin modèle jump")
# math_model = NLPModelsJuMP.MathOptNLPModel(jump_model)
# println("fin modèle math")
# partionned_nlp =  PartiallySeparableSolvers.PartionnedNLPModel(math_model)
# println("fin Partionned model")


println("début")
σ(a,b) = abs(a-b) < 1e-5

# n_array = [1000,2000,4000]
n_array = [100, 200, 400, 800]
# n_array = [100, 200]
res = Vector{BenchmarkTools.Trial}(undef,0)
for i in n_array
    i_f = i - (i %3)
    jump_model = OptimizationProblems.dixmaane(i_f)
    evaluator = JuMP.NLPEvaluator(jump_model)
    MathOptInterface.initialize(evaluator, [:ExprGraph])
    Expr_ = MathOptInterface.objective_expr(evaluator)
    println("fin modèle jump")
    @time sps = PartiallySeparableNLPModel._deduct_partially_separable_structure(Expr_, i_f)
    println("fin deduct")
    # bench_tmp = @benchmark PartiallySeparableNLPModel.deduct_partially_separable_structure($Expr_, $i_f)
    # push!(res, bench_tmp)

    x = ones(i_f)
    obj_SPS = PartiallySeparableNLPModel.evaluate_SPS( sps, x)
    obj_MOI = MathOptInterface.eval_objective( evaluator, x)
    @test σ(obj_SPS, obj_MOI)
end

# using ProfileView
println("début après boucle")
i = 500
i_final = i - (i %3)
jump_model = OptimizationProblems.dixmaane(i_final)
evaluator = JuMP.NLPEvaluator(jump_model)
MathOptInterface.initialize(evaluator, [:ExprGraph])
Expr_ = MathOptInterface.objective_expr(evaluator)

sps = PartiallySeparableNLPModel._deduct_partially_separable_structure(Expr_, i_final)
time = @time PartiallySeparableNLPModel._deduct_partially_separable_structure(Expr_, i_final)
println("fin Partionned model")

x = ones(i_final)
obj_SPS = PartiallySeparableNLPModel.evaluate_SPS( sps, x)
obj_MOI = MathOptInterface.eval_objective( evaluator, x)
@test σ(obj_SPS, obj_MOI)


# a = (x -> x+1).([1:10;])
# all(i -> i >2 , a)
