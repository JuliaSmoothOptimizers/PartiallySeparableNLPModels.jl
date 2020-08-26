using JuMP, MathOptInterface, NLPModelsJuMP, LinearAlgebra, SparseArrays, Test, JSOSolvers, BenchmarkTools,  OptimizationProblems, NLPModels
using CalculusTreeTools, PartiallySeparableNLPModel




println("début")
i = 20
i_final = i - (i %3)
jump_model = OptimizationProblems.dixmaane(i_final)
evaluator = JuMP.NLPEvaluator(jump_model)
MathOptInterface.initialize(evaluator, [:ExprGraph])
Expr_ = MathOptInterface.objective_expr(evaluator)

# convert(Function, Expr_)


# error("stop")
sps = PartiallySeparableNLPModel._deduct_partially_separable_structure(Expr_, i_final)
println("fin Partionned model")

x = ones(i_final)
# x = rand(i_final)
# x2 = (xi -> 2 * xi).(ones(i_final))


# vec_fun = PartiallySeparableNLPModel.get_vector_function(sps)


obj_sps = PartiallySeparableNLPModel.evaluate_SPS(sps, x)
obj_sps2 = PartiallySeparableNLPModel.evaluate_function(sps, x)
obj_jump = MathOptInterface.eval_objective(evaluator, x)

@show obj_sps
@show obj_jump

println("début bench")
b = false
if b
    bench_obj_SPS2 = @benchmark PartiallySeparableNLPModel.evaluate_function( $sps, $x)
    bench_obj_SPS = @benchmark PartiallySeparableNLPModel.evaluate_SPS( $sps, $x)
    bench_obj_MOI = @benchmark MathOptInterface.eval_objective( $evaluator, $x)
end

error("pour le bruit")
