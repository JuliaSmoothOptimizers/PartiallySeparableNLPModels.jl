using BenchmarkTools
using JSOSolvers, SolverBenchmark, SolverTools
using NLPModelsJuMP
using JuMP, MathOptInterface

using CalculusTreeTools
using PartiallySeparableNLPModels
# using ..My_SPS_Model_Module


const SUITE = BenchmarkGroup()


n = [100,200,500,1000,2000,5000]
# n = [10,20,30]

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


problems = create_Rosenbrock_JuMP_Model.(n)



SUITE["SPS_function"] = BenchmarkGroup()


for p in problems
  (m_ros, evaluator, obj_ros) = p
  obj_ros_expr_tree = CalculusTreeTools.transform_to_expr_tree(obj_ros)

  n = m_ros.moi_backend.model_cache.model.num_variables_created
  SPS_ros = PartiallySeparableNLPModels.deduct_partially_separable_structure(obj_ros_expr_tree, n)
  x = ones(n)
  #calcul de la fonction objectif
  SUITE["SPS_function"]["OBJ ros $n var"] = @benchmarkable PartiallySeparableNLPModels.evaluate_SPS($SPS_ros, $x)

  #calcul du gradient sous format gradient élémentaire
  f = (y :: PartiallySeparableNLPModels.element_function -> PartiallySeparableNLPModels.element_gradient{typeof(x[1])}(Vector{typeof(x[1])}(zeros(typeof(x[1]), length(y.used_variable)) )) )
  grad = PartiallySeparableNLPModels.grad_vector{typeof(x[1])}( f.(SPS_ros.structure) )
  SUITE["SPS_function"]["grad ros $n var"] = @benchmarkable PartiallySeparableNLPModels.evaluate_SPS_gradient!($SPS_ros, $x, $grad)

  #calcul du Hessien
  f = ( elm_fun :: PartiallySeparableNLPModels.element_function -> PartiallySeparableNLPModels.element_hessian{Float64}( Array{Float64,2}(undef, length(elm_fun.used_variable), length(elm_fun.used_variable) )) )
  t = f.(SPS_ros.structure) :: Vector{PartiallySeparableNLPModels.element_hessian{typeof(x[1])}}
  H = PartiallySeparableNLPModels.Hess_matrix{typeof(x[1])}(t)

  SUITE["SPS_function"]["Hessien ros $n var"] = @benchmarkable PartiallySeparableNLPModels.struct_hessian!($SPS_ros, $x, $H)

end

# const atol = 1.0e-5
# const rtol = 1.0e-6
# const max_time = 300.0
# max_eval = 5000
#
# nlp_problems = MathOptNLPModel.([p[1] for p in problems])
