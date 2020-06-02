using BenchmarkTools
using JuMP, MathOptInterface
# using PartiallySeparableStructure

using PartiallySeparableNLPModel
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

const SUITE = BenchmarkGroup()


n = [100,200,500]
# n = (x -> 100*x).([1:50;])

problems = create_Rosenbrock_JuMP_Model.(n)


for p in problems
  (m_ros, evaluator, obj_ros) = p
  n = m_ros.moi_backend.model_cache.model.num_variables_created
  x = ones(n)
  y = ones(n)
  SUITE["ROS $n variable"] = BenchmarkGroup()
  SUITE["ROS $n variable"] = BenchmarkGroup()
  SUITE["ROS $n variable"]["OBJ"] = BenchmarkGroup()
  SUITE["ROS $n variable"]["GRAD"] = BenchmarkGroup()
  SUITE["ROS $n variable"]["Hess"] = BenchmarkGroup()
  SUITE["ROS $n variable"]["Hv"] = BenchmarkGroup()
  # définition des variables nécessaires


  #calcul de la structure partiellement séparable
  obj_ros_expr_tree = CalculusTreeTools.transform_to_expr_tree(obj_ros)
  SPS_ros = PartiallySeparableNLPModel.deduct_partially_separable_structure(obj_ros_expr_tree, n)

  #calcul de la fonction objectif
  SUITE["ROS $n variable"]["OBJ"]["SPS"] = @benchmarkable PartiallySeparableNLPModel.evaluate_SPS($SPS_ros, $x)
  SUITE["ROS $n variable"]["OBJ"]["JuMP"] = @benchmarkable MathOptInterface.eval_objective( $evaluator, $x)

  #calcul du gradient sous format gradient élémentaire
  f = (y :: PartiallySeparableNLPModel.element_function -> PartiallySeparableNLPModel.element_gradient{typeof(x[1])}(Vector{typeof(x[1])}(zeros(typeof(x[1]), length(y.used_variable)) )) )
  grad = PartiallySeparableNLPModel.grad_vector{typeof(x[1])}( f.(SPS_ros.structure) )
  SUITE["ROS $n variable"]["GRAD"]["SPS"] = @benchmarkable PartiallySeparableNLPModel.evaluate_SPS_gradient!($SPS_ros, $x, $grad)

  grad_JuMP = Vector{Float64}(zeros(Float64,n))
  SUITE["ROS $n variable"]["GRAD"]["JuMP"] = @benchmarkable MathOptInterface.eval_objective_gradient($evaluator, $grad_JuMP, $x)


  MOI_pattern = MathOptInterface.hessian_lagrangian_structure(evaluator)
  MOI_value_Hessian = Vector{ typeof(x[1]) }(undef,length(MOI_pattern))
  SUITE["ROS $n variable"]["Hess"]["JuMP"] = @benchmarkable MathOptInterface.eval_hessian_lagrangian($evaluator, $MOI_value_Hessian, $x, 1.0, zeros(0))


  f = ( elm_fun :: PartiallySeparableNLPModel.element_function{} -> PartiallySeparableNLPModel.element_hessian{Float64}( Array{Float64,2}(undef, length(elm_fun.used_variable), length(elm_fun.used_variable) )) )
  t = f.(SPS_ros.structure) :: Vector{PartiallySeparableNLPModel.element_hessian{Float64}}
  H = PartiallySeparableNLPModel.Hess_matrix{Float64}(t)
  SUITE["ROS $n variable"]["Hess"]["SPS"] = @benchmarkable PartiallySeparableNLPModel.struct_hessian!($SPS_ros, $x, $H)


  SPS_Structured_Hessian_en_x = PartiallySeparableNLPModel.struct_hessian!(SPS_ros, x, H)
  SUITE["ROS $n variable"]["Hv"]["SPS"] = @benchmarkable PartiallySeparableNLPModel.product_matrix_sps($SPS_ros, $H, $y)

  MOI_Hessian_product_y = Vector{ typeof(y[1]) }(undef,n)
  SUITE["ROS $n variable"]["Hv"]["JuMP"] = @benchmarkable MathOptInterface.eval_hessian_lagrangian_product($evaluator, $MOI_Hessian_product_y, $x, $y, 1.0, zeros(0))

end
