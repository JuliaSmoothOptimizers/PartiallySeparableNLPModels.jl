using JuMP, MathOptInterface

function create_initial_point_Rosenbrock(n)
  point_initial = Vector{Float64}(undef, n)
  for i = 1:n
    if mod(i, 2) == 1
      point_initial[i] = -1.2
    elseif mod(i, 2) == 0
      point_initial[i] = 1.0
    else
      error("bizarre")
    end
  end
  return point_initial
end

function create_Rosenbrock_JuMP_Model(n::Int)
  m = Model()
  @variable(m, x[1:n])
  @NLobjective(m, Min, sum(100 * (x[j - 1]^2 - x[j])^2 + (x[j - 1] - 1)^2 for j = 2:n)) #rosenbrock function
  evaluator = JuMP.NLPEvaluator(m)
  MathOptInterface.initialize(evaluator, [:ExprGraph, :Hess])
  obj = MathOptInterface.objective_expr(evaluator)
  vec_var = JuMP.all_variables(m)
  x0 = create_initial_point_Rosenbrock(n)
  JuMP.set_start_value.(vec_var, x0)
  return (m, evaluator, obj, x0)
end

@testset "PBFGS structure" begin
  n = 40
  x = ones(n)
  y = rand(n)
  (m, evaluator, obj, x0) = create_Rosenbrock_JuMP_Model(n)

  ps_data = build_PartitionedData_TR_PBFGS(obj, n; x0 = x0)

  objx = evaluate_obj_part_data(ps_data, x)
  obj_MOI_x = MathOptInterface.eval_objective(evaluator, x)
  @test objx ≈ obj_MOI_x

  objy = evaluate_obj_part_data(ps_data, y)
  obj_MOI_y = MathOptInterface.eval_objective(evaluator, y)
  @test objy ≈ obj_MOI_y

  gx_MOI = similar(x)
  MathOptInterface.eval_objective_gradient(evaluator, gx_MOI, x)
  gx = evaluate_grad_part_data(ps_data, x)
  @test gx ≈ gx_MOI

  gy_MOI = similar(y)
  MathOptInterface.eval_objective_gradient(evaluator, gy_MOI, y)
  gy = evaluate_grad_part_data(ps_data, y)
  @test gy ≈ gy_MOI

  Bk = Matrix(ps_data.pB)
  @test minimum(eigen(Bk).values) > 0

  x = (x -> 2 * x).(ones(n))
  s = (x -> 0.1 * x).(ones(n))
  update_PBFGS(ps_data, x, s)
  Bk1 = Matrix(ps_data.pB)
  @test minimum(eigen(Bk1).values) > 0

  res = product_part_data_x(ps_data, x)
  @test res ≈ Bk1 * x
end
