using JuMP, MathOptInterface
using PartitionedStructures

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
	x0 = create_initial_point_Rosenbrock(n)
	JuMP.set_start_value.(vec_var, x0)
	return (m, evaluator,obj, x0)
end

@testset "PQN structure" begin
	n = 40
	x = (x -> 2*x).(ones(n))
	y = rand(n)
	(m, evaluator, obj, x0) = create_Rosenbrock_JuMP_Model(n)

	ps_data = build_PartitionedData_TR_PQN(obj, n;x0=x)

	objx = evaluate_obj_part_data(ps_data, x)
	obj_MOI_x = MathOptInterface.eval_objective(evaluator, x)
	@test objx ≈ obj_MOI_x 

	objy = evaluate_obj_part_data(ps_data, y)
	obj_MOI_y = MathOptInterface.eval_objective(evaluator, y)
	@test objy ≈ obj_MOI_y

	gx_MOI = similar(x)
	MathOptInterface.eval_objective_gradient(evaluator, gx_MOI, x)
	gx = evaluate_grad_part_data(ps_data,x)
	@test gx ≈ gx_MOI

	gy_MOI = similar(y)
	MathOptInterface.eval_objective_gradient(evaluator, gy_MOI, y)
	gy = evaluate_grad_part_data(ps_data,y)
	@test gy ≈ gy_MOI

	Bk = Matrix(ps_data.pB)

	x = (x->2*x).(ones(n))
	s = (x->0.1*x).(ones(n))	
	update_nlp!(ps_data,x,s)
	Bk1 = Matrix(ps_data.pB)
	epv_y = ps_data.py
	PartitionedStructures.build_v!(epv_y)
	_y = PartitionedStructures.get_v(epv_y)
	@test isapprox(norm(Bk1*s - _y), 0, atol=1e-10)

	res = product_part_data_x(ps_data, x)
	@test res ≈ Bk1 * x 
end 


@testset "PQNNLPModels" begin
	n = 40
	
	(m, evaluator, obj, x0) = create_Rosenbrock_JuMP_Model(n)
	x = (x -> 2*x).(ones(n))
	s = rand(n)

	ps_data_plbfgs = build_PartitionedData_TR_PQN(obj, n;x0=x, name=:plbfgs)
	ps_data_plsr1 = build_PartitionedData_TR_PQN(obj, n;x0=x, name=:plsr1)
	ps_data_plse = build_PartitionedData_TR_PQN(obj, n;x0=x, name=:plse)
	ps_data_pbfgs = build_PartitionedData_TR_PQN(obj, n;x0=x, name=:pbfgs)
	ps_data_psr1 = build_PartitionedData_TR_PQN(obj, n;x0=x, name=:psr1)
	ps_data_pse = build_PartitionedData_TR_PQN(obj, n;x0=x, name=:pse)

	update_nlp!(ps_data_plbfgs, x, s)
	update_nlp!(ps_data_plsr1, x, s)
	update_nlp!(ps_data_plse, x, s)
	update_nlp!(ps_data_pbfgs, x, s)
	update_nlp!(ps_data_psr1, x, s)
	update_nlp!(ps_data_pse, x, s)

	@test ps_data_plsr1.py == ps_data_plbfgs.py
	@test ps_data_plsr1.py == ps_data_plse.py
	@test ps_data_plsr1.py == ps_data_pbfgs.py
	@test ps_data_plsr1.py == ps_data_psr1.py
	@test ps_data_plsr1.py == ps_data_pse.py

	epv_y = ps_data_plsr1.py
	PartitionedStructures.build_v!(epv_y)
	y = PartitionedStructures.get_v(epv_y)

	partitioned_matrix(nlp) = Matrix(nlp.pB)

	# in the case of the Rosenbrock equation, for the given x,s and induces y, every partitioned update ensure the secant equation.
	@test isapprox(norm(partitioned_matrix(ps_data_plbfgs)*s - y), 0, atol=1e-10)
	@test isapprox(norm(partitioned_matrix(ps_data_plsr1)*s - y), 0, atol=1e-10)
	@test isapprox(norm(partitioned_matrix(ps_data_plse)*s - y), 0, atol=1e-10)
	@test isapprox(norm(partitioned_matrix(ps_data_pbfgs)*s - y), 0, atol=1e-10)
	@test isapprox(norm(partitioned_matrix(ps_data_psr1)*s - y), 0, atol=1e-10)
	@test isapprox(norm(partitioned_matrix(ps_data_pse)*s - y), 0, atol=1e-10)

end