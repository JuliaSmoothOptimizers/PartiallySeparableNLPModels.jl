using JuMP, MathOptInterface



m = Model()
n = 1000
@variable(m, x[1:n])
@NLobjective(m, Min, sum( x[j] * x[j+1] for j in 1:n-1 ) + (sin(x[1]))^2 + x[n-1]^3  + 5 )
evaluator = JuMP.NLPEvaluator(m)
MathOptInterface.initialize(evaluator, [:ExprGraph])
Expr_j = MathOptInterface.objective_expr(evaluator)
expr_tree_j = PartiallySeparableNLPModel.fonction_test2(Expr_j)


x = ones(n)
res_Expr = PartiallySeparableNLPModel.fonction_test(Expr_j, x)
res_expr_tree = PartiallySeparableNLPModel.fonction_test(expr_tree_j, x)

@test res_expr_tree == res_Expr
