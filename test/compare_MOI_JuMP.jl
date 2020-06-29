using JuMP, MathOptInterface, LinearAlgebra, SparseArrays

using CalculusTreeTools


println("\n\nCompare_With_MOI_JUMP\n\n")


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

function create_Rosenbrock_JuMP_Model2(n :: Int)
    m = Model()
    @variable(m, x[1:n])
    @NLobjective(m, Min, sum( 100 * (x[j-1]^2 - x[j])^2 + (x[j-1] )^2  for j in 2:n)) #rosenbrock function
    evaluator = JuMP.NLPEvaluator(m)
    MathOptInterface.initialize(evaluator, [:ExprGraph, :Hess])
    obj = MathOptInterface.objective_expr(evaluator)
    vec_var = JuMP.all_variables(m)
    vec_value = create_initial_point_Rosenbrock(n)
    JuMP.set_start_value.(vec_var, vec_value)
    return (m, evaluator,obj)
end


# n = 1000
# (m, evaluator,obj) = create_Rosenbrock_JuMP_Model(n)
n = 100
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




σ = 1e-5

ones_ = ones(n)
println("fin des initialisations")



println(" EVALUATION DES FONCTIONS ")
@testset "evaluation des fonctions par divers moyens" begin

    # obj_SPS_x = PartiallySeparableNLPModel.evaluate_SPS( SPS1, x)
    obj_SPS2_x = PartiallySeparableNLPModel.evaluate_SPS( SPS2, x)
    obj_SPS3_x = PartiallySeparableNLPModel.evaluate_SPS( SPS3, x)
    obj_MOI_x = MathOptInterface.eval_objective( evaluator, x)

    # @test abs(obj_SPS_x - obj_MOI_x) < σ
    @test abs(obj_SPS2_x - obj_MOI_x) < σ
    @test abs(obj_SPS3_x - obj_MOI_x) < σ


    # obj_SPS_y = PartiallySeparableNLPModel.evaluate_SPS(SPS1, y)
    obj_SPS2_y = PartiallySeparableNLPModel.evaluate_SPS(SPS2, y)
    obj_SPS3_y = PartiallySeparableNLPModel.evaluate_SPS(SPS2, y)
    obj_MOI_y = MathOptInterface.eval_objective(evaluator, y)

    # @test abs(obj_SPS_y - obj_MOI_y) < σ
    @test abs(obj_SPS2_y - obj_MOI_y) < σ
    @test abs(obj_SPS3_y - obj_MOI_y) < σ


    # obj_SPS_rdm = PartiallySeparableNLPModel.evaluate_SPS(SPS1, rdm)
    obj_SPS2_rdm = PartiallySeparableNLPModel.evaluate_SPS(SPS2, rdm)
    obj_SPS3_rdm = PartiallySeparableNLPModel.evaluate_SPS(SPS3, rdm)
    obj_MOI_rdm = MathOptInterface.eval_objective(evaluator, rdm)

    # @test abs(obj_SPS_rdm - obj_MOI_rdm) < σ
    @test abs(obj_SPS2_rdm - obj_MOI_rdm) < σ
    @test abs(obj_SPS3_rdm - obj_MOI_rdm) < σ


end



println(" EVALUATION DES GRADIENTS ")
@testset " evaluation du gradient par divers moyer" begin
    #fonction pour allouer un grad_vector facilement à partir d'une structure partiellement séparable
    f = (y :: PartiallySeparableNLPModel.element_function -> PartiallySeparableNLPModel.element_gradient{typeof(x[1])}(Vector{typeof(x[1])}(zeros(typeof(x[1]), length(y.used_variable)) )) )
    #fonction pour comparer les norms des gradient elements
    nrm_grad_elem = (g_elem :: PartiallySeparableNLPModel.element_gradient{} -> norm(g_elem.g_i) )

    # Définition des structure de résultats nécessaires
    MOI_gradient = Vector{ typeof(x[1]) }(undef,n)
    p_grad = PartiallySeparableNLPModel.grad_vector{typeof(x[1])}( f.(SPS1.structure) )
    p_grad2 = PartiallySeparableNLPModel.grad_vector{typeof(x[1])}( f.(SPS2.structure) )
    p_grad3 = PartiallySeparableNLPModel.grad_vector{typeof(x[1])}( f.(SPS3.structure) )
    p_grad_build = Vector{Float64}(zeros(Float64,n))
    p_grad_build2 = Vector{Float64}(zeros(Float64,n))
    p_grad_build3 = Vector{Float64}(zeros(Float64,n))

    MathOptInterface.eval_objective_gradient(evaluator, MOI_gradient, x)
    PartiallySeparableNLPModel.evaluate_SPS_gradient!(SPS1, x, p_grad)
    PartiallySeparableNLPModel.evaluate_SPS_gradient!(SPS2, x, p_grad2)
    PartiallySeparableNLPModel.evaluate_SPS_gradient!(SPS3, x, p_grad3)

    grad = PartiallySeparableNLPModel.build_gradient(SPS1, p_grad)
    grad2 = PartiallySeparableNLPModel.build_gradient(SPS2, p_grad2)
    grad3 = PartiallySeparableNLPModel.build_gradient(SPS3, p_grad3)
    PartiallySeparableNLPModel.build_gradient!(SPS1, p_grad, p_grad_build)
    PartiallySeparableNLPModel.build_gradient!(SPS2, p_grad2, p_grad_build2)
    PartiallySeparableNLPModel.build_gradient!(SPS3, p_grad3, p_grad_build3)

    @test norm(MOI_gradient - grad) < σ
    @test norm(MOI_gradient - grad2) < σ
    @test norm(MOI_gradient - grad3) < σ
    @test norm(MOI_gradient - p_grad_build) < σ
    @test norm(MOI_gradient - p_grad_build2) < σ
    @test norm(MOI_gradient - p_grad_build3) < σ

    @test sum(nrm_grad_elem.(p_grad.arr)) - sum(nrm_grad_elem.(p_grad2.arr)) < σ
    @test sum(nrm_grad_elem.(p_grad.arr)) - sum(nrm_grad_elem.(p_grad3.arr)) < σ


    MathOptInterface.eval_objective_gradient(evaluator, MOI_gradient, y)
    PartiallySeparableNLPModel.evaluate_SPS_gradient!(SPS1, y, p_grad)
    PartiallySeparableNLPModel.evaluate_SPS_gradient!(SPS2, y, p_grad2)
    PartiallySeparableNLPModel.evaluate_SPS_gradient!(SPS3, y, p_grad3)

    grad = PartiallySeparableNLPModel.build_gradient(SPS1, p_grad)
    grad2 = PartiallySeparableNLPModel.build_gradient(SPS2, p_grad2)
    grad3 = PartiallySeparableNLPModel.build_gradient(SPS3, p_grad3)
    PartiallySeparableNLPModel.build_gradient!(SPS1, p_grad, p_grad_build)
    PartiallySeparableNLPModel.build_gradient!(SPS2, p_grad2, p_grad_build2)
    PartiallySeparableNLPModel.build_gradient!(SPS3, p_grad3, p_grad_build3)


    @test norm(MOI_gradient - grad) < σ
    @test norm(MOI_gradient - grad2) < σ
    @test norm(MOI_gradient - grad3) < σ
    @test norm(MOI_gradient - p_grad_build) < σ
    @test norm(MOI_gradient - p_grad_build2) < σ
    @test norm(MOI_gradient - p_grad_build3) < σ

    @test sum(nrm_grad_elem.(p_grad.arr)) - sum(nrm_grad_elem.(p_grad2.arr)) < σ
    @test sum(nrm_grad_elem.(p_grad.arr)) - sum(nrm_grad_elem.(p_grad3.arr)) < σ
end



println(" EVALUATION DES HESSIANS ")
@testset "evaluation du Hessian par divers moyers" begin

    MOI_pattern = MathOptInterface.hessian_lagrangian_structure(evaluator)
    column = [x[1] for x in MOI_pattern]
    row = [x[2]  for x in MOI_pattern]

    f = ( elm_fun :: PartiallySeparableNLPModel.element_function -> PartiallySeparableNLPModel.element_hessian{Float64}( Array{Float64,2}(undef, length(elm_fun.used_variable), length(elm_fun.used_variable) )) )
    t = f.(SPS1.structure) :: Vector{PartiallySeparableNLPModel.element_hessian{Float64}}
    H = PartiallySeparableNLPModel.Hess_matrix{Float64}(t)
    H2 = PartiallySeparableNLPModel.Hess_matrix{Float64}(t)
    H3 = PartiallySeparableNLPModel.Hess_matrix{Float64}(t)

    MOI_value_Hessian = Vector{ typeof(x[1]) }(undef,length(MOI_pattern))
    MathOptInterface.eval_hessian_lagrangian(evaluator, MOI_value_Hessian, x, 1.0, zeros(0))
    values = [x for x in MOI_value_Hessian]

    MOI_half_hessian_en_x = sparse(row,column,values,n,n)
    MOI_hessian_en_x = Symmetric(MOI_half_hessian_en_x)

    PartiallySeparableNLPModel.struct_hessian!(SPS1, x, H)
    sp_H = PartiallySeparableNLPModel.construct_Sparse_Hessian(SPS1, H)
    PartiallySeparableNLPModel.struct_hessian!(SPS2, x, H2)
    sp_H2 = PartiallySeparableNLPModel.construct_Sparse_Hessian(SPS2, H2)
    PartiallySeparableNLPModel.struct_hessian!(SPS3, x, H3)
    sp_H3 = PartiallySeparableNLPModel.construct_Sparse_Hessian(SPS3, H3)

    @test norm(MOI_hessian_en_x - sp_H, 2) < σ
    @test norm(MOI_hessian_en_x - sp_H2, 2) < σ
    @test norm(MOI_hessian_en_x - sp_H3, 2) < σ



    # # on récupère le Hessian structuré du format SPS.
    # #Ensuite on calcul le produit entre le structure de donnée SPS_Structured_Hessian_en_x et y
    @testset "test du produit" begin
        x_H_y = PartiallySeparableNLPModel.product_matrix_sps(SPS1, H, y)
        x_H_y2 = PartiallySeparableNLPModel.product_matrix_sps(SPS1, H2, y)
        x_H_y3 = PartiallySeparableNLPModel.product_matrix_sps(SPS1, H3, y)


        v_tmp = Vector{ Float64 }(undef, length(MOI_pattern))
        x_MOI_Hessian_y = Vector{ typeof(y[1]) }(undef,n)
        MathOptInterface.eval_hessian_lagrangian_product(evaluator, x_MOI_Hessian_y, x, y, 1.0, zeros(0))
        #
        @test norm(x_MOI_Hessian_y - x_H_y, 2) < σ
        @test norm(x_MOI_Hessian_y - x_H_y2, 2) < σ
        @test norm(x_MOI_Hessian_y - x_H_y3, 2) < σ
        @test norm(x_H_y - MOI_hessian_en_x*y, 2) < σ
        @test norm(x_H_y2 - MOI_hessian_en_x*y, 2) < σ
        @test norm(x_H_y3 - MOI_hessian_en_x*y, 2) < σ

    end

end



    @testset "test du status de convexité des fonctions éléments" begin

        f = ( x -> CalculusTreeTools.get_convexity_wrapper( PartiallySeparableNLPModel.get_convexity_status(x)) )
        res_cvx1 = f.(PartiallySeparableNLPModel.get_structure(SPS1))
        res_cvx2 = f.(PartiallySeparableNLPModel.get_structure(SPS2))
        res_cvx3 = f.(PartiallySeparableNLPModel.get_structure(SPS3))
        res = []
        for i in 1:length(PartiallySeparableNLPModel.get_structure(SPS3))
            if i % 2 == 0
                push!(res,CalculusTreeTools.convex_type())
            else
                push!(res,CalculusTreeTools.unknown_type())
            end
        end
        @test res == res_cvx3
    end


    # @testset " test convexité " begin
    #     (m, evaluator,obj) = create_Rosenbrock_JuMP_Model2(n)
    #
    #     x = ones(n)
    #     y = zeros(n)
    #     rdm = rand(n)
    #
    #     expr_tree_obj = CalculusTreeTools.transform_to_expr_tree(obj)
    #     comp_ext = CalculusTreeTools.create_complete_tree(expr_tree_obj)
    #
    #     # détection de la structure partiellement séparable
    #     SPS1 = PartiallySeparableNLPModel.deduct_partially_separable_structure(obj, n)
    #
    #     obj2 = CalculusTreeTools.transform_to_expr_tree(obj)
    #     SPS2 = PartiallySeparableNLPModel.deduct_partially_separable_structure(obj2, n)
    #
    #     SPS3 = PartiallySeparableNLPModel.deduct_partially_separable_structure(comp_ext, n)
    #
    #
    #     f = ( x -> CalculusTreeTools.get_convexity_wrapper( PartiallySeparableNLPModel.get_convexity_status(x)) )
    #     res_cvx1 = f.(PartiallySeparableNLPModel.get_structure(SPS1))
    #     res_cvx2 = f.(PartiallySeparableNLPModel.get_structure(SPS2))
    #     res_cvx3 = f.(PartiallySeparableNLPModel.get_structure(SPS3))
    # end
