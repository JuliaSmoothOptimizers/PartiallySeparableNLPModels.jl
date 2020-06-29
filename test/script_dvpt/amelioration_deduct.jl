type = Float64
work_expr_tree = copy(complete_expr_tree)
CalculusTreeTools.cast_type_of_constant(work_expr_tree, type)
bench_cast = @benchmark CalculusTreeTools.cast_type_of_constant(work_expr_tree, type)
elmt_fun = CalculusTreeTools.delete_imbricated_plus(work_expr_tree)
bench_elmt_fun = @benchmark CalculusTreeTools.delete_imbricated_plus(work_expr_tree)
CalculusTreeTools.set_bounds!.(elmt_fun)
bench_bounds = @benchmark CalculusTreeTools.set_bounds!.(elmt_fun)
CalculusTreeTools.set_convexity!.(elmt_fun)
bench_cvx = @benchmark CalculusTreeTools.set_convexity!.(elmt_fun)

convexity_wrapper = map( (x -> CalculusTreeTools.convexity_wrapper(CalculusTreeTools.get_convexity_status(x)) ), elmt_fun)

m_i = length(elmt_fun)

type_i = Vector{CalculusTreeTools.type_calculus_tree}(undef, m_i)
for i in 1:m_i
    type_i[i] = CalculusTreeTools.get_type_tree(elmt_fun[i])
end

elmt_var_i =  Vector{ Vector{Int}}(undef, m_i)
length_vec = Threads.Atomic{Int}(0)
for i in 1:m_i
    elmt_var_i[i] = CalculusTreeTools.get_elemental_variable(elmt_fun[i])
    atomic_add!(length_vec, length(elmt_var_i[i]))
end
sort!.(elmt_var_i) #ligne importante, met dans l'ordre les variables élémentaires. Utile pour les U_i et le N_to_Ni

for i in 1:m_i
    CalculusTreeTools.element_fun_from_N_to_Ni!(elmt_fun[i],elmt_var_i[i])
end

(different_calculus_tree, different_calculus_tree_index) = PartiallySeparableNLPModel.get_different_CalculusTree(elmt_fun)
bench_diff_elmt_tree = @benchmark (PartiallySeparableNLPModel.get_different_CalculusTree(elmt_fun))

Sps = Vector{element_function}(undef,m_i)
for i in 1:m_i
    Sps[i] = element_function(different_calculus_tree_index[i], type_i[i], elmt_var_i[i], convexity_wrapper[i], i)
end

index_element_tree = PartiallySeparableNLPModel.get_related_function(Sps, different_calculus_tree)
bench_index = @benchmark (PartiallySeparableNLPModel.get_related_function(Sps, different_calculus_tree))

related_vars = PartiallySeparableNLPModel.get_related_var(Sps, index_element_tree)
bench_related_vars = @benchmark (PartiallySeparableNLPModel.get_related_var(Sps, index_element_tree))

compiled_gradients = map(x -> PartiallySeparableNLPModel.compiled_grad_of_elmt_fun(x), different_calculus_tree)

x = Vector{type}(undef,n)
x_views = PartiallySeparableNLPModel.construct_views(x, related_vars)
bench_views = @benchmark (PartiallySeparableNLPModel.construct_views(x, related_vars))

bench_deduct = @benchmark PartiallySeparableNLPModel.deduct_partially_separable_structure(complete_expr_tree, n)
