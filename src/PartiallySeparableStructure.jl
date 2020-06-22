
using CalculusTreeTools

using ForwardDiff, SparseArrays, LinearAlgebra, ReverseDiff
using Base.Threads
import Base.-

mutable struct element_function
    index_fun :: Int
    type :: CalculusTreeTools.type_calculus_tree
    used_variable :: Vector{Int}
    U :: SparseMatrixCSC{Int,Int}
    convexity_status :: CalculusTreeTools.convexity_wrapper
    index :: Int
end

get_fun_from_elmt_fun( elmt_fun :: element_function, diff_elmt_fun :: Vector{T}) where T = diff_elmt_fun[elmt_fun.index_fun]
get_type( elmt_fun :: element_function) = elmt_fun.type
get_used_variable( elmt_fun :: element_function) = elmt_fun.used_variable
get_U( elmt_fun :: element_function) = elmt_fun.U
get_convexity_status( elmt_fun :: element_function) = elmt_fun.convexity_status
get_index( elmt_fun :: element_function) = elmt_fun.index


mutable struct SPS{T}
    structure :: Vector{element_function}
    different_element_tree :: Vector{T}
    vec_length :: Int
    n_var :: Int
    compiled_gradients :: Vector{ReverseDiff.CompiledTape}
end

mutable struct element_hessian{T <: Number}
    elmt_hess :: Array{T,2}
end

mutable struct Hess_matrix{T <: Number}
    arr :: Vector{element_hessian{T}}
end

mutable struct element_gradient{ T <: Number}
    g_i :: Vector{T}
end

mutable struct grad_vector{ T <: Number}
    arr :: Vector{element_gradient{T}}
end


get_structure(sps :: SPS{T}) where T = sps.structure
get_convexity_status(elmt_fun :: element_function)  = elmt_fun.convexity_status


function get_different_CalculusTree(all_elemt_fun :: Vector{T}) where T
    # structure = sps.structure
    # elmt_fun = ( x -> PartiallySeparableNLPModel.get_fun(x)).(structure)
    work_elmt_fun = copy(all_elemt_fun)
    different_calculus_tree = Vector{T}(undef,0)

    while isempty(work_elmt_fun) == false
        current_tree = work_elmt_fun[1]
        push!(different_calculus_tree, current_tree)
        work_elmt_fun = filter( (x -> x != current_tree), work_elmt_fun)
    end

    different_calculus_tree_index = Vector{Int}(undef, length(all_elemt_fun))
    for i in 1:length(all_elemt_fun)
        for j in 1:length(different_calculus_tree)
            if all_elemt_fun[i] == different_calculus_tree[j]
                different_calculus_tree_index[i] = j
                break
            end
        end
    end
    # @show work_elmt_fun, different_calculus_tree, length(different_calculus_tree)
    # @show different_calculus_tree_index
    return (different_calculus_tree, different_calculus_tree_index)
end

"""
    deduct_partially_separable_structure(expr_tree, n)

Find the partially separable structure of a function f stored as an expression tree expr_tree.
To define properly the size of sparse matrix we need the size of the problem : n.
At the end, we get the partially separable structure of f, f(x) = ∑fᵢ(xᵢ)
"""
# deduct_partially_separable_structure(a :: Any, n :: Int) = _deduct_partially_separable_structure(a, trait_expr_tree.is_expr_tree(a), n)
# _deduct_partially_separable_structure(a, :: trait_expr_tree.type_not_expr_tree, n :: Int) = error("l'entrée de la fonction n'est pas un arbre d'expression")
# _deduct_partially_separable_structure(a, :: trait_expr_tree.type_expr_tree, n :: Int) = _deduct_partially_separable_structure(a, n )
deduct_partially_separable_structure(a :: Any, n :: Int) = _deduct_partially_separable_structure(a, n)
function _deduct_partially_separable_structure(expr_tree :: T , n :: Int) where T
    work_expr_tree = copy(expr_tree)

    # elmt_fun = CalculusTreeTools.delete_imbricated_plus(work_expr_tree) :: Vector{T}
    elmt_fun = CalculusTreeTools.delete_imbricated_plus(work_expr_tree)
    m_i = length(elmt_fun)

    # type_i = Vector{implementation_type_expr.t_type_expr_basic}(undef, m_i)
    type_i = Vector{CalculusTreeTools.type_calculus_tree}(undef, m_i)
    for i in 1:m_i
        type_i[i] = CalculusTreeTools.get_type_tree(elmt_fun[i])
    end

    # elmt_var_i = CalculusTreeTools.get_elemental_variable.(elmt_fun) :: Vector{ Vector{Int}}
    elmt_var_i =  Vector{ Vector{Int}}(undef,m_i)
    length_vec = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:m_i
        elmt_var_i[i] = CalculusTreeTools.get_elemental_variable(elmt_fun[i])
        atomic_add!(length_vec, length(elmt_var_i[i]))
    end
    sort!.(elmt_var_i) #ligne importante, met dans l'ordre les variables élémentaires. Utile pour les U_i et le N_to_Ni

    # U_i = CalculusTreeTools.get_Ui.(elmt_var_i, n) :: Vector{SparseMatrixCSC{Int,Int}}
    U_i = Vector{SparseMatrixCSC{Int,Int}}(undef,m_i)
    Threads.@threads for i in 1:m_i
        U_i[i] = CalculusTreeTools.get_Ui(elmt_var_i[i], n)
    end

    # CalculusTreeTools.element_fun_from_N_to_Ni!.(elmt_fun,elmt_var_i)
    # Threads.@threads for i in 1:m_i
    for i in 1:m_i
        CalculusTreeTools.element_fun_from_N_to_Ni!(elmt_fun[i],elmt_var_i[i])
    end

    Sps = Vector{element_function}(undef,m_i)
    convexity_wrapper = CalculusTreeTools.convexity_wrapper(CalculusTreeTools.unknown_type())


    (different_calculus_tree, different_calculus_tree_index) = get_different_CalculusTree(elmt_fun)

    # Threads.@threads for i in 1:m_i
    for i in 1:m_i
        Sps[i] = element_function(different_calculus_tree_index[i], type_i[i], elmt_var_i[i], U_i[i], convexity_wrapper, i)
    end


    compiled_gradients = map(x -> compiled_grad_of_elmt_fun(x, different_calculus_tree), Sps)

    return SPS{T}(Sps, different_calculus_tree, length_vec[], n, compiled_gradients)
end




function _deduct_partially_separable_structure(expr_tree :: CalculusTreeTools.complete_expr_tree, n :: Int)
    work_expr_tree = copy(expr_tree)

    # elmt_fun = CalculusTreeTools.delete_imbricated_plus(work_expr_tree) :: Vector{T}
    elmt_fun = CalculusTreeTools.delete_imbricated_plus(work_expr_tree)
    CalculusTreeTools.set_bounds!.(elmt_fun)
    CalculusTreeTools.set_convexity!.(elmt_fun)
    convexity_wrapper = map( (x -> CalculusTreeTools.convexity_wrapper(CalculusTreeTools.get_convexity_status(x)) ), elmt_fun)
    m_i = length(elmt_fun)


    # type_i = Vector{implementation_type_expr.t_type_expr_basic}(undef, m_i)
    type_i = Vector{CalculusTreeTools.type_calculus_tree}(undef, m_i)
    for i in 1:m_i
        type_i[i] = CalculusTreeTools.get_type_tree(elmt_fun[i])
    end

    # elmt_var_i = CalculusTreeTools.get_elemental_variable.(elmt_fun) :: Vector{ Vector{Int}}
    elmt_var_i =  Vector{ Vector{Int}}(undef,m_i)
    length_vec = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:m_i
        elmt_var_i[i] = CalculusTreeTools.get_elemental_variable(elmt_fun[i])
        atomic_add!(length_vec, length(elmt_var_i[i]))
    end
    sort!.(elmt_var_i) #ligne importante, met dans l'ordre les variables élémentaires. Utile pour les U_i et le N_to_Ni

    # U_i = CalculusTreeTools.get_Ui.(elmt_var_i, n) :: Vector{SparseMatrixCSC{Int,Int}}
    U_i = Vector{SparseMatrixCSC{Int,Int}}(undef,m_i)
    Threads.@threads for i in 1:m_i
        U_i[i] = CalculusTreeTools.get_Ui(elmt_var_i[i], n)
    end

    # CalculusTreeTools.element_fun_from_N_to_Ni!.(elmt_fun,elmt_var_i)
    # Threads.@threads for i in 1:m_i
    for i in 1:m_i
        CalculusTreeTools.element_fun_from_N_to_Ni!(elmt_fun[i],elmt_var_i[i])
    end

    (different_calculus_tree, different_calculus_tree_index) = get_different_CalculusTree(elmt_fun)


    Sps = Vector{element_function}(undef,m_i)
    # Threads.@threads for i in 1:m_i
    for i in 1:m_i
        Sps[i] = element_function(different_calculus_tree_index[i], type_i[i], elmt_var_i[i], U_i[i], convexity_wrapper[i], i)
    end


    compiled_gradients = map(x -> compiled_grad_of_elmt_fun(x, different_calculus_tree), Sps)

    return SPS{CalculusTreeTools.complete_expr_tree}(Sps, different_calculus_tree, length_vec[], n, compiled_gradients)
end


"""
    compiled_grad_of_elmt_fun(elmt_fun)
Return  the GradientTape compiled to speed up the ReverseDiff computation of the elmt_fun gradient in the future
"""
function compiled_grad_of_elmt_fun(elmt_fun :: element_function, different_calculus_tree :: Vector{T}) where T
    fun = get_fun_from_elmt_fun(elmt_fun,different_calculus_tree)
    f = CalculusTreeTools.evaluate_expr_tree(fun)
    n = length(get_used_variable(elmt_fun))
    f_tape = ReverseDiff.GradientTape(f, rand(n))
    compiled_f_tape = ReverseDiff.compile(f_tape)
    return compiled_f_tape
end

#
# """
#     evaluate_SPS(sps,x)
# evalutate the partially separable function f = ∑fᵢ, stored in the sps structure at the point x.
# f(x) = ∑fᵢ(xᵢ), so we compute independently each fᵢ(xᵢ) and we return the sum.
# """
# function evaluate_SPS(sps :: SPS{T}, x :: AbstractVector{Y} ) where T where Y <: Number
#     # on utilise un mapreduce de manière à ne pas allouer un tableau temporaire, on utilise l'opérateur + pour le reduce car cela correspond
#     # à la définition des fonctions partiellement séparable.
#     @inbounds @fastmath mapreduce(elmt_fun :: element_function{T} -> CalculusTreeTools.evaluate_expr_tree(elmt_fun.fun, view(x, elmt_fun.used_variable)) :: Y, + , sps.structure :: Vector{element_function{T}})
#     #les solutions à base de boucle for sont plus lente même avec @Thread.thread
# end
#
#
# """
#     evaluate_gradient(sps,x)
# evalutate the gradient of the partially separable function f = ∑ fι, stored in the sps structure
# at the point x, return a vector of size n (the number of variable) which is the gradient.
# Première version de la fonction inutile car inefficace.
# """
# function evaluate_gradient(sps :: SPS{T}, x :: Vector{Y} ) where T where Y <: Number
#     l_elmt_fun = length(sps.structure)
#     gradient_prl = Vector{Threads.Atomic{Y}}((x-> Threads.Atomic{Y}(0)).([1:sps.n_var;]) )
#      for i in 1:l_elmt_fun
#         if isempty(sps.structure[i].U) == false
#             (row, column, value) = findnz(sps.structure[i].U)
#             temp = ForwardDiff.gradient(CalculusTreeTools.evaluate_expr_tree(sps.structure[i].fun), Array(view(x, sps.structure[i].used_variable))  )
#             atomic_add!.(gradient_prl[column], temp)
#         end
#     end
#     gradient = (x -> x[]).(gradient_prl) :: Vector{Y}
#     return gradient
# end
#
#
# """
#     evaluate_SPS_gradient!(sps,x,g)
# Compute the gradient of the partially separable structure sps, and store the result in the grad_vector structure g.
# Using ReversDiff package. Not obvious good behaviour with Threads.@threads, sometime yes sometime no.
# Noted that we use the previously compiled GradientTape in element_gradient! that use ReverseDiff.
# """
# function evaluate_SPS_gradient!(sps :: SPS{T}, x :: AbstractVector{Y}, g :: grad_vector{Y} ) where T where Y <: Number
#     l_elmt_fun = length(sps.structure)
#     for i in 1:l_elmt_fun
#         if isempty(sps.structure[i].used_variable) == false  #fonction element ayant au moins une variable
#             element_gradient!(sps.compiled_gradients[i], view(x, sps.structure[i].used_variable), g.arr[i] )
#         end
#     end
# end
#
# """
#     element_gradient!(compil_tape, x, g)
# Compute the element grandient from the compil_tape compiled before according to the vector x, and store the result in the vector g
# Use of ReverseDiff
# """
# function element_gradient!(compiled_tape :: ReverseDiff.CompiledTape, x :: AbstractVector{T}, g :: element_gradient{T} ) where T <: Number where Y
#     ReverseDiff.gradient!(g.g_i, compiled_tape, x)
# end
#
# """
#     evaluate_SPS_gradient2!(sps,x,g)
# Compute the gradient of the partially separable structure sps, and store the result in the grad_vector structure g.
# Using ForwardDiff package. Bad behaviour with Threads.@threads.
# This was the previous version using ForwardDiff. The actual version using ReverseDiff is more efficient.
# """
# function evaluate_SPS_gradient2!(sps :: SPS{T}, x :: AbstractVector{Y}, g :: grad_vector{Y} ) where T where Y <: Number
#     l_elmt_fun = length(sps.structure)
#     for i in 1:l_elmt_fun
#         if isempty(sps.structure[i].used_variable) == false  #fonction element ayant au moins une variable
#             element_gradient2!(sps.structure[i].fun, view(x, sps.structure[i].used_variable), g.arr[i] )
#         end
#     end
# end
#
# """
#     element_gradient2!(expr_tree, x, g)
# Compute the element grandient of the function represents by expr_tree according to the vector x, and store the result in the vector g.
# This was the previous version using ForwardDiff. The actual version using ReverseDiff is more efficient.
# """
# function element_gradient2!(expr_tree :: Y, x :: AbstractVector{T}, g :: element_gradient{T} ) where T <: Number where Y
#     ForwardDiff.gradient!(g.g_i, CalculusTreeTools.evaluate_expr_tree(expr_tree), x  )
# end
#
#
# """
#     build_gradient(sps, g)
# Constructs a vector of size n from the list of element gradient of the sps structure which has numerous element gradient of size nᵢ.
# The purpose of the function is to gather these element gradient into a real gradient of size n.
# The function grad_ni_to_n! will add element gradient of size nᵢ at the right inside the gradient of size n.
# """
# function build_gradient(sps :: SPS{T}, g :: grad_vector{Y}) where T where Y <: Number
#     grad = Vector{Y}(undef, sps.n_var)
#     build_gradient!(sps, g, grad)
#     return grad
# end
#
# function build_gradient!(sps :: SPS{T}, g :: grad_vector{Y}, g_res :: AbstractVector{Y}) where T where Y <: Number
#     g_res[:] = zeros(Y, sps.n_var)
#     l_elmt_fun = length(sps.structure)
#     for i in 1:l_elmt_fun
#         grad_ni_to_n!(g.arr[i], sps.structure[i].used_variable, g_res)
#     end
# end
#
# function grad_ni_to_n!(g :: element_gradient{Y}, used_var :: Vector{Int}, g_res :: AbstractVector{Y}) where Y <: Number
#     for i in 1:length(g.g_i)
#         g_res[used_var[i]] += g.g_i[i]
#     end
# end
#
#
# """
#     minus_grad_vec!(g1,g2,res)
# Store in res: g1 minus g2, but g1 and g2 have a particular structure which is grad_vector{T}.
# We need this operation to have the difference for each element gradient for TR method.
# g1 = gₖ and g2 = gₖ₋₁.
# """
# function minus_grad_vec!(g1 :: grad_vector{T}, g2 :: grad_vector{T}, res :: grad_vector{T}) where T <: Number
#     l = length(g1.arr)
#     for i in 1:l
#         res.arr[i].g_i = g1.arr[i].g_i - g2.arr[i].g_i
#     end
# end
#
#
# """
#     evaluate_hessian(sps,x)
# evalutate the hessian of the partially separable function f = ∑ fᵢ, stored in the sps structure
# at the point x. Return the sparse matrix of the hessian of size n × n.
# """
# function evaluate_hessian(sps :: SPS{T}, x :: AbstractVector{Y} ) where T where Y <: Number
#     l_elmt_fun = length(sps.structure)
#     elmt_hess = Vector{Tuple{Vector{Int},Vector{Int},Vector{Y}}}(undef, l_elmt_fun)
#     # @Threads.threads for i in 1:l_elmt_fun # déterminer l'impact sur les performances de array(view())
#     for i in 1:l_elmt_fun
#         elmt_hess[i] = evaluate_element_hessian(sps.structure[i], Array(view(x, sps.structure[i].used_variable))) :: Tuple{Vector{Int},Vector{Int},Vector{Y}}
#     end
#     row = [x[1]  for x in elmt_hess] :: Vector{Vector{Int}}
#     column = [x[2] for x in elmt_hess] :: Vector{Vector{Int}}
#     values = [x[3] for x in elmt_hess] :: Vector{Vector{Y}}
#     G = sparse(vcat(row...) :: Vector{Int} , vcat(column...) :: Vector{Int}, vcat(values...) :: Vector{Y}) :: SparseMatrixCSC{Y,Int}
#     return G
# end
#
#
#
#
# """
#     evaluate_element_hessian(fᵢ,xᵢ)
# Compute the Hessian of the elemental function fᵢ : Gᵢ a nᵢ × nᵢ matrix. So xᵢ a vector of size nᵢ.
# The result of the function is the triplet of the sparse matrix Gᵢ.
# """
# function evaluate_element_hessian(elmt_fun :: element_function{T}, x :: AbstractVector{Y}) where T where Y <: Number
#     # if elmt_fun.type != implementation_type_expr.constant
#     if !CalculusTreeTools.is_linear(elmt_fun.type) && !CalculusTreeTools.is_constant(elmt_fun.type)
#         temp = ForwardDiff.hessian(CalculusTreeTools.evaluate_expr_tree(elmt_fun.fun), x ) :: Array{Y,2}
#         temp_sparse = sparse(temp) :: SparseMatrixCSC{Y,Int}
#         G = SparseMatrixCSC{Y,Int}(elmt_fun.U'*temp_sparse*elmt_fun.U)
#         return findnz(G) :: Tuple{Vector{Int}, Vector{Int}, Vector{Y}}
#     else
#         return (zeros(Int,0), zeros(Int,0), zeros(Y,0))
#     end
# end
#
#
# """
#     struct_hessian(sps,x)
# evalutate the hessian of the partially separable function, stored in the sps structure at the point x. Return the Hessian in a particular structure : Hess_matrix.
# """
# function struct_hessian(sps :: SPS{T}, x :: AbstractVector{Y} ) where T where Y <: Number
#     l_elmt_fun = length(sps.structure)
#     f = ( elm_fun :: element_function{T} -> element_hessian{Y}(zeros(Y, length(elm_fun.used_variable), length(elm_fun.used_variable)) :: Array{Y,2} ) )
#     t = f.(sps.structure) :: Vector{element_hessian{Y}}
#     temp = Hess_matrix{Y}(t)
#     for i in 1:l_elmt_fun # a voir si je laisse le array(view()) la ou non
#         # if sps.structure[i].type != implementation_type_expr.constant
#         if !CalculusTreeTools.is_linear(sps.structure[i].type) && !CalculusTreeTools.is_constant(sps.structure[i].type)
#             @inbounds ForwardDiff.hessian!(temp.arr[i].elmt_hess, CalculusTreeTools.evaluate_expr_tree(sps.structure[i].fun), view(x, sps.structure[i].used_variable) )
#         end
#     end
#     return temp
# end
#
# """
#     struct_hessian!(sps,x,H)
# Evalutate the hessian of the partially separable function, stored in the sps structure at the point x. Store the Hessian in a particular structure H :: Hess_matrix.
# """
# function struct_hessian!(sps :: SPS{T}, x :: AbstractVector{Y}, H :: Hess_matrix{Y} )  where Y <: Number where T
#     map( elt_fun -> element_hessian{Y}(ForwardDiff.hessian!(H.arr[elt_fun.index].elmt_hess :: Array{Y,2}, CalculusTreeTools.evaluate_expr_tree(elt_fun.fun :: T), view(x, elt_fun.used_variable :: Vector{Int}) )), sps.structure :: Vector{element_function{T}})
# end
#
#
# """
#     id_hessian!(sps, B)
# Construct a kinf of Id Hessian, it will initialize each element Hessian Bᵢ with an Id matrix, B =  ∑ᵢᵐ Uᵢᵀ Bᵢ Uᵢ
# """
# function id_hessian!(sps :: SPS{T}, H :: Hess_matrix{Y} )  where Y <: Number where T
#     for i in 1:length(sps.structure)
#         nᵢ = length(sps.structure[i].used_variable)
#         H.arr[sps.structure[i].index].elmt_hess[:] = Matrix{Y}(I, nᵢ, nᵢ)
#     end
# end
#
# """
#     construct_Sparse_Hessian(sps, B)
# Build from the Partially separable Structure sps and the Hessian approximation B a SpaseArray which represent B in other form.
# """
# function construct_Sparse_Hessian(sps :: SPS{T}, H :: Hess_matrix{Y} )  where Y <: Number where T
#     mapreduce(elt_fun :: element_function{T} -> elt_fun.U' * sparse(H.arr[elt_fun.index].elmt_hess) * elt_fun.U, +, sps.structure  )
# end
#
#
#
# """
#     product_matrix_sps(sps,B,x)
# This function make the product of the structure B which represents a symetric matrix and the vector x.
# We need the structure sps for the variable used in each B[i], to replace B[i]*x[i] in the result vector.
# """
# function product_matrix_sps(sps :: SPS{T}, B :: Hess_matrix{Y}, x :: Vector{Y}) where T where Y <: Number
#     Bx = Vector{Y}(undef, sps.n_var)
#     product_matrix_sps!(sps,B,x, Bx)
#     return Bx
# end
#
# """
#     product_matrix_sps!(sps,B,x,Bx)
# This function make the product of the structure B which represents a symetric matrix and the vector x, the result is stored in Bx.
# We need the structure sps for the variable used in each B[i], to replace B[i]*x[i] in the result vector by using f_inter!.
# """
# function product_matrix_sps!(sps :: SPS{T}, B :: Hess_matrix{Y}, x :: AbstractVector{Y}, Bx :: AbstractVector{Y}) where T where Y <: Number
#     l_elmt_fun = length(sps.structure)
#     Bx .= (zeros(Y, sps.n_var))
#     for i in 1:l_elmt_fun
#         temp = B.arr[i].elmt_hess :: Array{Y,2} * view(x, sps.structure[i].used_variable) :: SubArray{Y,1,Array{Y,1},Tuple{Array{Int64,1}},false}
#         f_inter!(Bx, sps.structure[i].used_variable, temp )
#     end
# end
#
# function f_inter!(res :: AbstractVector{Z}, indices ::  AbstractVector{Int}, values :: AbstractVector{Z}) where Z <: Number
#     l = length(indices)
#     for i in 1:l
#         @inbounds res[indices[i]] += values[i]
#     end
# end
#
#
# #= Fonction faisant la même chose mais moins efficacement (nul) =#
#     #=
#     function hess_matrix_dot_vector(sps :: SPS{T}, B :: Hess_matrix{Y}, x :: Vector{Y}) where T where Y <: Number
#         #utilisation de sparse car elt_fun.U' est un sparseArray et le produit entre les sparseArray est plus rapide
#         mapreduce(elt_fun :: element_function{T} -> elt_fun.U' * sparse(B.arr[elt_fun.index].elmt_hess * view(x, elt_fun.used_variable)), + , sps.structure)
#     end
#     function inefficient_product_matrix_sps(sps :: SPS{T}, B :: Hess_matrix{Y}, x :: Vector{Y}) where T where Y <: Number
#         construct_Sparse_Hessian(sps,B) * x
#     end
#     =#
#
# """
#     product_vector_sps(sps, g, x)
# compute the product g⊤ x = ∑ Uᵢ⊤ gᵢ⊤ xᵢ. So we need the sps structure to get the Uᵢ.
# On ne s'en sert pas en pratique mais peut-être pratique pour faire des vérifications
# """
# function product_vector_sps(sps :: SPS{T}, g :: grad_vector{Y}, x :: Vector{Z}) where T where Y <: Number where Z <: Number
#     l_elmt_fun = length(sps.structure)
#     res = Vector{Y}(undef,l_elmt_fun) #vecteur stockant le résultat des gradient élémentaire g_i * x_i
#     # à voir si on ne passe pas sur un résultat direct avec des opérations atomique
#     for i in 1:l_elmt_fun
#         res[i] = g.arr[i]' * Array(view(x, sps.structure[i].used_variable))
#     end
#     return sum(res)
# end
#
# """
#     check_Inf_Nan(B)
# function that check if an Hess_matrix is not full of Nan.
# """
# function check_Inf_Nan( B :: Hess_matrix{Y}) where Y <: Number
#     res = []
#     for i in 1:length(B.arr)
#         interet = check_Inf_Nan(B.arr[i].elmt_hess)
#         if interet != nothing
#             push!(res, (i,interet))
#         end
#     end
#     return res
# end
#
# function check_Inf_Nan(Bi :: Array{Y,2}) where Y <: Number
#     for i in Bi
#         if isnan(i) || isinf(i)
#             println("oui")
#             return Bi
#         end
#     end
# end
