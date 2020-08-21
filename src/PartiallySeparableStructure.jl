
using CalculusTreeTools

using ForwardDiff, SparseArrays, LinearAlgebra, ReverseDiff
using Base.Threads
import Base.-

mutable struct element_function
    index_fun :: Int
    index_element_tree_position :: Int
    type :: CalculusTreeTools.type_calculus_tree
    used_variable :: Vector{Int}
    convexity_status :: CalculusTreeTools.convexity_wrapper
    index :: Int
end

@inline get_index_fun( elmt_fun :: element_function) = elmt_fun.index_fun
@inline get_index_element_tree_position(elmt_fun :: element_function) = elmt_fun.index_element_tree_position
@inline get_fun_from_elmt_fun( elmt_fun :: element_function, diff_elmt_fun :: Vector{T}) where T = diff_elmt_fun[elmt_fun.index_fun]
@inline get_type( elmt_fun :: element_function) = elmt_fun.type
@inline get_used_variable( elmt_fun :: element_function) = elmt_fun.used_variable
@inline get_convexity_status( elmt_fun :: element_function) = elmt_fun.convexity_status
@inline get_index( elmt_fun :: element_function) = elmt_fun.index


mutable struct SPS{T, N <: Number}
    structure :: Vector{element_function}

    different_element_tree :: Vector{T}
    index_element_tree :: Vector{Vector{Int}}
    related_vars :: Vector{Vector{Vector{Int}}}

    obj_pre_compiled_trees :: Vector{CalculusTreeTools.pre_n_compiled_tree{N}}

    x :: Vector{N}
    x_views :: Vector{Vector{SubArray{N,1,Array{N,1},Tuple{Array{Int64,1}},false}}}

    v :: Vector{N}
    v_views :: Vector{Vector{SubArray{N,1,Array{N,1},Tuple{Array{Int64,1}},false}}}

    vec_length :: Int
    n_var :: Int
    compiled_gradients :: Vector{ReverseDiff.CompiledTape}
end

@inline get_obj_pre_compiled_trees(sps :: SPS{T,N}) where T where N <: Number = sps.obj_pre_compiled_trees
@inline get_fun_from_elmt_fun( elmt_fun :: element_function, sps :: SPS{T}) where T = sps.different_element_tree[elmt_fun.index_fun]
@inline get_var_from_elmt_fun( elmt_fun :: element_function, sps :: SPS{T}) where T = sps.x_views[get_index_fun(elmt_fun)][get_index_element_tree_position(elmt_fun)]
@inline get_element_function( sps :: SPS{T}, index :: Int) where T = get_structure(sps)[index]
@inline get_structure(sps :: SPS{T}) where T = sps.structure
@inline get_different_element_tree(sps :: SPS{T}) where T = sps.different_element_tree
@inline get_index_element_tree(sps :: SPS{T}) where T = sps.index_element_tree
@inline get_related_vars(sps :: SPS{T}) where T = sps.related_vars
@inline get_x_views(sps :: SPS{T,N}) where T where N <: Number = sps.x_views
@inline set_x_sps(sps :: SPS{T,N}, v :: AbstractVector{N}) where T where N <: Number = sps.x .= v
@inline get_v_views(sps :: SPS{T,N}) where T where N <: Number = sps.v_views
@inline set_v_sps(sps :: SPS{T,N}, vector :: AbstractVector{N}) where T where N <: Number = sps.v .= vector
@inline get_compiled_gradient(sps :: SPS{T,N}) where T where N <: Number = sps.compiled_gradients


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


@inline get_structure(sps :: SPS{T}) where T = sps.structure
@inline get_convexity_status(elmt_fun :: element_function)  = elmt_fun.convexity_status



"""
    deduct_partially_separable_structure(expr_tree, n)

Find the partially separable structure of a function f stored as an expression tree expr_tree.
To define properly the size of sparse matrix we need the size of the problem : n.
At the end, we get the partially separable structure of f, f(x) = ∑fᵢ(xᵢ)
"""
deduct_partially_separable_structure(a :: Any, n :: Int, type=Float64 :: DataType) = _deduct_partially_separable_structure(a, n, type)

function _deduct_partially_separable_structure(tree :: T , n :: Int, type=Float64 :: DataType) where T
    # transformation of the tree of type T into an expr tree of type t_expr_tree (the standard type used by my algorithms)
    expr_tree = CalculusTreeTools.transform_to_expr_tree(tree) :: CalculusTreeTools.t_expr_tree
    # work_tree = copy(expr_tree) # peut être non nécessaire
    elmt_fun = CalculusTreeTools.delete_imbricated_plus(work_tree) :: Vector{CalculusTreeTools.t_expr_tree} #séparation en fonction éléments
    m_i = length(elmt_fun)

    #récupération des variables élémentaires
    elmt_var_i =  Vector{ Vector{Int}}(undef,m_i)
    length_vec = (Int)(0)
    for i in 1:m_i
        elmt_var_i[i] = CalculusTreeTools.get_elemental_variable(elmt_fun[i])
        length_vec += length(elmt_var_i[i])
    end
    sort!.(elmt_var_i) #ligne importante, met dans l'ordre les variables élémentaires. Utile pour les U_i et le N_to_Ni

    # renumérotation des variables des fonctions éléments en variables internes
    for i in 1:m_i
        CalculusTreeTools.element_fun_from_N_to_Ni!(elmt_fun[i],elmt_var_i[i])
    end

    # à partir des fonctions éléments renumérotées factorisation en fonction éléments distinctes
    (different_calculus_expr_tree, different_calculus_tree_index) = get_different_CalculusTree(elmt_fun)
    # creation d'arbres complets à partir des ces fonctions éléments distinctes
    temp_complete_trees = CalculusTreeTools.create_complete_tree.(different_calculus_expr_tree)

    # cast des ces arbres complets au type voulu
    different_calculus_complete_trees = map(tree -> CalculusTreeTools.cast_type_of_constant(tree, type), temp_complete_trees)

    #calcul des bornes et de la convexité de ces arbres complets distincts
    CalculusTreeTools.set_bounds!.(different_calculus_complete_trees)
    CalculusTreeTools.set_convexity!.(different_calculus_complete_trees)
    # information concernant la convexité des arbres complets distincts
    convexity_wrapper = map( (complete_tree -> CalculusTreeTools.convexity_wrapper(CalculusTreeTools.get_convexity_status(complete_tree)) ), different_calculus_complete_trees)

    # récupération des types des fonctions éléments
    m_diff_element_fun = length(different_calculus_complete_trees)
    type_i = Vector{CalculusTreeTools.type_calculus_tree}(undef, m_diff_element_fun)
    for i in 1:m_diff_element_fun
        type_i[i] = CalculusTreeTools.get_type_tree(different_calculus_complete_trees[i])
    end

    #création des fonctions éléments
    structure = Vector{element_function}(undef,m_i)
    for i in 1:m_i
        index_distinct_element_tree = different_calculus_tree_index[i]
        undefined_index = 0
        structure[i] = element_function(index_distinct_element_tree, undefined_index, type_i[index_distinct_element_tree], elmt_var_i[i], convexity_wrapper[index_distinct_element_tree], i)
    end

    # on donne une valeur au champs index_element_tree_position de chaque fonction élément
    construct_set_index_element_view!(structure, different_calculus_complete_trees)
    # récupère pour chaque arbre complet distinct l'ensemble des fonctions éléments qui lui sont affectées.
    index_element_tree = get_related_function(structure, different_calculus_complete_trees)
    # récupère les indices assoxiés aux fonctions éléménts associées à chaque arbre complet distinct
    related_vars = get_related_var(structure, index_element_tree)

    # pré compilation du gradient.
    compiled_gradients = map(x -> compiled_grad_of_elmt_fun(x), different_calculus_complete_trees)

    x = Vector{type}(undef,n)
    x_views = construct_views(x, related_vars)

    distinct_casted_expr_tree = map(tree -> CalculusTreeTools.cast_type_of_constant(tree, type), different_calculus_expr_tree)
    obj_pre_compiled_trees = create_pre_compiled_tree(distinct_casted_expr_tree, x_views)


    v = Vector{type}(undef,n)
    v_views = construct_views(v, related_vars)

    sps = SPS{CalculusTreeTools.complete_expr_tree, type}(structure,
                                                          different_calculus_complete_trees,
                                                          index_element_tree,
                                                          related_vars,
                                                          obj_pre_compiled_trees,
                                                          x,
                                                          x_views,
                                                          v,
                                                          v_views,
                                                          length_vec,
                                                          n,
                                                          compiled_gradients)
    return sps
end


"""
    create_pre_compiled_tree(vector_calculus_trees, vector_x_views)
create a precompiled tree for a particuliar number of evaluation. The size of vector_calculus_trees and vector_x_views must match.
Each calculus_tree from vector_calculus_trees is pre-compiled for nᵢ evaluation, nᵢ the size of the corresponding element of vector_x_views
"""
function create_pre_compiled_tree(different_calculus_expr_tree :: Vector{CalculusTreeTools.t_expr_tree}, x_views :: Vector{Vector{SubArray{N,1,Array{N,1},Tuple{Array{Int64,1}},false}}} ) where N <: Number
    n = length(different_calculus_expr_tree)
    n == length(x_views) || error("error in  create_pre_compiled_tree, distinct element function does not match x_views")
    res = Vector{CalculusTreeTools.pre_n_compiled_tree{N}}(undef, n)
    for i in 1:n  # for each tree create a pre-compiled tree according to the correspondinf x_views wich represent the number of evaluation
        res[i] = CalculusTreeTools.create_pre_n_compiled_tree(different_calculus_expr_tree[i], x_views[i])
    end
    return res
end

"""
    evaluate_obj_pre_compiled(sps, point_x)
Evaluate the structure sps at point_x using the precompiled element tree for nᵢ evaluation.
This function is the current evaluate_SPS.
"""
function evaluate_obj_pre_compiled(sps :: SPS{T,Y}, x :: AbstractVector{Y} ) where T where Y <: Number
    different_element_tree = get_obj_pre_compiled_trees(sps)
    n = length(different_element_tree)
    res_distrinct_elmt_fun = Vector{Y}(undef,n)

    set_x_sps(sps, x) # set the vector x of the structure sps.

    for i in 1:n # we evaluate each pre compiled element tree. No argument x because the variables of the tree is linked with the vector x of sps.
        @inbounds res_distrinct_elmt_fun[i] = CalculusTreeTools.evaluate_expr_tree_multiple_points(different_element_tree[i])
    end

    @inbounds res = sum(res_distrinct_elmt_fun) # according to the sps structure we sum the results
    return res
end


"""
    construct_set_index_element_view(elmt_fun_vector, diff_elmt_tree)
For each elmt_fun ∈ elmt_fun_vector we set the attribute index_element_tree_position. For doing this we use the function get_elmnt_fun_index_view(elmt_fun_vector, diff_elmt_tree)
that builds a vector of size length(elmt_fun_vector) with the right indexes.
"""
construct_set_index_element_view!(structure :: Vector{element_function}, different_element_tree :: Vector{T}) where T = set_index_element_view!(structure, get_elmnt_fun_index_view(structure, different_element_tree))

function set_index_element_view!(structure :: Vector{element_function}, vector_index_view :: Vector{Int})
    for elmt_fun in structure
        index_elmt_fun = get_index(elmt_fun)
        elmt_fun.index_element_tree_position = vector_index_view[index_elmt_fun]
    end
end



"""
    get_elmnt_fun_index_view(vector_element_fun, different_element_tree)
For each element_fun ∈ vector_element_fun, returns the position of the linked element tree corresponding in different_element_tree.
It will be use later for retrieve a view of the variable used by each element function.
"""
function get_elmnt_fun_index_view(structure :: Vector{element_function}, different_element_tree :: Vector{T}) where T
    length_det = length(different_element_tree)
    vector_count = ones(Int, length_det)
    length_struct = length(structure)
    vector_index_view = Vector{Int}(undef, length_struct)

    for elmt_fun in structure
        index_element_tree = get_index_fun(elmt_fun)
        index_elmt_fun = get_index(elmt_fun)
        vector_index_view[index_elmt_fun] = vector_count[index_element_tree]
        vector_count[index_element_tree] +=1
    end

    return vector_index_view
end


"""
    construct_views(x, related_vars)
- related_vars a vector (from different element tree) of vector (several element function linked with each tree)
of vector{int} (a set of index, which represent the elemental variable).
- x a point ∈ Rⁿ
returns a view of x for each set of index representing the elemental variable.
"""
function construct_views(x :: Vector{N}, related_vars :: Vector{Vector{Vector{Int}}}) where N <: Number
    nb_elmt_fun = length(related_vars)
    res = Vector{Vector{SubArray{N,1,Array{N,1},Tuple{Array{Int64,1}},false}}}(undef, nb_elmt_fun)
    for i in 1:nb_elmt_fun
        res[i] = map( (y -> view(x,y)), related_vars[i])
    end
    return res
end


"""
    get_different_CalculusTree( element_functions)
Return a vector diffferent_elmt_fun of calculus tree from the vector of calculus tree element_function. diffferent_elmt_fun delete the
"""
function get_different_CalculusTree(all_elemt_fun :: Vector{T}) where T
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
    return (different_calculus_tree, different_calculus_tree_index)
end


"""
    get_related_var(vec_elmt_fun, index_elmt_tree)
Renvoie un tableau d'entier de 3 dimensions. La première dimension de taille length(index_elmt_tree) correspond au nombre de fonctions éléments distinctes (arbre de calcul).
Pour chaque fonction éléments distinctes e nous avons un vecteur vₑ :: Vector{Vector{Int}} tel que length(vₑ) == length(index_elmt_tree[e])
index_elmt_tree[e] contient les indices des fonctions éléments rattachés à la fonction élément e.
Le résultat de la fonction renverra pour chaque indice de fonctions éléments les variables utilisées ( :: Vector{Int}) par celle-ci ce qui nous donne un tableau 3 dimensions.
"""
function get_related_var(v_element_function :: Vector{element_function}, index_elmt_tree :: Vector{Vector{Int}})
    new_needed_var(index) = Vector{Int}(v_element_function[index].used_variable)
    new_view_vars(index_single_elmt_tree) = Vector{Vector{Int}}((index -> new_needed_var(index)).(index_single_elmt_tree) )
    related_vars = Vector{Vector{Vector{Int}}}( ( index_single_elmt_tree -> new_view_vars(index_single_elmt_tree)).(index_elmt_tree) )
    return related_vars
end


"""
    get_related_function(elmt_functionS, calculusTreeS)
Cette fonction récupère les indices des fonctions éléments elmt_functionS lié aux quelques arbres de calculs calculusTreeS de la structure
"""
function get_related_function(v_element_function :: Vector{element_function}, diff_element_tree :: Vector{T}) where T
    l_elmt_tree = length(diff_element_tree)
    fempty_vector() = Vector{Int}(undef,0)
    vector_index_elmt_fun = Vector{Vector{Int}}( (x -> fempty_vector()).([1:l_elmt_tree;]))
    for i in v_element_function
        push!((vector_index_elmt_fun[get_index_fun(i)]) , get_index(i))
    end
    return vector_index_elmt_fun
end


"""
    compiled_grad_of_elmt_fun(elmt_fun)
Return  the GradientTape compiled to speed up the ReverseDiff computation of the elmt_fun gradient in the future
"""
function compiled_grad_of_elmt_fun(elmt_fun :: T ) where T
    f = CalculusTreeTools.evaluate_expr_tree(elmt_fun)
    n = length(CalculusTreeTools.get_elemental_variable(elmt_fun))
    f_tape = ReverseDiff.GradientTape(f, rand(n))
    compiled_f_tape = ReverseDiff.compile(f_tape)
    return compiled_f_tape
end


"""
    evaluate_SPS(sps,x)
Evaluate the structure sps on the point x ∈ Rⁿ. Since we work on subarray of x, we allocate x to the structure sps in a first time.
Once this step is done we select the calculus tree needed as well as the view of x needed.
Then we evaluate the different calculus tree on the needed point (since the same function appear a lot of time). At the end we sum the result
"""
@inline evaluate_SPS(sps :: SPS{T,Y}, x :: AbstractVector{Y} ) where T where Y <: Number = evaluate_obj_pre_compiled(sps, x)
@inline evaluate_SPS(sps :: SPS{T,Y}) where T where Y <: Number = (x :: AbstractVector{Y} -> evaluate_SPS(sps,x) )


"""
    evaluate_gradient(sps,x)
evalutate the gradient of the partially separable function f = ∑ fι, stored in the sps structure
at the point x, return a vector of size n (the number of variable) which is the gradient.
Première version de la fonction inutile car inefficace.
"""
function evaluate_gradient(sps :: SPS{T}, x :: Vector{Y} ) where T where Y <: Number
    l_elmt_fun = length(sps.structure)
    gradient_prl = Vector{Threads.Atomic{Y}}((x-> Threads.Atomic{Y}(0)).([1:sps.n_var;]) )
     for i in 1:l_elmt_fun
        U = CalculusTreeTools.get_Ui(get_used_variable(sps.structure[i]), sps.n_var)
        if isempty(U) == false
            (row, column, value) = findnz(U)
            temp = ForwardDiff.gradient(CalculusTreeTools.evaluate_expr_tree(get_fun_from_elmt_fun(sps.structure[i],sps)), view(x, sps.structure[i].used_variable)  )
            atomic_add!.(gradient_prl[column], temp)
        end
    end
    gradient = (x -> x[]).(gradient_prl) :: Vector{Y}
    return gradient
end


"""
    evaluate_SPS_gradient(sps,x)
Return the gradient of the partially separable structure sps at the point x.
Using ReversDiff package. Not obvious good behaviour with Threads.@threads, sometime yes sometime no.
Noted that we use the previously compiled GradientTape in element_gradient! that use ReverseDiff.
"""
function evaluate_SPS_gradient(sps :: SPS{T,Y}, x :: AbstractVector{Y}) where T where Y <: Number
    create_element_gradient = (y :: element_function -> element_gradient{Y}( Vector{typeof(x[1])}( zeros(Y, length(y.used_variable)) )) )
    partitionned_g = grad_vector{Y}( create_element_gradient.(sps.structure) )
    evaluate_SPS_gradient!(sps, x, partitionned_g)
    g = zeros(Y,length(x))
    build_gradient!(sps, partitionned_g, g) #use builds_gradient! function to construct the hv vector of size n for the partitionned element vectors each of them of size nᵢ
    return g
end


"""
    evaluate_SPS_gradient!(sps,x,g)
Compute the gradient of the partially separable structure sps, and store the result in the grad_vector structure g.
Using ReversDiff package. Not obvious good behaviour with Threads.@threads, sometime yes sometime no.
Noted that we use the previously compiled GradientTape in element_gradient! that use ReverseDiff.
"""
function evaluate_SPS_gradient!(sps :: SPS{T}, x :: AbstractVector{Y}, g :: grad_vector{Y} ) where T where Y <: Number
    l_elmt_fun = length(sps.structure)
    for i in 1:l_elmt_fun
        if isempty(sps.structure[i].used_variable) == false  #fonction element ayant au moins une variable
            element_gradient!(sps.compiled_gradients[sps.structure[i].index_fun], view(x, sps.structure[i].used_variable), g.arr[i] )
        end
    end
end


"""
    element_gradient!(compil_tape, x, g)
Compute the element grandient from the compil_tape compiled before according to the vector x, and store the result in the vector g
Use of ReverseDiff
"""
function element_gradient!(compiled_tape :: ReverseDiff.CompiledTape, x :: AbstractVector{T}, g :: element_gradient{T} ) where T <: Number
    ReverseDiff.gradient!(g.g_i, compiled_tape, x)
end


"""
    Hv(sps, x, v)
Compute the product hessian vector of the hessian at the point x dot the vector v.
This version both ReverseDiff and ForwardDiff.
"""
Hv(sps :: SPS{T,Y}, v :: AbstractVector{Y}) where T where Y <: Number = Hv(sps, sps.x, v)
Hv(sps :: SPS{T,Y}, x :: AbstractVector{Y}, v :: AbstractVector{Y}) where T where Y <: Number = begin hv = similar(x); Hv!(hv,sps,x,v); return hv end
function Hv!(hv :: AbstractVector{Y}, sps :: SPS{T,Y}, x :: AbstractVector{Y}, v :: AbstractVector{Y}) where T where Y <: Number
    f = (y :: element_function -> element_gradient{typeof(x[1])}(Vector{typeof(x[1])}(zeros(typeof(x[1]), length(y.used_variable)) )) )
    partitionned_hv = grad_vector{typeof(x[1])}( f.(sps.structure) )

    l_elmt_fun = length(sps.structure)
    diff_element_tree = get_different_element_tree(sps)
    structure = get_structure(sps)
    set_x_sps(sps, x)
    set_v_sps(sps, v)
    x_views = get_x_views(sps)
    v_views = get_v_views(sps)
    for i in 1:l_elmt_fun
        if isempty(structure[i].used_variable) == false  #fonction element ayant au moins une variable
            index_fun = get_index_fun(structure[i])
            index_position_x = get_index_element_tree_position(structure[i])
            Hv_elem!(partitionned_hv.arr[i], diff_element_tree[index_fun], x_views[index_fun][index_position_x], v_views[index_fun][index_position_x])
        else
            partitionned_hv.arr[i].g_i .= zeros(length(partitionned_hv.arr[i].g_i))
        end
    end
    build_gradient!(sps, partitionned_hv, hv) #use builds_gradient! function to construct the hv vector of size n for the partitionned element vectors each of them of size nᵢ
end


@inline SPS_ReverseDiff_grad(tree, x :: AbstractVector{Y}) where Y <: Number = begin g = similar(x) ; SPS_ReverseDiff_grad!(g, tree, x) end
@inline SPS_ReverseDiff_grad!(g :: AbstractVector{Y}, tree, x) where Y <: Number = ReverseDiff.gradient!(g, CalculusTreeTools.evaluate_expr_tree(tree), x)

@inline ∇²fv(tree, x, v) = ForwardDiff.gradient(x -> dot(SPS_ReverseDiff_grad(tree, x), v), x)
@inline ∇²fv!(tree, x, v, hv) = (hv .= ∇²fv(tree, x, v))
@inline Hv_elem!(elemental_hv :: element_gradient, tree :: T, x :: AbstractVector{Y}, v :: AbstractVector{Y} ) where Y <: Number where T = ∇²fv!(tree, x, v, elemental_hv.g_i)




"""
    Hv2(sps, x, v)
Compute the product hessian vector of the hessian at the point x dot the vector v.
This version uses only ReverseDiff.
"""
Hv2(sps :: SPS{T,Y}, v :: AbstractVector{Y}) where T where Y <: Number = Hv2(sps, sps.x, v)
Hv2(sps :: SPS{T,Y}, x :: AbstractVector{Y}, v :: AbstractVector{Y}) where T where Y <: Number = begin hv = similar(x); Hv2!(hv,sps,x,v); return hv end
function Hv2!(hv :: AbstractVector{Y}, sps :: SPS{T,Y}, x :: AbstractVector{Y}, v :: AbstractVector{Y}) where T where Y <: Number
    f = (y :: element_function -> element_gradient{typeof(x[1])}(Vector{typeof(x[1])}(zeros(typeof(x[1]), length(y.used_variable)) )) )
    partitionned_hv = grad_vector{typeof(x[1])}( f.(sps.structure) )

    l_elmt_fun = length(sps.structure)
    diff_element_tree = get_different_element_tree(sps)
    structure = get_structure(sps)
    set_x_sps(sps, x)
    set_v_sps(sps, v)
    x_views = get_x_views(sps)
    v_views = get_v_views(sps)
    for i in 1:l_elmt_fun
        if isempty(structure[i].used_variable) == false  #fonction element ayant au moins une variable
            index_fun = get_index_fun(structure[i])
            index_position_x = get_index_element_tree_position(structure[i])
            Hv_elem2!(partitionned_hv.arr[i], diff_element_tree[index_fun], x_views[index_fun][index_position_x], v_views[index_fun][index_position_x])
        else
            partitionned_hv.arr[i].g_i .= zeros(length(partitionned_hv.arr[i].g_i))
        end
    end
    build_gradient!(sps, partitionned_hv, hv) #use builds_gradient! function to construct the hv vector of size n for the partitionned element vectors each of them of size nᵢ
end


@inline SPS_ReverseDiff_grad2(tree, x :: AbstractVector{Y}) where Y <: Number = begin g = similar(x) ; SPS_ReverseDiff_grad2!(g, tree, x) end
@inline SPS_ReverseDiff_grad2!(g :: AbstractVector{Y}, tree, x) where Y <: Number = ReverseDiff.gradient!(g, CalculusTreeTools.evaluate_expr_tree(tree), x)

@inline ∇²fv2(tree, x, v) = ReverseDiff.gradient(x -> dot(SPS_ReverseDiff_grad2(tree, x), v), x)
@inline ∇²fv2!(tree, x, v, hv) = (hv .= ∇²fv2(tree, x, v))
@inline Hv_elem2!(elemental_hv :: element_gradient, tree :: T, x :: AbstractVector{Y}, v :: AbstractVector{Y} ) where Y <: Number where T = ∇²fv2!(tree, x, v, elemental_hv.g_i)




"""
    build_gradient(sps, g)
Constructs a vector of size n from the list of element gradient of the sps structure which has numerous element gradient of size nᵢ.
The purpose of the function is to gather these element gradient into a real gradient of size n.
The function grad_ni_to_n! will add element gradient of size nᵢ at the right inside the gradient of size n.
"""
function build_gradient(sps :: SPS{T}, g :: grad_vector{Y}) where T where Y <: Number
    grad = Vector{Y}(undef, sps.n_var)
    build_gradient!(sps, g, grad)
    return grad
end

function build_gradient!(sps :: SPS{T}, g :: grad_vector{Y}, g_res :: AbstractVector{Y}) where T where Y <: Number
    g_res[:] = zeros(Y, sps.n_var)
    l_elmt_fun = length(sps.structure)
    for i in 1:l_elmt_fun
        grad_ni_to_n!(g.arr[i], sps.structure[i].used_variable, g_res)
    end
end

"""
    grad_ni_to_n!(element_gradient, used_var, gradient)
Add to the gradient the value of element_gradient according to the vector of used_var given.
length(used_var) == length(element_gradient)
"""
function grad_ni_to_n!(g :: element_gradient{Y}, used_var :: Vector{Int}, g_res :: AbstractVector{Y}) where Y <: Number
    for i in 1:length(g.g_i)
        g_res[used_var[i]] += g.g_i[i]
    end
end


"""
    minus_grad_vec!(g1,g2,res)
Store in res: g1 minus g2, but g1 and g2 have a particular structure which is grad_vector{T}.
We need this operation to have the difference for each element gradient for TR method.
g1 = gₖ and g2 = gₖ₋₁.
"""
function minus_grad_vec!(g1 :: grad_vector{T}, g2 :: grad_vector{T}, res :: grad_vector{T}) where T <: Number
    l = length(g1.arr)
    for i in 1:l
        res.arr[i].g_i = g1.arr[i].g_i - g2.arr[i].g_i
    end
end


"""
    evaluate_hessian(sps,x)
evalutate the hessian of the partially separable function f = ∑ fᵢ, stored in the sps structure
at the point x. Return the sparse matrix of the hessian of size n × n.
"""
function evaluate_hessian(sps :: SPS{T}, x :: AbstractVector{Y} ) where T where Y <: Number
    l_elmt_fun = length(sps.structure)
    elmt_hess = Vector{Tuple{Vector{Int},Vector{Int},Vector{Y}}}(undef, l_elmt_fun)
    # @Threads.threads for i in 1:l_elmt_fun # déterminer l'impact sur les performances de array(view())
    for i in 1:l_elmt_fun
        elmt_hess[i] = evaluate_element_hessian(sps.structure[i], Array(view(x, sps.structure[i].used_variable)), sps) :: Tuple{Vector{Int},Vector{Int},Vector{Y}}
    end
    row = [x[1]  for x in elmt_hess] :: Vector{Vector{Int}}
    column = [x[2] for x in elmt_hess] :: Vector{Vector{Int}}
    values = [x[3] for x in elmt_hess] :: Vector{Vector{Y}}
    G = sparse(vcat(row...) :: Vector{Int} , vcat(column...) :: Vector{Int}, vcat(values...) :: Vector{Y}) :: SparseMatrixCSC{Y,Int}
    return G
end




"""
    evaluate_element_hessian(fᵢ,xᵢ)
Compute the Hessian of the elemental function fᵢ : Gᵢ a nᵢ × nᵢ matrix. So xᵢ a vector of size nᵢ.
The result of the function is the triplet of the sparse matrix Gᵢ.
"""
function evaluate_element_hessian(elmt_fun :: element_function, x :: AbstractVector{Y}, sps :: SPS{T}) where T where Y <: Number
    # if elmt_fun.type != implementation_type_expr.constant
    if !CalculusTreeTools.is_linear(elmt_fun.type) && !CalculusTreeTools.is_constant(elmt_fun.type)
        temp = ForwardDiff.hessian(CalculusTreeTools.evaluate_expr_tree(get_fun_from_elmt_fun(elmt_fun,sps)), x ) :: Array{Y,2}
        temp_sparse = sparse(temp) :: SparseMatrixCSC{Y,Int}
        G = SparseMatrixCSC{Y,Int}(elmt_fun.U'*temp_sparse*elmt_fun.U)
        return findnz(G) :: Tuple{Vector{Int}, Vector{Int}, Vector{Y}}
    else
        return (zeros(Int,0), zeros(Int,0), zeros(Y,0))
    end
end


"""
    struct_hessian(sps,x)
evalutate the hessian of the partially separable function, stored in the sps structure at the point x. Return the Hessian in a particular structure : Hess_matrix.
"""
function struct_hessian(sps :: SPS{T}, x :: AbstractVector{Y} ) where T where Y <: Number
    l_elmt_fun = length(sps.structure)
    f = ( elm_fun :: element_function -> element_hessian{Y}(zeros(Y, length(elm_fun.used_variable), length(elm_fun.used_variable)) :: Array{Y,2} ) )
    t = f.(sps.structure) :: Vector{element_hessian{Y}}
    temp = Hess_matrix{Y}(t)
    for i in 1:l_elmt_fun # a voir si je laisse le array(view()) la ou non
        # if sps.structure[i].type != implementation_type_expr.constant
        if !CalculusTreeTools.is_linear(sps.structure[i].type) && !CalculusTreeTools.is_constant(sps.structure[i].type)
            @inbounds ForwardDiff.hessian!(temp.arr[i].elmt_hess, CalculusTreeTools.evaluate_expr_tree(get_fun_from_elmt_fun(sps.structure[i], sps)), view(x, sps.structure[i].used_variable) )
        end

    end
    return temp
end

"""
    struct_hessian!(sps,x,H)
Evalutate the hessian of the partially separable function, stored in the sps structure at the point x. Store the Hessian in a particular structure H :: Hess_matrix.
"""
function struct_hessian!(sps :: SPS{T}, x :: AbstractVector{Y}, H :: Hess_matrix{Y} )  where Y <: Number where T
    # map( elt_fun :: element_function -> element_hessian{Y}(ForwardDiff.hessian!(H.arr[elt_fun.index].elmt_hess :: Array{Y,2}, CalculusTreeTools.evaluate_expr_tree(get_fun_from_elmt_fun(elt_fun, sps) :: T), view(x, elt_fun.used_variable :: Vector{Int}) )), sps.structure :: Vector{element_function{T}})
    for i in sps.structure
        (ForwardDiff.hessian!(H.arr[i.index].elmt_hess :: Array{Y,2}, CalculusTreeTools.evaluate_expr_tree(get_fun_from_elmt_fun(i, sps) :: T), view(x, i.used_variable :: Vector{Int}) ))
    end
end



"""
    id_hessian!(sps, B)
Construct a kinf of Id Hessian, it will initialize each element Hessian Bᵢ with an Id matrix, B =  ∑ᵢᵐ Uᵢᵀ Bᵢ Uᵢ
"""
function id_hessian!(sps :: SPS{T}, H :: Hess_matrix{Y} )  where Y <: Number where T
    for i in 1:length(sps.structure)
        nᵢ = length(sps.structure[i].used_variable)
        H.arr[sps.structure[i].index].elmt_hess[:] = Matrix{Y}(I, nᵢ, nᵢ)
    end
end


construct_full_Hessian(sps :: SPS{T,Y}, H :: Hess_matrix{Y} )  where Y <: Number where T =  Array(construct_Sparse_Hessian,H)

"""
    construct_Sparse_Hessian(sps, B)
Build from the Partially separable Structure sps and the Hessian approximation B a SpaseArray which represent B in other form.
"""
function construct_Sparse_Hessian(sps :: SPS{T}, H :: Hess_matrix{Y} )  where Y <: Number where T
    mapreduce(elt_fun :: element_function -> CalculusTreeTools.get_Ui(get_used_variable(elt_fun), sps.n_var)' * sparse(H.arr[elt_fun.index].elmt_hess) * CalculusTreeTools.get_Ui(get_used_variable(elt_fun), sps.n_var), +, sps.structure  )
end



"""
    product_matrix_sps(sps,B,x)
This function make the product of the structure B which represents a symetric matrix and the vector x.
We need the structure sps for the variable used in each B[i], to replace B[i]*x[i] in the result vector.
"""
function product_matrix_sps(sps :: SPS{T}, B :: Hess_matrix{Y}, x :: Vector{Y}) where T where Y <: Number
    Bx = Vector{Y}(undef, sps.n_var)
    product_matrix_sps!(sps,B,x, Bx)
    return Bx
end

"""
    product_matrix_sps!(sps,B,x,Bx)
This function make the product of the structure B which represents a symetric matrix and the vector x, the result is stored in Bx.
We need the structure sps for the variable used in each B[i], to replace B[i]*x[i] in the result vector by using f_inter!.
"""
function product_matrix_sps!(sps :: SPS{T}, B :: Hess_matrix{Y}, x :: AbstractVector{Y}, Bx :: AbstractVector{Y}) where T where Y <: Number
    @inbounds sps.x .= x
    l_elmt_fun = length(sps.structure)
    Bx .= (zeros(Y, sps.n_var))
    for i in 1:l_elmt_fun
        @fastmath @inbounds temp = B.arr[i].elmt_hess :: Array{Y,2} * sps.x_views[sps.structure[i].index_fun][sps.structure[i].index_element_tree_position] :: SubArray{Y,1,Array{Y,1},Tuple{Array{Int64,1}},false}
        f_inter!(Bx, sps.structure[i].used_variable, temp )
    end
end


"""

"""
function f_inter!(res :: AbstractVector{Z}, indices ::  AbstractVector{Int}, values :: AbstractVector{Z}) where Z <: Number
    l = length(indices)
    for i in 1:l
        @inbounds res[indices[i]] += values[i]
    end
end



#= Fonction en développement =#
"""
    Hv_only_product(sps, Hess_matrix, x)
return the product dot(Hess_matrix, x), the function fits with the partitionned representation of the matrix Hess_matrix.
In order to do the right operations in the right order we need the partially separable structure sps. Less efficient than product_matrix_sps.
"""
function Hv_only_product(sps :: SPS{T}, B :: Hess_matrix{Y}, x :: AbstractVector{Y}) where T where Y <: Number
    Bx = Vector{Y}(zeros(Y, sps.n_var))
    Hv_only_product!(sps,B,x,Bx)
    return Bx
end

function Hv_only_product!(sps :: SPS{T}, B :: Hess_matrix{Y}, x :: AbstractVector{Y}, Bx :: AbstractVector{Y}) where T where Y <: Number
    sps.x .= x
    structure = get_structure(sps)
    x_views = get_x_views(sps)
    for elmt_fun in structure
        @inbounds index_elmt_fun = get_index(elmt_fun)
        @inbounds index_calculus_tree = get_index_fun(elmt_fun)
        @inbounds index_position = get_index_element_tree_position(elmt_fun)
        @inbounds view(Bx, get_used_variable(elmt_fun)) .+= (B.arr[index_elmt_fun].elmt_hess * x_views[index_calculus_tree][index_position])
    end
end






#= FIN Fonction de développement =#

#= FONCTIONS DE TESTS =#

"""
    product_vector_sps(sps, g, x)
compute the product g⊤ x = ∑ Uᵢ⊤ gᵢ⊤ xᵢ. So we need the sps structure to get the Uᵢ.
On ne s'en sert pas en pratique mais peut-être pratique pour faire des vérifications
"""
function product_vector_sps(sps :: SPS{T}, g :: grad_vector{Y}, x :: Vector{Z}) where T where Y <: Number where Z <: Number
    l_elmt_fun = length(sps.structure)
    res = Vector{Y}(undef,l_elmt_fun) #vecteur stockant le résultat des gradient élémentaire g_i * x_i
    # à voir si on ne passe pas sur un résultat direct avec des opérations atomique
    for i in 1:l_elmt_fun
        res[i] = g.arr[i]' * Array(view(x, sps.structure[i].used_variable))
    end
    return sum(res)
end


"""
    check_Inf_Nan(B)
function that check if an Hess_matrix is not full of Nan.
"""
function check_Inf_Nan( B :: Hess_matrix{Y}) where Y <: Number
    res = []
    for i in 1:length(B.arr)
        interet = check_Inf_Nan(B.arr[i].elmt_hess)
        if interet != nothing
            push!(res, (i,interet))
        end
    end
    return res
end

"""
    check_Inf_Nan(B)
function that check if an array contains a Nan.
"""
function check_Inf_Nan(Bi :: Array{Y,2}) where Y <: Number
    for i in Bi
        if isnan(i) || isinf(i)
            println("oui")
            return Bi
        end
    end
end


function empty_Bᵢxᵢ(sps :: SPS{T, Y} ) where T where Y <: Number
    Bx = Vector{Y}(zeros(Y, sps.n_var))
    @inbounds f = ( elt_fun :: element_function -> view(Bx, get_used_variable(elt_fun)))
    Bᵢxᵢ = map(f, get_structure(sps))
    return (Bx, Bᵢxᵢ)
end


function test_parcours(sps :: SPS{T, Y}, B :: Hess_matrix{Y} ) where T where Y <: Number
    structure = get_structure(sps)
    for i in 1:length(structure)
        B.arr[i].elmt_hess :: Array{Y,2} * sps.x_views[sps.structure[i].index_fun][sps.structure[i].index_element_tree_position] :: SubArray{Y,1,Array{Y,1},Tuple{Array{Int64,1}},false}
    end
end


#= Code non utilisé =#


hess_full(sps :: SPS{T}, x :: AbstractVector{Y}) where T where Y <: Number = construct_full_Hessian(hess(sps,x))

function hess(sps :: SPS{T}, x :: AbstractVector{Y}) where T where Y <: Number
    define_element_hessian = ( elm_fun :: element_function -> element_hessian{Y}( zeros(Y, length(elm_fun.used_variable), length(elm_fun.used_variable) )) )
    hess_matrix = Hess_matrix{Y}(define_element_hessian.(sps.structure))
    hess!( hess_matrix, sps, x )
    return hess_matrix
end

function hessian_tapes(elmt_tree :: T ) where T
    f = CalculusTreeTools.evaluate_expr_tree(elmt_tree)
    n = length(CalculusTreeTools.get_elemental_variable(elmt_tree))
    f_tape = ReverseDiff.HessianTape(f, rand(n) )
end

function hess!(partitionned_hess :: Hess_matrix{Y}, sps :: SPS{T}, x :: AbstractVector{Y}) where T where Y <: Number
    structure = get_structure(sps)
    different_calculus_tree = get_different_element_tree(sps)
    compiled_hessians = map(x -> hessian_tapes(x), different_calculus_tree)
    @show length(different_calculus_tree), length(compiled_hessians)
    for i in 1:length(structure)
        elmt_fun = structure[i]
        index_fun = get_index_fun(elmt_fun)
        related_x = get_var_from_elmt_fun(elmt_fun, sps)
        @show index_fun, related_x
        elemental_hess!(partitionned_hess.arr[i], compiled_hessians[index_fun], related_x)
    end
end

function elemental_hess!(h :: element_hessian{Y}, compiled_hessian :: ReverseDiff.HessianTape, x :: AbstractVector{Y}) where Y <: Number
    @show typeof(compiled_hessian)
    ReverseDiff.hessian!(h.elmt_hess, compiled_hessian, x)
end

#= Fin Code non utilisé =#


#=
Ancien code
=#

"""
    evaluate_SPS_gradient2!(sps,x,g)
Compute the gradient of the partially separable structure sps, and store the result in the grad_vector structure g.
Using ForwardDiff package. Bad behaviour with Threads.@threads.
This was the previous version using ForwardDiff. The actual version using ReverseDiff is more efficient.
"""
function evaluate_SPS_gradient2!(sps :: SPS{T}, x :: AbstractVector{Y}, g :: grad_vector{Y} ) where T where Y <: Number
    l_elmt_fun = length(sps.structure)
    for i in 1:l_elmt_fun
        if isempty(sps.structure[i].used_variable) == false  #fonction element ayant au moins une variable
            element_gradient2!(get_fun_from_elmt_fun(sps.structure[i],sps), view(x, sps.structure[i].used_variable), g.arr[i] )
        end
    end
end

"""
    element_gradient2!(expr_tree, x, g)
Compute the element grandient of the function represents by expr_tree according to the vector x, and store the result in the vector g.
This was the previous version using ForwardDiff. The actual version using ReverseDiff is more efficient.
"""
function element_gradient2!(expr_tree :: Y, x :: AbstractVector{T}, g :: element_gradient{T} ) where T <: Number where Y
    ForwardDiff.gradient!(g.g_i, CalculusTreeTools.evaluate_expr_tree(expr_tree), x  )
end




"""
    evaluate_SPS(sps,x)
evalutate the partially separable function f = ∑fᵢ, stored in the sps structure at the point x.
f(x) = ∑fᵢ(xᵢ), so we compute independently each fᵢ(xᵢ) and we return the sum.
"""
# function evaluate_SPS(sps :: SPS{T}, x :: AbstractVector{Y} ) where T where Y <: Number
#     # on utilise un mapreduce de manière à ne pas allouer un tableau temporaire, on utilise l'opérateur + pour le reduce car cela correspond
#     # à la définition des fonctions partiellement séparable.
#     # @inbounds @fastmath mapreduce(elmt_fun :: element_function{T} -> CalculusTreeTools.evaluate_expr_tree(elmt_fun.fun, view(x, elmt_fun.used_variable)) :: Y, + , sps.structure :: Vector{element_function{T}})
#     @inbounds @fastmath mapreduce(elmt_fun :: element_function -> CalculusTreeTools.evaluate_expr_tree(get_fun_from_elmt_fun(elmt_fun,sps), view(x, elmt_fun.used_variable)) :: Y, + , sps.structure :: Vector{element_function})
#     #les solutions à base de boucle for sont plus lente même avec @Thread.thread
# end


function evaluate_SPS2(sps :: SPS{T,Y}, x :: AbstractVector{Y} ) where T where Y <: Number
    sps.x .= x
    related_vars = get_related_vars(sps)
    diff_calculus_tree = get_different_element_tree(sps)
    x_views = get_x_views(sps)
    (length(diff_calculus_tree) == length(related_vars) && length(related_vars) == length(x_views))|| error("mismatch evaluate SPS")
    nb_elem_fun = length(diff_calculus_tree)
    res = Vector{Y}(undef, nb_elem_fun)
    for i in 1:nb_elem_fun
        vars = related_vars[i]
        calculus_tree = diff_calculus_tree[i] :: T
        selected_vars = x_views[i]
        res[i] = sum(CalculusTreeTools.evaluate_expr_tree_multiple_points(calculus_tree :: T , selected_vars))
    end
    return sum(res)
end




#=
Previous deduct_partially_separable_structure
=#

# deduct_partially_separable_structure(a :: Any, n :: Int) = _deduct_partially_separable_structure(a, trait_expr_tree.is_expr_tree(a), n)
# _deduct_partially_separable_structure(a, :: trait_expr_tree.type_not_expr_tree, n :: Int) = error("l'entrée de la fonction n'est pas un arbre d'expression")
# _deduct_partially_separable_structure(a, :: trait_expr_tree.type_expr_tree, n :: Int) = _deduct_partially_separable_structure(a, n )

# function _deduct_partially_separable_structure(expr_tree :: T , n :: Int, type=Float64 :: DataType) where T
#     work_expr_tree = copy(expr_tree)
#     if isa(expr_tree, Expr) == false
#         CalculusTreeTools.cast_type_of_constant(work_expr_tree, type)
#     end
#
#     elmt_fun = CalculusTreeTools.delete_imbricated_plus(work_expr_tree)
#     m_i = length(elmt_fun)
#
#     type_i = Vector{CalculusTreeTools.type_calculus_tree}(undef, m_i)
#     for i in 1:m_i
#         type_i[i] = CalculusTreeTools.get_type_tree(elmt_fun[i])
#     end
#
#     elmt_var_i =  Vector{ Vector{Int}}(undef,m_i)
#     length_vec = Threads.Atomic{Int}(0)
#     for i in 1:m_i
#         elmt_var_i[i] = CalculusTreeTools.get_elemental_variable(elmt_fun[i])
#         atomic_add!(length_vec, length(elmt_var_i[i]))
#     end
#     sort!.(elmt_var_i) #ligne importante, met dans l'ordre les variables élémentaires. Utile pour les U_i et le N_to_Ni
#
#     for i in 1:m_i
#         CalculusTreeTools.element_fun_from_N_to_Ni!(elmt_fun[i],elmt_var_i[i])
#     end
#
#     structure = Vector{element_function}(undef,m_i)
#     convexity_wrapper = CalculusTreeTools.convexity_wrapper(CalculusTreeTools.unknown_type())
#
#
#     (different_calculus_tree, different_calculus_tree_index) = get_different_CalculusTree(elmt_fun)
#
#     for i in 1:m_i
#         undefined_index = 0
#         structure[i] = element_function(different_calculus_tree_index[i], undefined_index, type_i[i], elmt_var_i[i], convexity_wrapper, i)
#     end
#     construct_set_index_element_view!(structure, different_calculus_tree)
#
#     index_element_tree = get_related_function(structure, different_calculus_tree)
#
#     related_vars = get_related_var(structure, index_element_tree)
#
#     compiled_gradients = map(x -> compiled_grad_of_elmt_fun(x), different_calculus_tree)
#
#     x = Vector{type}(undef,n)
#     x_views = construct_views(x, related_vars)
#
#     v = Vector{type}(undef,n)
#     v_views = construct_views(v, related_vars)
#
#
#     return SPS{T, type}(structure, different_calculus_tree, index_element_tree, related_vars, x, x_views, v, v_views, length_vec[], n, compiled_gradients)
# end




# function _deduct_partially_separable_structure(expr_tree :: CalculusTreeTools.complete_expr_tree, n :: Int, type=Float64 :: DataType)
#     work_expr_tree = copy(expr_tree)
#     CalculusTreeTools.cast_type_of_constant(work_expr_tree, type)
#
#     # elmt_fun = CalculusTreeTools.delete_imbricated_plus(work_expr_tree) :: Vector{T}
#     elmt_fun = CalculusTreeTools.delete_imbricated_plus(work_expr_tree)
#     m_i = length(elmt_fun)
#
#
#     # type_i = Vector{implementation_type_expr.t_type_expr_basic}(undef, m_i)
#     type_i = Vector{CalculusTreeTools.type_calculus_tree}(undef, m_i)
#     for i in 1:m_i
#         type_i[i] = CalculusTreeTools.get_type_tree(elmt_fun[i])
#     end
#
#     elmt_var_i =  Vector{ Vector{Int}}(undef,m_i)
#     length_vec = 0
#     for i in 1:m_i
#         elmt_var_i[i] = CalculusTreeTools.get_elemental_variable(elmt_fun[i])
#         length_vec += length(elmt_var_i[i])
#     end
#     sort!.(elmt_var_i) #ligne importante, met dans l'ordre les variables élémentaires. Utile pour les U_i et le N_to_Ni
#
#     for i in 1:m_i
#         CalculusTreeTools.element_fun_from_N_to_Ni!(elmt_fun[i],elmt_var_i[i])
#     end
#
#     (different_calculus_tree, different_calculus_tree_index) = get_different_CalculusTree(elmt_fun)
#     CalculusTreeTools.set_bounds!.(different_calculus_tree)
#     CalculusTreeTools.set_convexity!.(different_calculus_tree)
#     convexity_wrapper = map( (x -> CalculusTreeTools.convexity_wrapper(CalculusTreeTools.get_convexity_status(x)) ), different_calculus_tree)
#
#     structure = Vector{element_function}(undef,m_i)
#     for i in 1:m_i
#         undefined_index = 0
#         structure[i] = element_function(different_calculus_tree_index[i], undefined_index, type_i[i], elmt_var_i[i], convexity_wrapper[different_calculus_tree_index[i]], i)
#     end
#     construct_set_index_element_view!(structure, different_calculus_tree)
#
#     index_element_tree = get_related_function(structure, different_calculus_tree)
#
#     related_vars = get_related_var(structure, index_element_tree)
#
#     compiled_gradients = map(x -> compiled_grad_of_elmt_fun(x), different_calculus_tree)
#
#     x = Vector{type}(undef,n)
#     x_views = construct_views(x, related_vars)
#
#     v = Vector{type}(undef,n)
#     v_views = construct_views(v, related_vars)
#
#     return SPS{CalculusTreeTools.complete_expr_tree, type}(structure, different_calculus_tree, index_element_tree, related_vars, x, x_views, v, v_views, length_vec[], n, compiled_gradients)
# end
