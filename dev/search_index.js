var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [PartiallySeparableNLPModels,Mod_ab_partitioned_data, Mod_PBFGS, Mod_PLBFGS, Mod_common]","category":"page"},{"location":"reference/#PartiallySeparableNLPModels.Hv-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T, Y}, AbstractVector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.Hv","text":"Hv(sps, x, v)\n\nCompute the product hessian vector of the hessian at the point x dot the vector v. This version both ReverseDiff and ForwardDiff.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.Hv2-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T, Y}, AbstractVector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.Hv2","text":"Hv2(sps, x, v)\n\nCompute the product hessian vector of the hessian at the point x dot the vector v. This version uses only ReverseDiff.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.Hv_only_product-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T}, Hess_matrix{Y}, AbstractVector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.Hv_only_product","text":"Hv_only_product(sps, Hess_matrix, x)\n\nreturn the product dot(Hessmatrix, x), the function fits with the partitionned representation of the matrix Hessmatrix. In order to do the right operations in the right order we need the partially separable structure sps. Less efficient than productmatrixsps.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.build_gradient-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T}, grad_vector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.build_gradient","text":"build_gradient(sps, g)\n\nConstructs a vector of size n from the list of element gradient of the sps structure which has numerous element gradient of size nᵢ. The purpose of the function is to gather these element gradient into a real gradient of size n. The function gradnito_n! will add element gradient of size nᵢ at the right inside the gradient of size n.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.check_Inf_Nan-Union{Tuple{Hess_matrix{Y}}, Tuple{Y}} where Y<:Number","page":"Reference","title":"PartiallySeparableNLPModels.check_Inf_Nan","text":"check_Inf_Nan(B)\n\nfunction that check if an Hess_matrix is not full of Nan.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.check_Inf_Nan-Union{Tuple{Matrix{Y}}, Tuple{Y}} where Y<:Number","page":"Reference","title":"PartiallySeparableNLPModels.check_Inf_Nan","text":"check_Inf_Nan(B)\n\nfunction that check if an array contains a Nan.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.compiled_grad_of_elmt_fun-Tuple{T} where T","page":"Reference","title":"PartiallySeparableNLPModels.compiled_grad_of_elmt_fun","text":"compiled_grad_of_elmt_fun(elmt_fun)\n\nReturn  the GradientTape compiled to speed up the ReverseDiff computation of the elmt_fun gradient in the future\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.construct_Sparse_Hessian-Union{Tuple{Y}, Tuple{T}, Tuple{SPS{T}, Hess_matrix{Y}}} where {T, Y<:Number}","page":"Reference","title":"PartiallySeparableNLPModels.construct_Sparse_Hessian","text":"construct_Sparse_Hessian(sps, B)\n\nBuild from the Partially separable Structure sps and the Hessian approximation B a SpaseArray which represent B in other form.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.construct_set_index_element_view!-Union{Tuple{T}, Tuple{Vector{element_function}, Vector{T}}} where T","page":"Reference","title":"PartiallySeparableNLPModels.construct_set_index_element_view!","text":"construct_set_index_element_view(elmt_fun_vector, diff_elmt_tree)\n\nFor each elmtfun ∈ elmtfunvector we set the attribute indexelementtreeposition. For doing this we use the function getelmntfunindexview(elmtfunvector, diffelmttree) that builds a vector of size length(elmtfunvector) with the right indexes.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.construct_views-Union{Tuple{N}, Tuple{Vector{N}, Vector{Vector{Vector{Int64}}}}} where N<:Number","page":"Reference","title":"PartiallySeparableNLPModels.construct_views","text":"construct_views(x, related_vars)\n\nrelated_vars a vector (from different element tree) of vector (several element function linked with each tree)\n\nof vector{int} (a set of index, which represent the elemental variable).\n\nx a point ∈ Rⁿ\n\nreturns a view of x for each set of index representing the elemental variable.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.create_pre_compiled_tree-Union{Tuple{N}, Tuple{Vector{CalculusTreeTools.implementation_tree.type_node{CalculusTreeTools.abstract_expr_node.ab_ex_nd}}, Array{Array{SubArray{N, 1, Vector{N}, Tuple{Vector{Int64}}, false}, 1}, 1}}} where N<:Number","page":"Reference","title":"PartiallySeparableNLPModels.create_pre_compiled_tree","text":"create_pre_compiled_tree(vector_calculus_trees, vector_x_views)\n\ncreate a precompiled tree for a particuliar number of evaluation. The size of vectorcalculustrees and vectorxviews must match. Each calculustree from vectorcalculustrees is pre-compiled for nᵢ evaluation, nᵢ the size of the corresponding element of vectorx_views\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.deduct_partially_separable_structure","page":"Reference","title":"PartiallySeparableNLPModels.deduct_partially_separable_structure","text":"deduct_partially_separable_structure(expr_tree, n)\n\nFind the partially separable structure of a function f stored as an expression tree expr_tree. To define properly the size of sparse matrix we need the size of the problem : n. At the end, we get the partially separable structure of f, f(x) = ∑fᵢ(xᵢ)\n\n\n\n\n\n","category":"function"},{"location":"reference/#PartiallySeparableNLPModels.element_gradient!-Union{Tuple{T}, Tuple{ReverseDiff.CompiledTape, AbstractVector{T}, element_gradient{T}}} where T<:Number","page":"Reference","title":"PartiallySeparableNLPModels.element_gradient!","text":"element_gradient!(compil_tape, x, g)\n\nCompute the element grandient from the compil_tape compiled before according to the vector x, and store the result in the vector g Use of ReverseDiff\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.element_gradient2!-Union{Tuple{T}, Tuple{Y}, Tuple{Y, AbstractVector{T}, element_gradient{T}}} where {Y, T<:Number}","page":"Reference","title":"PartiallySeparableNLPModels.element_gradient2!","text":"element_gradient2!(expr_tree, x, g)\n\nCompute the element grandient of the function represents by expr_tree according to the vector x, and store the result in the vector g. This was the previous version using ForwardDiff. The actual version using ReverseDiff is more efficient.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.eval_ni_ones-Union{Tuple{T}, Tuple{T, Int64}} where T","page":"Reference","title":"PartiallySeparableNLPModels.eval_ni_ones","text":"get_different_CalculusTree( all_element_tree, element_vars)\n\nSelects the distinct tree from the allelementtree to create a Vector of distinct element tree: differentcalculustree. In addition to that create a Vector of index to linked the tree from allelementtree to differentcalculustree : differentcalculustreeindex. elementvars is used to count the numbers of variables used in a tree, if the number of variables is different we don't test the tree and avoid to test the equality between trees. length(differentcalculustreeindex) == length(allelementtree) == length(elementvars) return (differentcalculustree, differentcalculustree_index)\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.evaluate_SPS-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T, Y}, AbstractVector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.evaluate_SPS","text":"evaluate_SPS(sps,x)\n\nEvaluate the structure sps on the point x ∈ Rⁿ. Since we work on subarray of x, we allocate x to the structure sps in a first time. Once this step is done we select the calculus tree needed as well as the view of x needed. Then we evaluate the different calculus tree on the needed point (since the same function appear a lot of time). At the end we sum the result\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.evaluate_SPS_gradient!-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T}, AbstractVector{Y}, grad_vector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.evaluate_SPS_gradient!","text":"evaluate_SPS_gradient!(sps,x,g)\n\nCompute the gradient of the partially separable structure sps, and store the result in the gradvector structure g. Using ReversDiff package. Not obvious good behaviour with Threads.@threads, sometime yes sometime no. Noted that we use the previously compiled GradientTape in elementgradient! that use ReverseDiff.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.evaluate_SPS_gradient-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T, Y}, AbstractVector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.evaluate_SPS_gradient","text":"evaluate_SPS_gradient(sps,x)\n\nReturn the gradient of the partially separable structure sps at the point x. Using ReversDiff package. Not obvious good behaviour with Threads.@threads, sometime yes sometime no. Noted that we use the previously compiled GradientTape in element_gradient! that use ReverseDiff.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.evaluate_SPS_gradient2!-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T}, AbstractVector{Y}, grad_vector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.evaluate_SPS_gradient2!","text":"evaluate_SPS_gradient2!(sps,x,g)\n\nCompute the gradient of the partially separable structure sps, and store the result in the grad_vector structure g. Using ForwardDiff package. Bad behaviour with Threads.@threads. This was the previous version using ForwardDiff. The actual version using ReverseDiff is more efficient.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.evaluate_element_hessian-Union{Tuple{T}, Tuple{Y}, Tuple{element_function, AbstractVector{Y}, SPS{T}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.evaluate_element_hessian","text":"evaluate_element_hessian(fᵢ,xᵢ)\n\nCompute the Hessian of the elemental function fᵢ : Gᵢ a nᵢ × nᵢ matrix. So xᵢ a vector of size nᵢ. The result of the function is the triplet of the sparse matrix Gᵢ.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.evaluate_gradient-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T}, Vector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.evaluate_gradient","text":"evaluate_gradient(sps,x)\n\nevalutate the gradient of the partially separable function f = ∑ fι, stored in the sps structure at the point x, return a vector of size n (the number of variable) which is the gradient. Première version de la fonction inutile car inefficace.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.evaluate_hessian-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T}, AbstractVector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.evaluate_hessian","text":"evaluate_hessian(sps,x)\n\nevalutate the hessian of the partially separable function f = ∑ fᵢ, stored in the sps structure at the point x. Return the sparse matrix of the hessian of size n × n.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.evaluate_obj_pre_compiled-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T, Y}, AbstractVector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.evaluate_obj_pre_compiled","text":"evaluate_obj_pre_compiled(sps, point_x)\n\nEvaluate the structure sps at pointx using the precompiled element tree for nᵢ evaluation. This function is the current evaluateSPS.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.get_elmnt_fun_index_view-Union{Tuple{T}, Tuple{Vector{element_function}, Vector{T}}} where T","page":"Reference","title":"PartiallySeparableNLPModels.get_elmnt_fun_index_view","text":"get_elmnt_fun_index_view(vector_element_fun, different_element_tree)\n\nFor each elementfun ∈ vectorelementfun, returns the position of the linked element tree corresponding in differentelement_tree. It will be use later for retrieve a view of the variable used by each element function.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.get_index_deleted-Union{Tuple{T}, Tuple{Vector{T}, T, Int64, Float64, Vector{Int64}, Vector{Float64}}} where T","page":"Reference","title":"PartiallySeparableNLPModels.get_index_deleted","text":"get_index_deleted(tree_vector, tree, numbervar, vector_numbervar)\n\nretrieve the indexes of treevector such that for all i ∈ indexes treevector[i] == tree. This function used numbervar and vector_numbervar to speed up the comparison by avoiding equality between trees.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.get_related_function-Union{Tuple{T}, Tuple{Vector{element_function}, Vector{T}}} where T","page":"Reference","title":"PartiallySeparableNLPModels.get_related_function","text":"get_related_function(elmt_functionS, calculusTreeS)\n\nCette fonction récupère les indices des fonctions éléments elmt_functionS lié aux quelques arbres de calculs calculusTreeS de la structure\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.get_related_var-Tuple{Vector{element_function}, Vector{Vector{Int64}}}","page":"Reference","title":"PartiallySeparableNLPModels.get_related_var","text":"get_related_var(vec_elmt_fun, index_elmt_tree)\n\nRenvoie un tableau d'entier de 3 dimensions. La première dimension de taille length(indexelmttree) correspond au nombre de fonctions éléments distinctes (arbre de calcul). Pour chaque fonction éléments distinctes e nous avons un vecteur vₑ :: Vector{Vector{Int}} tel que length(vₑ) == length(indexelmttree[e]) indexelmttree[e] contient les indices des fonctions éléments rattachés à la fonction élément e. Le résultat de la fonction renverra pour chaque indice de fonctions éléments les variables utilisées ( :: Vector{Int}) par celle-ci ce qui nous donne un tableau 3 dimensions.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.grad_ni_to_n!-Union{Tuple{Y}, Tuple{element_gradient{Y}, Vector{Int64}, AbstractVector{Y}}} where Y<:Number","page":"Reference","title":"PartiallySeparableNLPModels.grad_ni_to_n!","text":"grad_ni_to_n!(element_gradient, used_var, gradient)\n\nAdd to the gradient the value of elementgradient according to the vector of usedvar given. length(usedvar) == length(elementgradient)\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.id_hessian!-Union{Tuple{Y}, Tuple{T}, Tuple{SPS{T}, Hess_matrix{Y}}} where {T, Y<:Number}","page":"Reference","title":"PartiallySeparableNLPModels.id_hessian!","text":"id_hessian!(sps, B)\n\nConstruct a kinf of Id Hessian, it will initialize each element Hessian Bᵢ with an Id matrix, B =  ∑ᵢᵐ Uᵢᵀ Bᵢ Uᵢ\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.minus_grad_vec!-Union{Tuple{T}, Tuple{grad_vector{T}, grad_vector{T}, grad_vector{T}}} where T<:Number","page":"Reference","title":"PartiallySeparableNLPModels.minus_grad_vec!","text":"minus_grad_vec!(g1,g2,res)\n\nStore in res: g1 minus g2, but g1 and g2 have a particular structure which is grad_vector{T}. We need this operation to have the difference for each element gradient for TR method. g1 = gₖ and g2 = gₖ₋₁.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.product_matrix_sps!-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T}, Hess_matrix{Y}, AbstractVector{Y}, AbstractVector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.product_matrix_sps!","text":"product_matrix_sps!(sps,B,x,Bx)\n\nThis function make the product of the structure B which represents a symetric matrix and the vector x, the result is stored in Bx. We need the structure sps for the variable used in each B[i], to replace B[i]*x[i] in the result vector by using f_inter!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.product_matrix_sps-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T}, Hess_matrix{Y}, Vector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.product_matrix_sps","text":"product_matrix_sps(sps,B,x)\n\nThis function make the product of the structure B which represents a symetric matrix and the vector x. We need the structure sps for the variable used in each B[i], to replace B[i]*x[i] in the result vector.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.product_vector_sps-Union{Tuple{T}, Tuple{Y}, Tuple{Z}, Tuple{SPS{T}, grad_vector{Y}, Vector{Z}}} where {Z<:Number, Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.product_vector_sps","text":"product_vector_sps(sps, g, x)\n\ncompute the product g⊤ x = ∑ Uᵢ⊤ gᵢ⊤ xᵢ. So we need the sps structure to get the Uᵢ. On ne s'en sert pas en pratique mais peut-être pratique pour faire des vérifications\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.struct_hessian!-Union{Tuple{Y}, Tuple{T}, Tuple{SPS{T}, AbstractVector{Y}, Hess_matrix{Y}}} where {T, Y<:Number}","page":"Reference","title":"PartiallySeparableNLPModels.struct_hessian!","text":"struct_hessian!(sps,x,H)\n\nEvalutate the hessian of the partially separable function, stored in the sps structure at the point x. Store the Hessian in a particular structure H :: Hess_matrix.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.struct_hessian-Union{Tuple{T}, Tuple{Y}, Tuple{SPS{T}, AbstractVector{Y}}} where {Y<:Number, T}","page":"Reference","title":"PartiallySeparableNLPModels.struct_hessian","text":"struct_hessian(sps,x)\n\nevalutate the hessian of the partially separable function, stored in the sps structure at the point x. Return the Hessian in a particular structure : Hess_matrix.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.Mod_ab_partitioned_data.evaluate_grad_part_data-Union{Tuple{Y}, Tuple{T}, Tuple{T, Vector{Y}}} where {T<:PartiallySeparableNLPModels.Mod_ab_partitioned_data.PartitionedData, Y<:Number}","page":"Reference","title":"PartiallySeparableNLPModels.Mod_ab_partitioned_data.evaluate_grad_part_data","text":"\tevaluate_grad_part_data(part_data,x)\n\nBuild the gradient vector at the point x from the element gradient computed and stored in part_data.pg .\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.Mod_ab_partitioned_data.evaluate_y_part_data!-Union{Tuple{Y}, Tuple{T}, Tuple{T, Vector{Y}, Vector{Y}}} where {T<:PartiallySeparableNLPModels.Mod_ab_partitioned_data.PartitionedData, Y<:Number}","page":"Reference","title":"PartiallySeparableNLPModels.Mod_ab_partitioned_data.evaluate_y_part_data!","text":"\tevaluate_y_part_data!(part_data,x,s)\n\nCompute the element gradients differences such as ∇̂fᵢ(x+s)-∇̂fᵢ(x) for each element functions.  It stores the results in part_data.pv\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.Mod_PBFGS.build_PartitionedData_TR_PBFGS-Union{Tuple{T}, Tuple{G}, Tuple{G, Int64}} where {G, T}","page":"Reference","title":"PartiallySeparableNLPModels.Mod_PBFGS.build_PartitionedData_TR_PBFGS","text":"build_PartitionedData_TR_PBFGS(expr_tree, n)\n\nFind the partially separable structure of a function f stored as an expression tree expr_tree. To define properly the size of sparse matrix we need the size of the problem : n. At the end, we get the partially separable structure of f, f(x) = ∑fᵢ(xᵢ)\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.Mod_PBFGS.update_PBFGS!-Union{Tuple{T}, Tuple{G}, Tuple{PartitionedData_TR_PBFGS{G, T}, Vector{T}}} where {G, T}","page":"Reference","title":"PartiallySeparableNLPModels.Mod_PBFGS.update_PBFGS!","text":"\tupdate_PBFGS(pd_pbfgs,s)\n\nPerform the PBFGS update givent the current iterate x and the next iterate s\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.Mod_PBFGS.update_PBFGS-Union{Tuple{T}, Tuple{G}, Tuple{PartitionedData_TR_PBFGS{G, T}, Vector{T}, Vector{T}}} where {G, T}","page":"Reference","title":"PartiallySeparableNLPModels.Mod_PBFGS.update_PBFGS","text":"\tupdate_PBFGS(pd_pbfgs,x,s)\n\nPerform the PBFGS update givent the two iterate x and s\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.Mod_PLBFGS.build_PartitionedData_TR_PLBFGS-Union{Tuple{T}, Tuple{G}, Tuple{G, Int64}} where {G, T}","page":"Reference","title":"PartiallySeparableNLPModels.Mod_PLBFGS.build_PartitionedData_TR_PLBFGS","text":"build_PartitionedData_TR_PLBFGS(expr_tree, n)\n\nFind the partially separable structure of a function f stored as an expression tree expr_tree. To define properly the size of sparse matrix we need the size of the problem : n. At the end, we get the partially separable structure of f, f(x) = ∑fᵢ(xᵢ)\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.Mod_PLBFGS.update_PLBFGS!-Union{Tuple{T}, Tuple{G}, Tuple{PartitionedData_TR_PLBFGS{G, T}, Vector{T}}} where {G, T}","page":"Reference","title":"PartiallySeparableNLPModels.Mod_PLBFGS.update_PLBFGS!","text":"\tupdate_PLBFGS(pd_pblfgs,s)\n\nPerform the PBFGS update givent the current iterate x and the next iterate s\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.Mod_PLBFGS.update_PLBFGS-Union{Tuple{T}, Tuple{G}, Tuple{PartitionedData_TR_PLBFGS{G, T}, Vector{T}, Vector{T}}} where {G, T}","page":"Reference","title":"PartiallySeparableNLPModels.Mod_PLBFGS.update_PLBFGS","text":"\tupdate_PLBFGS(pd_pblfgs,x,s)\n\nPerform the PBFGS update givent the two iterate x and s\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.Mod_common.compiled_grad_elmt_fun-Tuple{T} where T","page":"Reference","title":"PartiallySeparableNLPModels.Mod_common.compiled_grad_elmt_fun","text":"compiledgradelmtfun(elmtfun, ni) Return  the GradientTape compiled to speed up the ReverseDiff computation of the elmt_fun gradient in the future\n\n\n\n\n\n","category":"method"},{"location":"reference/#PartiallySeparableNLPModels.Mod_common.distinct_element_expr_tree-Union{Tuple{T}, Tuple{Vector{T}, Vector{Vector{Int64}}}} where T","page":"Reference","title":"PartiallySeparableNLPModels.Mod_common.distinct_element_expr_tree","text":"distinct_element_expr_tree(vec_element_expr_tree, vec_element_variables; N)\n\nFilter the vector vecelementexprtree to obtain only the element functions that are distincts as elementexprtree. length(elementexprtree) == M. In addition it returns indexelement_tree, who records the index 1 <= i <= M of each element function\n\n\n\n\n\n","category":"method"},{"location":"#PartiallySeparableNLPModels.jl","page":"Home","title":"PartiallySeparableNLPModels.jl","text":"","category":"section"},{"location":"tutorial/#PartiallySeparableNLPModels.jl-Tutorial","page":"Tutorial","title":"PartiallySeparableNLPModels.jl Tutorial","text":"","category":"section"}]
}
