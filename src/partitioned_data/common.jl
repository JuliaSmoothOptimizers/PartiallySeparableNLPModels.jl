module Mod_common 
	using ReverseDiff, LinearAlgebra, SparseArrays
	using CalculusTreeTools, PartitionedStructures

	export Element_function
	export distinct_element_expr_tree, compiled_grad_elmt_fun
	# export create_eev, create_id_eem, identity_epm, identity_eplom_lbfgs

	mutable struct Element_function
	  i :: Int # the index of the function 1 ≤ i ≤ N
	  index_element_tree :: Int # 1 ≤ index_element_tree ≤ M
	  variable_indices :: Vector{Int} # ≈ Uᵢᴱ
	  type :: CalculusTreeTools.type_calculus_tree
	  convexity_status :: CalculusTreeTools.convexity_wrapper
	end

	"""
	    distinct_element_expr_tree(vec_element_expr_tree, vec_element_variables; N)
	Filter the vector vec_element_expr_tree to obtain only the element functions that are distincts as element_expr_tree.
	length(element_expr_tree) == M.
	In addition it returns index_element_tree, who records the index 1 <= i <= M of each element function
	"""
	function distinct_element_expr_tree(vec_element_expr_tree :: Vector{T}, vec_element_variables :: Vector{Vector{Int}}; N::Int=length(vec_element_expr_tree)) where T
	  N == length(vec_element_variables) || @error("The sizes vec_element_expr_tree and vec_element_variables are differents")
	  index_element_tree = (xi -> -xi).(ones(Int,N))
	  element_expr_tree = Vector{T}(undef,0)
	  vec_val_elt_fun_ones = map( (elt_fun,elt_vars) -> CalculusTreeTools.evaluate_expr_tree(elt_fun, ones(length(elt_vars))), vec_element_expr_tree, vec_element_variables) # evaluate as first equality test
	  working_array = map( (val_elt_fun_ones,i) -> (val_elt_fun_ones,i), vec_val_elt_fun_ones, 1:N)
	  current_expr_tree_index = 1
	  while isempty(working_array) == false
	    val = working_array[1][1]
	    comparator_value_elt_fun(val_elt_fun) = val_elt_fun[1] == val
	    current_indices_similar_element_functions = findall(comparator_value_elt_fun, working_array[:,1])
	    real_indices_similar_element_functions = (tup -> tup[2]).(working_array[current_indices_similar_element_functions])
	    current_expr_tree = vec_element_expr_tree[working_array[1][2]]
	    push!(element_expr_tree, current_expr_tree) 
	    comparator_elt_expr_tree(expr_tree) = expr_tree == current_expr_tree
	    current_indices_equal_element_function = findall(comparator_elt_expr_tree, vec_element_expr_tree[real_indices_similar_element_functions])
			real_indices_equal_element_function = (tup -> tup[2]).(working_array[current_indices_equal_element_function])
	    deleteat!(working_array, current_indices_equal_element_function)
	    index_element_tree[real_indices_equal_element_function] .= current_expr_tree_index
	    current_expr_tree_index += 1
	  end
	  minimum(index_element_tree) == -1 && @error("Not every element function is attributed")
	  return element_expr_tree, index_element_tree
	end

	"""
	compiled_grad_elmt_fun(elmt_fun, ni)
	Return  the GradientTape compiled to speed up the ReverseDiff computation of the elmt_fun gradient in the future
	"""
	function compiled_grad_elmt_fun(elmt_fun :: T; ni::Int=length(CalculusTreeTools.get_elemental_variable(elmt_fun)), type=Float64) where T
	  f = CalculusTreeTools.evaluate_expr_tree(elmt_fun)
	  f_tape = ReverseDiff.GradientTape(f, rand(type,ni))
	  compiled_f_tape = ReverseDiff.compile(f_tape)
	  return compiled_f_tape
	end

	# function create_eev(elt_var::Vector{Int}; type=Float64)
	  # nie = length(elt_var)
	  # eev_value = rand(nie)
	  # eev = PartitionedStructures.Elemental_elt_vec{type}(eev_value, elt_var, nie)
	  # return eev
	# end 
	
	# function create_epv(vec_elt_var::Vector{Vector{Int}})
	  # eev_set = map((elt_var -> create_eev(elt_var,type=type)), vec_elt_var)  
	  # pg = create_epv(eev_set; n=n)
	# end 

	# function create_id_eem(elt_var::Vector{Int}; type=Float64)
	  # nie = length(elt_var)
	  # Bie = zeros(type,nie,nie)
	  # [Bie[i,i]=1 for i in 1:nie]  
	  # eem = PartitionedStructures.Elemental_em{type}(nie, elt_var, Symmetric(Bie))
	  # return eem
	# end

	# function identity_epm(element_variables::Vector{Vector{Int}}, N :: Int, n ::Int; type=Float64)
	  # eem_set = map( (elt_var -> create_id_eem(elt_var;type=type)), element_variables)
	  # spm = spzeros(type,n,n)
	  # L = spzeros(type,n,n)
	  # component_list = map(i -> Vector{Int}(undef,0), [1:n;])
	  # no_perm = [1:n;]
	  # epm = PartitionedStructures.Elemental_pm{type}(N,n,eem_set,spm,L,component_list,no_perm)
	  # PartitionedStructures.initialize_component_list!(epm)
	  # return epm
	# end 
	
	# function init_eelom(elt_var::Vector{Int}; type=Float64)
		# nie = length(elt_var)
		# Bie = LinearOperators.LBFGSOperator(type, nie)
		# elom = Elemental_elom_bfgs{type}(nie, elt_var, Bie)
		# return elom
	# end 
	
	# function identity_eplom_lbfgs(element_variables::Vector{Vector{Int}}, N :: Int, n ::Int; type=Float64)		
	  # eelom_set = map( (elt_var -> init_eelom(elt_var;type=type)), element_variables)
	  # spm = spzeros(type,n,n)
	  # L = spzeros(type,n,n)
	  # component_list = map(i -> Vector{Int}(undef,0), [1:n;])
	  # no_perm = [1:n;]
	  # eplom = PartitionedStructures.Elemental_plom_bfgs{type}(N,n,eelom_set,spm,L,component_list,no_perm)
	  # PartitionedStructures.initialize_component_list!(eplom)
	  # return epm
	# end 

end 