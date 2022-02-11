module Mod_PBFGS
	using ReverseDiff
	using PartitionedStructures, CalculusTreeTools
	using ..Mod_ab_partitioned_data, ..Mod_common 

	export PartitionedData_TR_PBFGS
	export update_PBFGS, update_PBFGS!, build_PartitionedData_TR_PBFGS

	# mutable struct Element_function
	  # i :: Int # the index of the function 1 ≤ i ≤ N
	  # index_element_tree :: Int # 1 ≤ index_element_tree ≤ M
	  # variable_indices :: Vector{Int} # ≈ Uᵢᴱ
	  # type :: CalculusTreeTools.type_calculus_tree
	  # convexity_status :: CalculusTreeTools.convexity_wrapper
	# end

	mutable struct PartitionedData_TR_PBFGS{G,T<:Number} <: Mod_ab_partitioned_data.PartitionedData
	  n :: Int 
	  N :: Int
	  vec_elt_fun :: Vector{Element_function} #length(vec_elt_fun) == N
	  # Vector composed by the different expression graph of element function .
	  # Several element function may have the same expression graph
	  M :: Int
	  vec_elt_complete_expr_tree :: Vector{G} # length(element_expr_tree) == M < N
	  element_expr_tree_table :: Vector{Vector{Int}} # length(element_expr_tree_table) == M
	  # element_expr_tree_table store the indices of every element function using each element_expr_tree, ∀i,j, 1 ≤ element_expr_tree_table[i][j] \leq N
	  index_element_tree :: Vector{Int} # length(index_element_tree) == N, index_element_tree[i] ≤ M
	
	  vec_compiled_element_gradients :: Vector{ReverseDiff.CompiledTape}	

	  x :: Vector{T} # length(x)==n
	  v :: Vector{T} # length(v)==n
		s :: Vector{T} # length(v)==n
	  pg :: PartitionedStructures.Elemental_pv{T} # partitioned gradient
	  pv :: PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
		ps :: PartitionedStructures.Elemental_pv{T} # partitioned vector, temporary partitioned vector
	  pB :: PartitionedStructures.Elemental_pm{T} # partitioned B

		fx :: T
		# g is build directly from pg
		# the result of pB*v will be store and build from pv
	end

	# @inline get_n(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.n
	# @inline get_N(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.N
	# @inline get_vec_elt_fun(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.vec_elt_fun
	# @inline get_M(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.M
	# @inline get_vec_elt_complete_expr_tree(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.vec_elt_complete_expr_tree
	# @inline get_vec_elt_complete_expr_tree(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, i::Int) where {G,T} = pd_pbfgs.vec_elt_complete_expr_tree[i]
	# @inline get_element_expr_tree_table(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.element_expr_tree_table
	# @inline get_index_element_tree(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.index_element_tree
	# @inline get_vec_compiled_element_gradients(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.vec_compiled_element_gradients
	# @inline get_vec_compiled_element_gradients(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, i::Int) where {G,T} = pd_pbfgs.vec_compiled_element_gradients[i]
	# @inline get_x(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.x
	# @inline get_v(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.v
	# @inline get_s(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.s
	# @inline get_pg(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.pg
	# @inline get_pv(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.pv
	# @inline get_ps(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.ps
	# @inline get_pB(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.pB
	# @inline get_fx(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T} = pd_pbfgs.fx

	# @inline set_n!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, n::Int) where {G,T} = pd_pbfgs.n = n
	# @inline set_N!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, N::Int) where {G,T} = pd_pbfgs.N = N
	# @inline set_vec_elt_fun!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, vec_elt_fun::Vector{Element_function}) where {G,T} = pd_pbfgs.vec_elt_fun .= vec_elt_fun
	# @inline set_M!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, M::Int) where {G,T} = pd_pbfgs.M = M
	# @inline set_vec_elt_complete_expr_tree!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, vec_elt_complete_expr_tree::Vector{G} ) where {G,T} = pd_pbfgs.vec_elt_complete_expr_tree .= vec_elt_complete_expr_tree
	# @inline set_element_expr_tree_table!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, element_expr_tree_table::Vector{Vector{Int}}) where {G,T} = pd_pbfgs.element_expr_tree_table .= element_expr_tree_table
	# @inline set_index_element_tree!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, index_element_tree::Vector{Int}) where {G,T} = pd_pbfgs.index_element_tree .= index_element_tree
	# @inline set_vec_compiled_element_gradients!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, vec_compiled_element_gradients::Vector{ReverseDiff.CompiledTape}) where {G,T} = pd_pbfgs.vec_compiled_element_gradients = vec_compiled_element_gradients
	# @inline set_x!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, x::Vector{T}) where {G,T} = pd_pbfgs.x .= x
	# @inline set_v!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, v::Vector{T}) where {G,T} = pd_pbfgs.v .= v
	# @inline set_s!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, s::Vector{T}) where {G,T} = pd_pbfgs.s .= s
	# @inline set_pg!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, pg::PartitionedStructures.Elemental_pm{T}) where {G,T} = pd_pbfgs.pg = pg
	# @inline set_pv!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, pv::PartitionedStructures.Elemental_pv{T}) where {G,T} = pd_pbfgs.pv = pv
	# @inline set_ps!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, ps::PartitionedStructures.Elemental_pv{T}) where {G,T} = pd_pbfgs.ps = ps
	# @inline set_pg!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, x::Vector{T}) where {G,T} = PartitionedStructures.epv_from_v!(pd_pbfgs.px, x)
	# @inline set_pv!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, v::Vector{T}) where {G,T} = PartitionedStructures.epv_from_v!(pd_pbfgs.pv, v)
	# @inline set_ps!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, s::Vector{T}) where {G,T} = PartitionedStructures.epv_from_v!(pd_pbfgs.ps, s)
	# @inline set_pB!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, pB::PartitionedStructures.Elemental_pm{T}) where {G,T} = pd_pbfgs.pB = pB
	# @inline set_fx!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, fx::T) where {G,T} = pd_pbfgs.fx = fx

	# function product_pd_pbfgs_x(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, x :: Vector{T}) where {G,T} 
		# res = similar(x)
		# product_pd_pbfgs_x!(res, pd_pbfgs,x)
		# return res
	# end 

	# function product_pd_pbfgs_x!(res::Vector{T}, pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, x :: Vector{T}) where {G,T} 
		# pB = get_pB(pd_pbfgs)
		# epvx = PartitionedStructures.epv_from_epm(pB)
		# epv_from_v!(epvx,x)
		# epv_res = similar(epvx)
		# product_pd_pbfgs_x!(epv_res, pB, epvx)
		# PartitionedStructures.build_v!(epv_res)
		# res .= PartitionedStructures.get_v(epv_res)
	# end 

	# product_pd_pbfgs_x!(epv_res::PartitionedStructures.Elemental_pv{T}, pB::PartitionedStructures.Elemental_pm{T}, epv::PartitionedStructures.Elemental_pv{T}) where T =	PartitionedStructures.mul_epm_epv!(epv_res, pB, epv)


	"""
			update_PBFGS(pd_pbfgs,x,s)
	Perform the PBFGS update givent the two iterate x and s
	"""
	update_PBFGS(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, x :: Vector{T}, s :: Vector{T}) where {G,T} = begin update_PBFGS!(pd_pbfgs,x,s); return Matrix(get_pB(pd_pbfgs)) end
	function update_PBFGS!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, x :: Vector{T}, s :: Vector{T}) where {G,T} 
		x != get_x(pd_pbfgs) && set_x!(pd_pbfgs, x)
		update_PBFGS!(pd_pbfgs,s)
	end 

	"""
			update_PBFGS(pd_pbfgs,s)
	Perform the PBFGS update givent the current iterate x and the next iterate s
	"""
	function update_PBFGS!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, s :: Vector{T}) where {G,T} 
		evaluate_y_part_data!(pd_pbfgs,s)
		py = get_pv(pd_pbfgs)
		set_ps!(pd_pbfgs,s)
		ps = get_ps(pd_pbfgs)
		pB = get_pB(pd_pbfgs)
		PartitionedStructures.PBFGS_update!(pB, py, ps)
	end 
	
	# """
			# evaluate_y_pd_pbfgs!(pd_pbfgs,x,s)
	# Compute the element gradients differences such as ∇̂fᵢ(x+s)-∇̂fᵢ(x) for each element functions. 
	# It stores the results in pd_pbfgs.pv
	# """
	# function evaluate_y_pd_pbfgs!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, x :: Vector{T}, s :: Vector{T}) where {G,T} 
		# x != get_x(pd_pbfgs) && set_x!(pd_pbfgs, x)
		# evaluate_y_pd_pbfgs!(pd_pbfgs,s)	
	# end

	# function evaluate_y_pd_pbfgs!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, s :: Vector{T}) where {G,T} 
		# evaluate_grad_pd_pbfgs!(pd_pbfgs)	
		# set_pv!(pd_pbfgs, get_pg(pd_pbfgs))
		# PartitionedStructures.minus_epv!(get_pv(pd_pbfgs))
		# set_x!(pd_pbfgs, get_x(pd_pbfgs)+s)
		# evaluate_grad_pd_pbfgs!(pd_pbfgs)
		# PartitionedStructures.add_epv!(get_pg(pd_pbfgs), get_pv(pd_pbfgs))
	# end 

	# """
			# evaluate_grad_pd_pbfgs(pd_pbfgs,x)
	# Build the gradient vector at the point x from the element gradient computed and stored in pd_pbfgs.pg .
	# """
	# evaluate_grad_pd_pbfgs(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, x :: Vector{T}) where {G,T} = begin g = similar(x); evaluate_grad_pd_pbfgs!(pd_pbfgs,x,g); g end 
	# function evaluate_grad_pd_pbfgs!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, x :: Vector{T}, g::Vector{T}) where {G,T}
		# x != get_x(pd_pbfgs) && set_x!(pd_pbfgs, x)
		# evaluate_grad_pd_pbfgs!(pd_pbfgs)
		# g .= PartitionedStructures.get_v(get_pg(pd_pbfgs))
	# end

	# function evaluate_grad_pd_pbfgs!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T}
		# set_pv!(pd_pbfgs, get_x(pd_pbfgs))
		# pg = get_pg(pd_pbfgs)
		# index_element_tree = get_index_element_tree(pd_pbfgs)
		# N = get_N(pd_pbfgs)
		# for i in 1:N
			# compiled_tape = get_vec_compiled_element_gradients(pd_pbfgs, index_element_tree[i])
			# Uix = PartitionedStructures.get_eev_value(get_pv(pd_pbfgs),i)
			# gi = PartitionedStructures.get_eev_value(get_pg(pd_pbfgs),i)
			# ReverseDiff.gradient!(gi, compiled_tape, Uix)
		# end
		# PartitionedStructures.build_v!(pg)
	# end

	# function evaluate_obj_pd_pbfgs(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}, x :: Vector{T}) where {G,T}
		# x != get_x(pd_pbfgs) && set_x!(pd_pbfgs, x)
		# evaluate_obj_pd_pbfgs!(pd_pbfgs)
		# return get_fx(pd_pbfgs)
	# end

	# function evaluate_obj_pd_pbfgs!(pd_pbfgs::PartitionedData_TR_PBFGS{G,T}) where {G,T}
		# set_pv!(pd_pbfgs, get_x(pd_pbfgs))
		# index_element_tree = get_index_element_tree(pd_pbfgs)
		# N = get_N(pd_pbfgs)
		# acc=0
		# for i in 1:N
			# elt_expr_tree = get_vec_elt_complete_expr_tree(pd_pbfgs, index_element_tree[i])
			# fix = CalculusTreeTools.evaluate_expr_tree(elt_expr_tree, PartitionedStructures.get_eev_value(get_pv(pd_pbfgs),i))
			# acc += fix
		# end
		# set_fx!(pd_pbfgs, acc)
	# end 
	
	"""
	    build_PartitionedData_TR_PBFGS(expr_tree, n)
	Find the partially separable structure of a function f stored as an expression tree expr_tree.
	To define properly the size of sparse matrix we need the size of the problem : n.
	At the end, we get the partially separable structure of f, f(x) = ∑fᵢ(xᵢ)
	"""
	function build_PartitionedData_TR_PBFGS(tree::G, n::Int; type::DataType=Float64, x0::Vector{T}=rand(type,n)) where {G,T}
		T != type && @error("confusion between type and the x0, abort")
	  expr_tree = CalculusTreeTools.transform_to_expr_tree(tree) :: CalculusTreeTools.t_expr_tree # transform the expression tree of type G into an expr tree of type t_expr_tree (the standard type used by my algorithms)
	  vec_element_function = CalculusTreeTools.delete_imbricated_plus(expr_tree) :: Vector{CalculusTreeTools.t_expr_tree} #séparation en fonction éléments
	  N = length(vec_element_function)

	  element_variables = map( (i -> CalculusTreeTools.get_elemental_variable(vec_element_function[i])), 1:N) ::Vector{Vector{Int}}	# retrieve elemental variables
	  sort!.(element_variables) # important line, sort the elemental varaibles. Mandatory for N_to_Ni and the partitioned structures
	  map( ((elt_fun,elt_var) -> CalculusTreeTools.element_fun_from_N_to_Ni!(elt_fun,elt_var)), vec_element_function, element_variables) # renumérotation des variables des fonctions éléments en variables internes
	
	  (element_expr_tree, index_element_tree) = distinct_element_expr_tree(vec_element_function, element_variables) # Remains only the distinct expr graph functions
	  M = length(element_expr_tree)
	  element_expr_tree_table = map( (i->findall((x -> x==i), index_element_tree)), 1:M) # create a table that give for each distinct element expr grah, every element function using it
	
	  vec_elt_complete_expr_tree = CalculusTreeTools.create_complete_tree.(element_expr_tree) # create complete tree given the remaining expr graph
	  vec_type_complete_element_tree = map(tree -> CalculusTreeTools.cast_type_of_constant(tree, type), vec_elt_complete_expr_tree) # cast the constant of the complete trees
	
	  CalculusTreeTools.set_bounds!.(vec_type_complete_element_tree) # deduce the bounds 
	  CalculusTreeTools.set_convexity!.(vec_type_complete_element_tree) # deduce the bounds 
	  # information concernant la convexité des arbres complets distincts

	  convexity_wrapper = map( (complete_tree -> CalculusTreeTools.convexity_wrapper(CalculusTreeTools.get_convexity_status(complete_tree)) ), vec_type_complete_element_tree) # convexity of element function
	  type_element_function = map(elt_fun -> CalculusTreeTools.get_type_tree(elt_fun), vec_type_complete_element_tree) # type of element function

	  vec_elt_fun = Vector{Element_function}(undef,N)
	  for i in 1:N  # Define the N element functions
			index_distinct_element_tree = index_element_tree[i]
			elt_fun = Element_function(i, index_distinct_element_tree, element_variables[i], type_element_function[index_distinct_element_tree], convexity_wrapper[index_distinct_element_tree])
			vec_elt_fun[i] = elt_fun
	  end

		vec_compiled_element_gradients = map( (tree -> compiled_grad_elmt_fun(tree; type=type)), element_expr_tree)

	  x = x0
	  v = similar(x)
		s = similar(x)

	  eev_set = map((elt_var -> create_eev(elt_var,type=type)), element_variables)  
	  pg = create_epv(eev_set; n=n)
	  pv = similar(pg)
		ps = similar(pg)
	
	  pB = identity_epm(element_variables, N, n; type=type)
		fx = -1
	  pd_pbfgs = PartitionedData_TR_PBFGS{CalculusTreeTools.complete_expr_tree, type}(n, N, vec_elt_fun, M, vec_elt_complete_expr_tree, element_expr_tree_table, index_element_tree, vec_compiled_element_gradients, x, v, s, pg, pv, ps, pB, fx)
	
	  return pd_pbfgs
	end

end 