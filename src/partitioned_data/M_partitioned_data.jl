module Mod_ab_partitioned_data
	using ReverseDiff
	using PartitionedStructures, CalculusTreeTools
	using ..Mod_common

	export PartitionedData
	export get_n, get_N, get_vec_elt_fun, get_M, get_vec_elt_complete_expr_tree, get_element_expr_tree_table, get_index_element_tree, get_vec_compiled_element_gradients
	export get_x, get_v, get_s, get_pg, get_pv, get_py, get_ps, get_pB, get_fx
	export set_n!, set_N!, set_vec_elt_fun!, set_M!, set_vec_elt_complete_expr_tree!, set_element_expr_tree_table!, set_index_element_tree!, set_vec_compiled_element_gradients!
	export set_x!, set_v!, set_s!, set_pg!, set_pv!, set_ps!, set_pg!, set_pv!, set_py!, set_ps!, set_pB!, set_fx!

	export product_part_data_x, evaluate_obj_part_data, evaluate_grad_part_data
	export product_part_data_x!, evaluate_obj_part_data!, evaluate_y_part_data!, evaluate_grad_part_data!
	export update_nlp!

	abstract type PartitionedData end 	

	@inline get_n(part_data::T) where T <: PartitionedData = part_data.n
	@inline get_N(part_data::T) where T <: PartitionedData = part_data.N
	@inline get_vec_elt_fun(part_data::T) where T <: PartitionedData = part_data.vec_elt_fun
	@inline get_M(part_data::T) where T <: PartitionedData = part_data.M
	@inline get_vec_elt_complete_expr_tree(part_data::T) where T <: PartitionedData = part_data.vec_elt_complete_expr_tree
	@inline get_vec_elt_complete_expr_tree(part_data::T, i::Int) where T <: PartitionedData = part_data.vec_elt_complete_expr_tree[i]
	@inline get_element_expr_tree_table(part_data::T) where T <: PartitionedData = part_data.element_expr_tree_table
	@inline get_index_element_tree(part_data::T) where T <: PartitionedData = part_data.index_element_tree
	@inline get_vec_compiled_element_gradients(part_data::T) where T <: PartitionedData = part_data.vec_compiled_element_gradients
	@inline get_vec_compiled_element_gradients(part_data::T, i::Int) where T <: PartitionedData = part_data.vec_compiled_element_gradients[i]
	@inline get_x(part_data::T) where T <: PartitionedData = part_data.x
	@inline get_v(part_data::T) where T <: PartitionedData = part_data.v
	@inline get_s(part_data::T) where T <: PartitionedData = part_data.s
	@inline get_pg(part_data::T) where T <: PartitionedData = part_data.pg
	@inline get_pv(part_data::T) where T <: PartitionedData = part_data.pv
	@inline get_py(part_data::T) where T <: PartitionedData = part_data.py
	@inline get_ps(part_data::T) where T <: PartitionedData = part_data.ps
	@inline get_pB(part_data::T) where T <: PartitionedData = part_data.pB
	@inline get_fx(part_data::T) where T <: PartitionedData = part_data.fx

	@inline set_n!(part_data::T, n::Int) where T <: PartitionedData = part_data.n = n
	@inline set_N!(part_data::T, N::Int) where T <: PartitionedData = part_data.N = N
	@inline set_vec_elt_fun!(part_data::T, vec_elt_fun::Vector{Element_function}) where T <: PartitionedData = part_data.vec_elt_fun .= vec_elt_fun
	@inline set_M!(part_data::T, M::Int) where T <: PartitionedData = part_data.M = M
	@inline set_vec_elt_complete_expr_tree!(part_data::T, vec_elt_complete_expr_tree::Vector{G} ) where {T<:PartitionedData,G} = part_data.vec_elt_complete_expr_tree .= vec_elt_complete_expr_tree
	@inline set_element_expr_tree_table!(part_data::T, element_expr_tree_table::Vector{Vector{Int}}) where T <: PartitionedData = part_data.element_expr_tree_table .= element_expr_tree_table
	@inline set_index_element_tree!(part_data::T, index_element_tree::Vector{Int}) where T <: PartitionedData = part_data.index_element_tree .= index_element_tree
	@inline set_vec_compiled_element_gradients!(part_data::T, vec_compiled_element_gradients::Vector{ReverseDiff.CompiledTape}) where T <: PartitionedData = part_data.vec_compiled_element_gradients = vec_compiled_element_gradients
	@inline set_x!(part_data::T, x::Vector{Y}) where {T<:PartitionedData,Y<:Number} = part_data.x .= x
	@inline set_v!(part_data::T, v::Vector{Y}) where {T<:PartitionedData,Y<:Number} = part_data.v .= v
	@inline set_s!(part_data::T, s::Vector{Y}) where {T<:PartitionedData,Y<:Number} = part_data.s .= s
	
	@inline set_pg!(part_data::T, pg::PartitionedStructures.Elemental_pv{Y}) where {T<:PartitionedData,Y<:Number} = PartitionedStructures.epv_from_epv!(part_data.pg, pg)
	@inline set_pv!(part_data::T, pv::PartitionedStructures.Elemental_pv{Y}) where {T<:PartitionedData,Y<:Number} = PartitionedStructures.epv_from_epv!(part_data.pv, pv)
	@inline set_py!(part_data::T, py::PartitionedStructures.Elemental_pv{Y}) where {T<:PartitionedData,Y<:Number} = PartitionedStructures.epv_from_epv!(part_data.py, py)
	@inline set_ps!(part_data::T, ps::PartitionedStructures.Elemental_pv{Y}) where {T<:PartitionedData,Y<:Number} = PartitionedStructures.epv_from_epv!(part_data.ps, ps)
	@inline set_pg!(part_data::T, x::Vector{Y}) where {T<:PartitionedData,Y<:Number} = PartitionedStructures.epv_from_v!(part_data.px, x)
	@inline set_pv!(part_data::T, v::Vector{Y}) where {T<:PartitionedData,Y<:Number} = PartitionedStructures.epv_from_v!(part_data.pv, v)
	@inline set_py!(part_data::T, y::Vector{Y}) where {T<:PartitionedData,Y<:Number} = PartitionedStructures.epv_from_v!(part_data.py, y)
	@inline set_ps!(part_data::T, s::Vector{Y}) where {T<:PartitionedData,Y<:Number} = PartitionedStructures.epv_from_v!(part_data.ps, s)
	@inline set_pB!(part_data::T, pB::PartitionedStructures.Elemental_pm{Y}) where {T <: PartitionedData,Y<:Number} = part_data.pB = pB
	@inline set_fx!(part_data::T, fx::Y) where {T<:PartitionedData,Y<:Number} = part_data.fx = fx

	update_nlp!(part_data :: T) where T <: PartitionedData = @error("Should not be called")

	"""
			product_part_data_x!(part_data, x)
	Return the product of the partitioned matrix `part_data*x`.
	"""
	function product_part_data_x(part_data::T, x :: Vector{Y}) where {T<:PartitionedData,Y<:Number}
		res = similar(x)
		product_part_data_x!(res, part_data,x)
		return res
	end 

	function product_part_data_x!(res::Vector{Y}, part_data::T, x::Vector{Y}) where {T<:PartitionedData,Y<:Number} 
		pB = get_pB(part_data)
		epvx = PartitionedStructures.epv_from_epm(pB)
		PartitionedStructures.epv_from_v!(epvx,x)
		epv_res = similar(epvx)
		product_part_data_x!(epv_res, pB, epvx)
		PartitionedStructures.build_v!(epv_res)
		res .= PartitionedStructures.get_v(epv_res)
	end 

	product_part_data_x!(epv_res::PartitionedStructures.Elemental_pv{Y}, part_data :: T, epv::PartitionedStructures.Elemental_pv{Y}) where {T <: PartitionedData, Y <: Number} =	PartitionedStructures.mul_epm_epv!(epv_res, get_pB(part_data), epv)
	product_part_data_x!(epv_res::PartitionedStructures.Elemental_pv{Y}, pB::T, epv::PartitionedStructures.Elemental_pv{Y}) where T <: PartitionedStructures.Part_mat{Y} where Y <: Number =	PartitionedStructures.mul_epm_epv!(epv_res, pB, epv)

	function evaluate_obj_part_data(part_data::T, x :: Vector{Y}) where {T<:PartitionedData,Y<:Number}
		set_x!(part_data, x)
		evaluate_obj_part_data!(part_data)
		return get_fx(part_data)
	end

	function evaluate_obj_part_data!(part_data::T) where T <: PartitionedData
		set_pv!(part_data, get_x(part_data))
		index_element_tree = get_index_element_tree(part_data)
		N = get_N(part_data)
		acc=0
		for i in 1:N
			elt_expr_tree = get_vec_elt_complete_expr_tree(part_data, index_element_tree[i])
			fix = CalculusTreeTools.evaluate_expr_tree(elt_expr_tree, PartitionedStructures.get_eev_value(get_pv(part_data),i))
			acc += fix
		end
		set_fx!(part_data, acc)
	end 

	# function evaluate_obj_part_data!(part_data::T) where T <: PartitionedData
	# 	set_pv!(part_data, get_x(part_data))	
	# 	element_expr_tree_table = get_element_expr_tree_table(part_data)
	# 	M = get_M(part_data)
	# 	acc=0
	# 	for i in 1:M
	# 		elt_expr_tree = get_vec_elt_complete_expr_tree(part_data, i)
	# 		indices_elt_fun = element_expr_tree_table[i]
	# 		for j in indices_elt_fun		
	# 			fix = CalculusTreeTools.evaluate_expr_tree(elt_expr_tree, PartitionedStructures.get_eev_value(get_pv(part_data),j))
	# 			acc += fix
	# 		end
	# 	end
	# 	set_fx!(part_data, acc)
	# end 


	"""
			evaluate_y_part_data!(part_data,x,s)
			evaluate_y_part_data!(part_data,s)
	Compute the element gradients differences such as ∇̂fᵢ(x+s)-∇̂fᵢ(x) for each element functions. 
	It stores the results in part_data.pv.
	evaluate_y_part_data!(part_data,s) consider that pg is alreagy the gradient of the point x
	"""
	function evaluate_y_part_data!(part_data::T, x :: Vector{Y}, s :: Vector{Y}) where {T<:PartitionedData,Y<:Number} 
		set_x!(part_data, x)
		evaluate_grad_part_data!(part_data)
		evaluate_y_part_data!(part_data,s)	
	end
	
	function evaluate_y_part_data!(part_data::T, s :: Vector{Y}) where {T<:PartitionedData,Y<:Number}		
		set_s!(part_data, s)
		set_py!(part_data, get_pg(part_data))
		PartitionedStructures.minus_epv!(get_py(part_data))
		set_x!(part_data, get_x(part_data)+s)
		evaluate_grad_part_data!(part_data)		
		PartitionedStructures.add_epv!(get_pg(part_data), get_py(part_data))

	end 

	"""
			evaluate_grad_part_data(part_data,x)
	Build the gradient vector at the point x from the element gradient computed and stored in part_data.pg .
	"""
	evaluate_grad_part_data(part_data::T, x :: Vector{Y}) where {T<:PartitionedData,Y<:Number} = begin g = similar(x); evaluate_grad_part_data!(g, part_data, x); g end 
	function evaluate_grad_part_data!(g::Vector{Y}, part_data::T, x :: Vector{Y}) where {T<:PartitionedData,Y<:Number}
		x != get_x(part_data) && set_x!(part_data, x)
		evaluate_grad_part_data!(part_data)
		g .= PartitionedStructures.get_v(get_pg(part_data))
	end
	
	function evaluate_grad_part_data!(part_data::T) where T <: PartitionedData
		set_pv!(part_data, get_x(part_data))
		pg = get_pg(part_data)
		index_element_tree = get_index_element_tree(part_data)
		N = get_N(part_data)
		for i in 1:N
			compiled_tape = get_vec_compiled_element_gradients(part_data, index_element_tree[i])
			Uix = PartitionedStructures.get_eev_value(get_pv(part_data),i)
			gi = PartitionedStructures.get_eev_value(get_pg(part_data),i)
			ReverseDiff.gradient!(gi, compiled_tape, Uix)
			@show gi
		end
		PartitionedStructures.build_v!(pg)
		@show part_data.pg
	end

end


