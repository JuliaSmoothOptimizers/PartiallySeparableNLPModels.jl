module Mod_partitionedNLPModel

	using CalculusTreeTools
	using ADNLPModels, NLPModels, NLPModelsJuMP
	using JuMP, MathOptInterface, ModelingToolkit
	using ..Mod_ab_partitioned_data
	using ..Mod_PBFGS, ..Mod_PLBFGS

	abstract type PartitionedNLPModel{T, S} <: AbstractNLPModel{T, S} end
	abstract type PQNNLPModel{T, S} <: PartitionedNLPModel{T, S} end


	SupportedNLPModel = Union{ADNLPModel, MathOptNLPModel} 
	# SupportedNLPModel{T, S} = Union{ADNLPModel{T, S}, MathOptNLPModel{T, S}} 
	# mutable struct typeA{T} end
	# mutable struct typeB{T} end
	# typeC{T} where T = Union{type1{T}, typeB{T}}

	function get_expr_tree(nlp :: MathOptNLPModel; x0 :: Vector{T}=copy(nlp.meta.x0), kwargs...) where T <: Number
		model = nlp.eval.m
		n = nlp.meta.nvar		
		evaluator = JuMP.NLPEvaluator(model)
		MathOptInterface.initialize(evaluator, [:ExprGraph])
		obj_Expr = MathOptInterface.objective_expr(evaluator) :: Expr
		ex = CalculusTreeTools.transform_to_expr_tree(obj_Expr) :: CalculusTreeTools.t_expr_tree
		CalculusTreeTools.print_tree(ex)
		return ex, n, x0
	end

	function get_expr_tree(adnlp :: ADNLPModel; x0 :: Vector{T}=copy(adnlp.meta.x0), kwargs...) where T <: Number
		n = adnlp.meta.nvar
		ModelingToolkit.@variables x[1:n]
		fun = adnlp.f(x)
		ex = CalculusTreeTools.transform_to_expr_tree(fun) :: CalculusTreeTools.t_expr_tree		
		CalculusTreeTools.print_tree(ex)
		return ex, n, x0
	end

	include("pbfgsNLPModel.jl")
	include("plbfgsNLPModel.jl")


	"""
	    f = obj(nlp, x)

	Evaluate `f(x)`, the objective function of `nlp` at `x`.
	"""
	NLPModels.obj(nlp :: P, x :: AbstractVector{T}) where {P <: PQNNLPModel{T,S}} where {T,S} =	evaluate_obj_part_data(nlp.part_data, x)

	"""
  	  g = grad!(nlp, x, g)

	Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
	"""
	function NLPModels.grad!(nlp :: P, x :: AbstractVector{T}, g :: AbstractVector{T}) where {P <: PQNNLPModel{T,S}} where {T,S}
		evaluate_grad_part_data!(g, nlp.part_data, x)
		return g
	end 

	export PBFGSNLPModel, PLBFGSNLPModel

end 