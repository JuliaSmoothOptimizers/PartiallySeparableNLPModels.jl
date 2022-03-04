
mutable struct PLBFGSNLPModel{T, S, G, M <: AbstractNLPModel{T, S}, Meta <: AbstractNLPModelMeta{T, S}} <: PQNNLPModel{T, S}
	meta :: Meta
	part_data :: PartitionedData_TR_PLBFGS{G, T}
	nlp :: M
end

function PLBFGSNLPModel(nlp :: N ) where N <: SupportedNLPModel
	ex, n, x0 = get_expr_tree(nlp)
	part_data_plbfgs = build_PartitionedData_TR_PLBFGS(ex, n; x0=x0)
	meta = nlp.meta
	PLBFGSNLPModel(meta, part_data_plbfgs, nlp)
end 