
mutable struct PBFGSNLPModel{
  T,
  S,
  G,
  M <: AbstractNLPModel{T, S},
  Meta <: AbstractNLPModelMeta{T, S},
} <: AbstractPQNNLPModel{T, S}
  meta::Meta
  part_data::PartitionedData_TR_PBFGS{G, T}
  nlp::M
end

function PBFGSNLPModel(nlp::N) where {N <: SupportedNLPModel}
  ex, n, x0 = get_expr_tree(nlp)
  part_data_pbfgs = build_PartitionedData_TR_PBFGS(ex, n; x0 = x0)
  meta = nlp.meta
  PBFGSNLPModel(meta, part_data_pbfgs, nlp)
end
