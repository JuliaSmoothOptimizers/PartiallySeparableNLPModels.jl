"""
    PQNNLPModel{ T, S, G, M <: AbstractNLPModel{T, S}, Meta <: AbstractNLPModelMeta{T, S}} <: AbstractPQNNLPModel{T, S}

A partitioned quasi-Newton `NLPModel`.
A `PQNNLPModel` has field:

* `meta` counting numerous information about the `PQNNLPModel`;
* `part_data` allocate the partitioned structures required by a partitioned quasi-Newton trust-region method;
* `nlp` the original `NLPModel`.
"""
mutable struct PQNNLPModel{
  T,
  S,
  G,
  M <: AbstractNLPModel{T, S},
  Meta <: AbstractNLPModelMeta{T, S},
} <: AbstractPQNNLPModel{T, S}
  meta::Meta
  part_data::PartitionedData_TR_PQN{G, T}
  nlp::M
end

function PQNNLPModel(nlp::N; kwargs...) where {N <: SupportedNLPModel}
  ex, n, x0 = get_expr_tree(nlp)
  part_data_plbfgs = build_PartitionedData_TR_PQN(ex, n; x0 = x0, kwargs...)
  meta = nlp.meta
  PQNNLPModel(meta, part_data_plbfgs, nlp)
end