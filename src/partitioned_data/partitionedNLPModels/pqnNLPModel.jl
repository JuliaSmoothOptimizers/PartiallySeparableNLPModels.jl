"""
    PQNNLPModel{ T, S, G, M <: AbstractNLPModel{T, S}, Meta <: AbstractNLPModelMeta{T, S}} <: AbstractPQNNLPModel{T, S}

A partitioned quasi-Newton `NLPModel`.
A `PQNNLPModel` has field:

* `meta` counting numerous information about the `PQNNLPModel`;
* `part_data` allocate the partitioned structures required by a partitioned quasi-Newton trust-region method;
* `nlp` the original `NLPModel`;
* `counters` counting the how many standards methods are called.
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
  counters::NLPModels.Counters
end

function PQNNLPModel(nlp::SupportedNLPModel; kwargs...)
  ex, n, x0 = get_expr_tree(nlp)
  part_data_plbfgs = build_PartitionedData_TR_PQN(ex, n; x0 = x0, kwargs...)
  meta = nlp.meta
  counters = NLPModels.Counters()
  PQNNLPModel(meta, part_data_plbfgs, nlp, counters)
end

function update_nlp(nlp::PQNNLPModel, x::AbstractVector{T}, s::AbstractVector{T}) where T
  update_nlp!(nlp, x, s)
  return Matrix(get_pB(nlp.part_data))
end 

function Mod_PQN.update_nlp!(nlp::PQNNLPModel, x::AbstractVector{T}, s::AbstractVector{T}) where T
  part_data = nlp.part_data
  update_PQN!(part_data, x, s)
  return part_data
end

Mod_ab_partitioned_data.get_pB(nlp::PQNNLPModel) = get_pB(nlp.part_data)

Mod_ab_partitioned_data.get_py(nlp::PQNNLPModel) = get_py(nlp.part_data)

Mod_ab_partitioned_data.product_part_data_x!(
  res::Vector{Y},
  nlp::PQNNLPModel,
  x::Vector{Y}) where {Y <: Number} =
  product_part_data_x!(res, nlp.part_data, x) 

Mod_ab_partitioned_data.product_part_data_x(
  nlp::PQNNLPModel,
  x::Vector{Y}) where {Y <: Number} =
  product_part_data_x(nlp.part_data, x)