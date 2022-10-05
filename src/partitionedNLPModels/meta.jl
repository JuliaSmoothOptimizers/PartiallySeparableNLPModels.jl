module Meta

using NLPModels, PartitionedVectors
export partitioned_meta

"""    
    meta = partitioned_meta(meta::NLPModels.NLPModelMeta{T, Vector{T}}, x0::PartitionedVector{T})

Return an `NLPModelMeta` dedicated to `PartitionedVector`s, i.e. `meta.x0` is a `PartitionedVector`.
"""
function partitioned_meta(meta::NLPModels.NLPModelMeta{T, Vector{T}}, x0::PartitionedVector{T}) where T
  n = length(meta.x0)
  set!(x0, meta.x0)
  lvar = similar(x0)
  uvar = similar(x0)
  constraint_partition = [Vector{Int}(undef,0)]
  empty_pv = PartitionedVector(constraint_partition; T=T)
  y0 = similar(empty_pv)
  lcon = similar(empty_pv)
  ucon = similar(empty_pv)

  psmeta = NLPModels.NLPModelMeta{T, PartitionedVector{T}}(
    n, #var::Int
    x0, #::S
    lvar, #::S
    uvar, #::S
    Int[], #ifix::Vector{Int}
    Int[], #ilow::Vector{Int}
    Int[], #iupp::Vector{Int}
    Int[], #irng::Vector{Int}
    Int[1:n;], #ifree::Vector{Int}
    Int[], #iinf::Vector{Int}
    n, #nlvb::Int
    n, #nlvo::Int
    n, #nlvc::Int
    0, #ncon::Int
    y0, #::S
    lcon, #::S
    ucon, #::S
    Int[], #jfix::Vector{Int}
    Int[], #jlow::Vector{Int}
    Int[], #jupp::Vector{Int}
    Int[], #jrng::Vector{Int}
    Int[], #jfree::Vector{Int}
    Int[], #jinf::Vector{Int}
    n ,#nnzo::Int
    meta.nnzj, #nvar * ncon, nnzj::Int (0)
    meta.lin_nnzj, #lin_nnzj::Int (0)
    meta.nln_nnzj, #nln_nnzj::Int (0)
    meta.nnzh, #nnzh::Int (n*(n+1)/2)
    meta.nlin, #nlin::Int (n)
    meta.nnln, #nnln::Int (n)
    meta.lin, #lin::Vector{Int} (Int[])
    meta.nln, #nln::Vector{Int} (Int[])
    true, #minimize::Bool
    false, #islp::Bool
    "PS (WIP)", #name::String
  )
  return psmeta
end

Base.show(io::IO, psnlp::NLPModels.NLPModelMeta{T, PartitionedVector{T}}) where T = println("not done yet")

Base.show(psnlp::NLPModels.NLPModelMeta{T, PartitionedVector{T}}) where T = show(stdout, psnlp)

end