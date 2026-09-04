module Meta

using NLPModels, PartitionedVectors
export partitioned_meta

"""
    meta = partitioned_meta(meta::NLPModels.NLPModelMeta{T, Vector{T}}, x0::PartitionedVector{T}; nnzj::Int = meta.nnzj, jac_available::Bool = false)

Return an `NLPModelMeta` dedicated to `PartitionedVector`s, i.e. `meta.x0` is a `PartitionedVector`.
`meta.ncon`/`lcon`/`ucon`/`y0` are carried over from `meta`; `y0`, `lcon` and `ucon` are
wrapped as a single-block `PartitionedVector` (as required by `NLPModelMeta{T, S}`, which
uses the same type `S` for `x0` and `y0`). `nnzj` may be overridden by models computing
their own (typically sparser) Jacobian sparsity pattern; `jac_available` must be set to
`true` by models that actually implement `NLPModels.jac_structure!`/`jac_coord!`.
"""
function partitioned_meta(
  meta::NLPModels.NLPModelMeta{T, Vector{T}},
  x0::PartitionedVector{T};
  nnzj::Int = meta.nnzj,
  jac_available::Bool = false,
) where {T}
  n = length(meta.x0)
  set!(x0, meta.x0)
  lvar = similar(x0)
  uvar = similar(x0)

  ncon = jac_available ? meta.ncon : 0
  constraint_partition = [collect(1:ncon)]
  cons_pv = PartitionedVector(constraint_partition; T = T, simulate_vector = true)
  y0 = similar(cons_pv)
  lcon = similar(cons_pv)
  ucon = similar(cons_pv)
  if ncon > 0
    set!(y0, meta.y0)
    set!(lcon, meta.lcon)
    set!(ucon, meta.ucon)
  end

  jfix = findall(j -> meta.lcon[j] == meta.ucon[j], 1:ncon)
  jlow = findall(j -> meta.lcon[j] > -Inf && meta.ucon[j] == Inf, 1:ncon)
  jupp = findall(j -> meta.lcon[j] == -Inf && meta.ucon[j] < Inf, 1:ncon)
  jrng =
    findall(j -> meta.lcon[j] > -Inf && meta.ucon[j] < Inf && meta.lcon[j] != meta.ucon[j], 1:ncon)
  jfree = findall(j -> meta.lcon[j] == -Inf && meta.ucon[j] == Inf, 1:ncon)
  jinf = findall(j -> meta.lcon[j] > meta.ucon[j], 1:ncon)
  setdiff!(jlow, jfix)
  setdiff!(jupp, jfix)

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
    ncon, #ncon::Int
    y0, #::S
    lcon, #::S
    ucon, #::S
    jfix, #jfix::Vector{Int}
    jlow, #jlow::Vector{Int}
    jupp, #jupp::Vector{Int}
    jrng, #jrng::Vector{Int}
    jfree, #jfree::Vector{Int}
    jinf, #jinf::Vector{Int}
    n,#nnzo::Int
    nnzj, #nnzj::Int
    meta.lin_nnzj, #lin_nnzj::Int (0)
    meta.nln_nnzj, #nln_nnzj::Int (0)
    meta.nnzh, #nnzh::Int (n*(n+1)/2)
    meta.nlin, #nlin::Int (n)
    meta.nnln, #nnln::Int (n)
    meta.lin, #lin::Vector{Int} (Int[])
    meta.nln, #nln::Vector{Int} (Int[])
    true, #minimize::Bool
    false, #islp::Bool
    meta.name * " (PS)", #name::String
    true, #variable_bounds_analysis::Bool
    true, #constraint_bounds_analysis::Bool
    true, #sparse_jacobian::Bool
    true, #sparse_hessian::Bool
    true, #grad_available::Bool
    jac_available, #jac_available::Bool
    true, #hess_available::Bool
    false, #jprod_available::Bool
    false, #jtprod_available::Bool
    true, #hprod_available::Bool
  )
  return psmeta
end

Base.show(io::IO, psnlp::NLPModels.NLPModelMeta{T, PartitionedVector{T}}) where {T} =
  println("not done yet")

Base.show(psnlp::NLPModels.NLPModelMeta{T, PartitionedVector{T}}) where {T} = show(stdout, psnlp)

end
