using FastClosures
using LinearOperators, NLPModels, ReverseDiff, ForwardDiff
using PartitionedVectors

using ..ModAbstractPSNLPModels, ..PartitionedBackends

"""
    f = obj(nlp, x)

Evaluate `f(x)`, the objective function of `nlp` at `x`.
"""
function NLPModels.obj(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S, # PartitionedVector
) where {T, S <: AbstractVector{T}}
  increment!(psnlp, :neval_obj)
  objective(psnlp.objective_backend, x)
end

"""
    g = grad(nlp, x)

Evaluate `∇f(x)`, the gradient of the objective function at `x`.
"""
function NLPModels.grad(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S, # PartitionedVector  
) where {T, S <: AbstractVector{T}}
  g = similar(x; simulate_vector = false)
  grad!(psnlp, x, g)
  return g
end

"""
    g = grad!(nlp, x, g)

Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
"""
function NLPModels.grad!(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S, # PartitionedVector
  g::S, # PartitionedVector
) where {T, S <: AbstractVector{T}}
  increment!(psnlp, :neval_grad)
  partitioned_gradient!(psnlp.gradient_backend, x, g)
  return g
end

"""
    hprod!(nlp::AbstractPartiallySeparableNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight=1.)

Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function NLPModels.hprod(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S,
  v::S;
  obj_weight = 1.0,
  β = 0.0,
) where {T, S <: AbstractVector{T}}
  Hv = similar(x; simulate_vector = false)
  NLPModels.hprod!(psnlp, x, v, Hv; obj_weight, β)
  return Hv
end

"""
    hprod!(nlp::AbstractPartiallySeparableNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight=1.)

Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function NLPModels.hprod!(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S,
  v::S,
  Hv::S;
  obj_weight = 1.0,
  β = 0.0,
) where {T, S <: AbstractVector{T}}
  increment!(psnlp, :neval_hprod)
  partitioned_hessian_prod!(psnlp.hprod_backend, x, v, Hv; obj_weight)
  return Hv
end

"""
    c = cons(nlp::AbstractPartiallySeparableNLPModel, x)

Evaluate `c(x)`, the vector of constraints of `nlp` at `x`.
"""
function NLPModels.cons(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S, # PartitionedVector
) where {T, S <: AbstractVector{T}}
  c = similar(psnlp.meta.y0)
  NLPModels.cons!(psnlp, x, c)
  return c
end

"""
    cons!(nlp::AbstractPartiallySeparableNLPModel, x, c)

Evaluate `c(x)`, the vector of constraints of `nlp` at `x`, in place.
"""
function NLPModels.cons!(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S, # PartitionedVector
  c::S, # PartitionedVector
) where {T, S <: AbstractVector{T}}
  increment!(psnlp, :neval_cons)
  cvals = Vector{T}(undef, length(psnlp.vec_cons))
  for cons_j in psnlp.vec_cons
    cvals[cons_j.j] = objective(cons_j.objective_backend, x)
  end
  PartitionedVectors.set!(c, cvals)
  return c
end

"""
    jac_structure!(nlp::AbstractPartiallySeparableNLPModel, rows, cols)

Return the structure of the constraints Jacobian of `nlp` in sparse coordinate format,
in place. The sparsity pattern of row `j` is the union of the elemental variables of every
element function composing the `j`-th constraint.
"""
function NLPModels.jac_structure!(
  psnlp::AbstractPartiallySeparableNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  cpt = 1
  for cons_j in psnlp.vec_cons
    for k in cons_j.variable_indices
      rows[cpt] = cons_j.j
      cols[cpt] = k
      cpt += 1
    end
  end
  return rows, cols
end

"""
    vals = jac_coord(nlp::AbstractPartiallySeparableNLPModel, x)

Evaluate `J(x)`, the constraints Jacobian of `nlp` at `x`, in sparse coordinate format.
`vals` is a plain `Vector{T}`: unlike `x`/`g`, the Jacobian coordinate values are not
naturally a `PartitionedVector`, so the generic `NLPModels.jac_coord` fallback (which
allocates `S(undef, nnzj)`) cannot be used here.
"""
function NLPModels.jac_coord(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S, # PartitionedVector
) where {T, S <: AbstractVector{T}}
  vals = Vector{T}(undef, psnlp.meta.nnzj)
  return NLPModels.jac_coord!(psnlp, x, vals)
end

"""
    jac_coord!(nlp::AbstractPartiallySeparableNLPModel, x, vals)

Evaluate `J(x)`, the constraints Jacobian of `nlp` at `x`, in sparse coordinate format, in
place. The coordinates match the order produced by `jac_structure!`.
"""
function NLPModels.jac_coord!(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S, # PartitionedVector
  vals::AbstractVector{T},
) where {T, S <: AbstractVector{T}}
  increment!(psnlp, :neval_jac)
  xvec = Vector(x)
  cpt = 1
  for cons_j in psnlp.vec_cons
    # cons_j.gradient_backend requires an input sharing its own (constraint-specific)
    # partition, which generally differs from psnlp's own (objective-based) partition of x
    PartitionedVectors.set!(cons_j.local_x, xvec)
    g = similar(cons_j.local_x; simulate_vector = false)
    partitioned_gradient!(cons_j.gradient_backend, cons_j.local_x, g)
    gvec = Vector(g)
    for k in cons_j.variable_indices
      vals[cpt] = gvec[k]
      cpt += 1
    end
  end
  return vals
end

function NLPModels.hess_op(
  pqnnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S;
  obj_weight = 1.0,
) where {T, S <: AbstractVector{T}}
  Hv = similar(x; simulate_vector = false)
  return hess_op!(pqnnlp, x, Hv; obj_weight)
end

function NLPModels.hess_op!(
  pqnnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S,
  Hv::S;
  obj_weight = 1.0,
) where {T, S <: AbstractVector{T}}
  n = get_n(pqnnlp)
  prod! = @closure (res, v, α, β) -> begin
    hprod!(pqnnlp, x, v, Hv; obj_weight = obj_weight)
    if β == 0
      @. res = α * Hv
    else
      @. res = α * Hv + β * res
    end
  end
  B = LinearOperator(T, n, n, true, true, prod!)
  return B
end
