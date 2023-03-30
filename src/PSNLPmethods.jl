using FastClosures
using LinearOperators, NLPModels, ReverseDiff, ForwardDiff

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
  PartitionedBackends.partitioned_gradient!(psnlp.gradient_backend, x, g)
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
