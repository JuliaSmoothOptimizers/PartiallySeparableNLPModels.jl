"""
    f = obj(nlp, x)

Evaluate `f(x)`, the objective function of `nlp` at `x`.
"""
function NLPModels.obj(
  pqnnlp::AbstractPQNNLPModel{T, S},
  x::S, # PartitionedVector
) where {T, S<:AbstractVector{T}} 
  increment!(pqnnlp, :neval_obj)
  PartitionedVectors.build!(x)
  NLPModels.obj(pqnnlp.model, x.epv.v)
end

"""
    g = grad(nlp, x)

Evaluate `∇f(x)`, the gradient of the objective function at `x`.
"""
function NLPModels.grad(
  pqnnlp::AbstractPQNNLPModel{T, S},
  x::S, # PartitionedVector  
) where {T, S<:AbstractVector{T}} 
  g = similar(x; simulate_vector=false)
  grad!(pqnnlp, x, g)
  return g
end

"""
    g = grad!(nlp, x, g)

Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
"""
function NLPModels.grad!(
  pqnnlp::AbstractPQNNLPModel{T, S},
  x::S, # PartitionedVector
  g::S, # PartitionedVector
) where {T, S<:AbstractVector{T}} 
  increment!(pqnnlp, :neval_grad)  
  epv_x = x.epv
  epv_g = g.epv
  index_element_tree = get_index_element_tree(pqnnlp)
  N = get_N(pqnnlp)
  for i = 1:N
    compiled_tape = get_vec_compiled_element_gradients(pqnnlp, index_element_tree[i])
    Uix = PartitionedStructures.get_eev_value(epv_x, i)
    gi = PartitionedStructures.get_eev_value(epv_g, i)
    ReverseDiff.gradient!(gi, compiled_tape, Uix)
  end
  return g
end

"""
    hprod!(nlp::AbstractPQNNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight=1.)

Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function NLPModels.hprod(
  pqnnlp::AbstractPQNNLPModel{T,S},
  x::S,
  v::S;
  obj_weight = 1.0,
) where {T, S<:AbstractVector{T}} 
  Hv = similar(x; simulate_vector=false)
  NLPModels.hprod!(pqnnlp, x, v, Hv; obj_weight)
  return Hv
end 

"""
    hprod!(nlp::AbstractPQNNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight=1.)

Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function NLPModels.hprod!(
  pqnnlp::AbstractPQNNLPModel{T,S},
  x::S,
  v::S,
  Hv::S;
  obj_weight = 1.0,
) where {T, S<:AbstractVector{T}} 
  increment!(pqnnlp, :neval_hprod)
  epv_Hv = Hv.epv
  epv_v = v.epv
  op = pqnnlp.op
  mul_epm_epv!(epv_Hv, op, epv_v)
  Hv .*= obj_weight
  return Hv
end

function NLPModels.hess_op(
  pqnnlp::AbstractPQNNLPModel{T,S},
  x::S;
  obj_weight = 1.0,
) where {T, S<:AbstractVector{T}} 
  Hv = similar(x; simulate_vector=false)
  return hess_op!(pqnnlp, x, Hv; obj_weight)
end

function NLPModels.hess_op!(
  pqnnlp::AbstractPQNNLPModel{T,S},
  x::S,
  Hv::S;
  obj_weight = 1.0,
) where {T, S<:AbstractVector{T}} 
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

function Base.push!(
  pqn_nlp::AbstractPQNNLPModel{T,S},
  s::S,
  y::S;
  kwargs...,
) where {T, S<:AbstractVector{T}} 
  epv_s = s.epv
  epv_y = y.epv
  op = pqn_nlp.op
  PartitionedStructures.update!(op, epv_y, epv_s; name = pqn_nlp.name, verbose=false, kwargs...)
  return op
end

function NLPModels.reset_data!(pqnnlp::AbstractPQNNLPModel; name=pqnnlp.name)
  convex_vector = map(eem -> PartitionedStructures.get_convex(eem),pqnnlp.op.eem_set)
  epv = pqnnlp.meta.x0.epv
  (name == :pbfgs) && (op = epm_from_epv(epv))
  (name == :psr1) && (op = epm_from_epv(epv))
  (name == :pse) && (op = epm_from_epv(epv))
  (name == :pcs) && (op = epm_from_epv(epv); convex_vector)
  (name == :plbfgs) && (op = eplo_lbfgs_from_epv(epv; kwargs...))
  (name == :plsr1) && (op = eplo_lsr1_from_epv(epv))
  (name == :plse) && (op = eplo_lose_from_epv(epv; kwargs...))
  pqnnlp.op = op
  return pqnnlp
end