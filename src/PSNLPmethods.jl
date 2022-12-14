"""
    f = obj(nlp, x)

Evaluate `f(x)`, the objective function of `nlp` at `x`.
"""
function NLPModels.obj(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S, # PartitionedVector
) where {T, S<:AbstractVector{T}} 
  increment!(psnlp, :neval_obj)
  build!(x)
  NLPModels.obj(psnlp.nlp, x.epv.v)
end

"""
    g = grad(nlp, x)

Evaluate `∇f(x)`, the gradient of the objective function at `x`.
"""
function NLPModels.grad(
  psnlp::AbstractPartiallySeparableNLPModel{T, S},
  x::S, # PartitionedVector  
) where {T, S<:AbstractVector{T}} 
  g = similar(x; simulate_vector=false)
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
) where {T, S<:AbstractVector{T}} 
  increment!(psnlp, :neval_grad)  
  epv_x = x.epv
  epv_g = g.epv
  index_element_tree = get_index_element_tree(psnlp)
  N = get_N(psnlp)
  for i = 1:N
    compiled_tape = get_vec_compiled_element_gradients(psnlp, index_element_tree[i])
    Uix = PartitionedStructures.get_eev_value(epv_x, i)
    gi = PartitionedStructures.get_eev_value(epv_g, i)
    ReverseDiff.gradient!(gi, compiled_tape, Uix)
  end
  return g
end

"""
    hprod!(nlp::AbstractPartiallySeparableNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight=1.)

Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function NLPModels.hprod(
  psnlp::AbstractPartiallySeparableNLPModel{T,S},
  x::S,
  v::S;
  obj_weight = 1.0,
  β = 0.0,
) where {T, S<:AbstractVector{T}} 
  Hv = similar(x; simulate_vector=false)
  NLPModels.hprod!(psnlp, x, v, Hv; obj_weight, β)
  return Hv
end 

"""
    hprod!(nlp::AbstractPartiallySeparableNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight=1.)

Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function NLPModels.hprod!(
  psnlp::AbstractPartiallySeparableNLPModel{T,S},
  x::S,
  v::S,
  Hv::S;
  obj_weight = 1.0,
  β = 0.0,
) where {T, S<:AbstractVector{T}} 
  increment!(psnlp, :neval_hprod)
  epv_x = x.epv
  epv_v = v.epv
  epv_Hv = Hv.epv

  index_element_tree = get_index_element_tree(psnlp)
  N = get_N(psnlp)
  ∇f(x; f) = ReverseDiff.gradient(f, x)
  ∇²fv!(x, v, Hv; f) = ForwardDiff.derivative!(Hv, t -> ∇f(x + t * v; f), 0)

  for i = 1:N
    complete_tree = get_vec_elt_complete_expr_tree(psnlp, index_element_tree[i])
    elf_fun = ExpressionTreeForge.evaluate_expr_tree(complete_tree)

    Uix = PartitionedStructures.get_eev_value(epv_x, i)
    Uiv = PartitionedStructures.get_eev_value(epv_v, i)
    Hvi = PartitionedStructures.get_eev_value(epv_Hv, i)
    ∇²fv!(Uix, Uiv, Hvi; f = elf_fun)
  end
  Hv .*= obj_weight
  return Hv 
end

function NLPModels.hess_op(
  pqnnlp::AbstractPartiallySeparableNLPModel{T,S},
  x::S;
  obj_weight = 1.0,
) where {T, S<:AbstractVector{T}} 
  Hv = similar(x; simulate_vector=false)
  return hess_op!(pqnnlp, x, Hv; obj_weight)
end

function NLPModels.hess_op!(
  pqnnlp::AbstractPartiallySeparableNLPModel{T,S},
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

