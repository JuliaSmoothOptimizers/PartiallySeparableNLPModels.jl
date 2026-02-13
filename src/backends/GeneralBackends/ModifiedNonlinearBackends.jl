export ModifiedObjectiveMOIModelBackend
export set_vector_from_pv!, set_pv_from_vector!

"""
    ModifiedObjectiveMOIModelBackend{T}

Composed of:
- `evaluator`: the `Evaluator` of a modified partially-separable objective function : f(x) = ∑ᵢ fᵢ(x), where x ∈ ℜⁿ where n = ∑ᵢ nᵢ;
- `x_modified`: a vector x ∈ ℜⁿ where n = ∑ᵢ nᵢ;
- `v_modified`: a vector v ∈ ℜⁿ where n = ∑ᵢ nᵢ;
Each partial derivative of f corresponds to a partial derivative of an element function fᵢ.
"""
mutable struct ModifiedObjectiveMOIModelBackend{T} <: PartitionedBackend{Float64}
  evaluator::MOI.Nonlinear.Evaluator{MOI.Nonlinear.ReverseAD.NLPEvaluator}
  x_modified::Vector{T} # length = ∑ᵢᴺ nᵢ
  v_modified::Vector{T} # length = ∑ᵢᴺ nᵢ
end

"""
    backend = ModifiedObjectiveMOIModelBackend(vec_elt_expr_tree::Vector, index_element_tree::Vector{Int}; type=Float64)

Return an `ModifiedObjectiveMOIModelBackend` from a `Vector` of expression trees
(supported by [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl))
of size `length(vec_elt_expr_tree)=N` and their element variables affiliated.
Suppose f(x) = f₁(x) + f₂(x) partially-separable considering the element functions f₁(x) = x₁ * x₂ * x₃² and f₂(x) = x₂ * x₃ * x₄ (N=2),
ModifiedObjectiveMOIModelBackend defines a MOI.Nonlinear.Model where F(y) = y₁ * y₂ * y₃² + y₄ * y₅ * y₆ and its evaluator.
Each partial derivative of F corresponds to a partial derivative of a single element function fᵢ.
"""
function ModifiedObjectiveMOIModelBackend(
  vec_elt_expr_tree::Vector;
  type::Type{T} = Float64,
) where {T}
  element_variables = ExpressionTreeForge.get_elemental_variables.(vec_elt_expr_tree)
  acc = 0
  N = length(element_variables)
  for i = 1:N
    elt_fun = vec_elt_expr_tree[i]
    elt_var = element_variables[i]
    ExpressionTreeForge.normalize_indices!(elt_fun, elt_var; initial_index = acc)
    acc += length(elt_var)
  end
  modified_expr_tree = ExpressionTreeForge.sum_expr_trees(vec_elt_expr_tree)
  evaluator = ExpressionTreeForge.non_linear_JuMP_model_evaluator(modified_expr_tree)

  sum_nie = mapreduce(elt_var -> length(elt_var), +, element_variables)
  x_modified = Vector{type}(undef, sum_nie)
  v_modified = similar(x_modified)
  return ModifiedObjectiveMOIModelBackend{type}(evaluator, x_modified, v_modified)
end

function set_vector_from_pv!(v::Vector{T}, pv::PartitionedVector{T}) where {T}
  cpt = 1
  for i = 1:size(pv, 1)
    nie = pv[i].nie
    range = cpt:(cpt + nie - 1)
    view(v, range) .= pv[i].vec
    cpt += nie
  end
  return v
end

function set_pv_from_vector!(pv::PartitionedVector{T}, v::Vector{T}) where {T}
  cpt = 1
  for i = 1:size(pv, 1)
    nie = pv[i].nie
    range = cpt:(cpt + nie - 1)
    pv[i].vec .= view(v, range)
    cpt += nie
  end
  return pv
end

function partitioned_gradient!(
  backend::ModifiedObjectiveMOIModelBackend{T},
  x::PartitionedVector{T},
  g::PartitionedVector{T},
) where {T}
  x_modified = backend.x_modified
  v_modified = backend.v_modified
  set_vector_from_pv!(x_modified, x)
  evaluator = backend.evaluator
  MathOptInterface.eval_objective_gradient(evaluator, v_modified, x_modified)
  set_pv_from_vector!(g, v_modified)
  return g
end

function objective(backend::ModifiedObjectiveMOIModelBackend{T}, x::PartitionedVector{T}) where {T}
  x_modified = backend.x_modified
  set_vector_from_pv!(x_modified, x)
  evaluator = backend.evaluator
  return MathOptInterface.eval_objective(evaluator, x_modified)
end
