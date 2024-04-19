export SparseJacobianMoiModelBackend
export set_pv_from_sparse_jacobian!

"""
    SparseJacobianMoiModelBackend{T}

Compute the partitioned derivatives from the sparse Jacobian of the constraints.
Composed of:
- `evaluator::MOI.Nonlinear.Evaluator{MOI.Nonlinear.ReverseAD.NLPEvaluator}`: an evaluator of a `MOI.Nonlinear.Model` with a partially-separable objective and `N` constraits (=0) each parametrized by an element functions;
- `sparse_jacobian::Vector{T}`: mandatory to store in place the sparse Jacobianwith MathOptInterface;
- `translated_x::PartitionedVector{T}`: handle the partial derivative translations, usefull when the model doesn't depend on every variable.
"""
mutable struct SparseJacobianMoiModelBackend{T} <: PartitionedBackend{Float64}
  evaluator::MOI.Nonlinear.Evaluator{MOI.Nonlinear.ReverseAD.NLPEvaluator}
  sparse_jacobian::Vector{T}
  translated_x::PartitionedVector{T}
end

"""
    backend = SparseJacobianMoiModelBackend(vec_elt_expr_tree::Vector, n::Int; elemental_variables::Vector{Vector{Int}}), type=Float64)

Return an `SparseJacobianMoiModelBackend` from a `Vector` of expression trees
(supported by [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl))
of size `length(vec_elt_expr_tree)=N` and their element variables affiliated.
Suppose f(x) = f₁(x) + f₂(x) partially-separable considering the element functions f₁(x) = x₁ * x₂ * x₃² and f₂(x) = x₂ * x₃ * x₄ (N=2),
SparseJacobianMoiModelBackend defines a MOI.Nonlinear.Model having f(x) as the objective f₁ and f₂ as two constraints.
"""
function SparseJacobianMoiModelBackend(
  vec_elt_expr_tree::Vector,
  n::Int;
  elemental_variables = unique!(
    sort!(vcat(ExpressionTreeForge.get_elemental_variables.(vec_elt_expr_tree))),
  ),
  type::Type{T} = Float64,
) where {T}
  _elemental_variables = reduce((x, y) -> unique!(sort!(vcat(x, y))), elemental_variables)
  translated_x = PartitionedVector([_elemental_variables]; n, simulate_vector = true)
  evaluator = ExpressionTreeForge.sparse_jacobian_JuMP_model(vec_elt_expr_tree)
  non_empty_indices = MOI.jacobian_structure(evaluator)
  sparse_jacobian = Vector{Float64}(undef, length(non_empty_indices))
  return SparseJacobianMoiModelBackend{type}(evaluator, sparse_jacobian, translated_x)
end

function set_pv_from_sparse_jacobian!(
  pv::PartitionedVector{T},
  sparse_jacobian::Vector{T},
) where {T}
  cpt = 1
  for i = 1:size(pv, 1)
    nie = pv[i].nie
    range_sparse_jacobian = cpt:(cpt + nie - 1)
    pv[i].vec .= view(sparse_jacobian, range_sparse_jacobian)
    cpt += nie
  end
  return pv
end

function partitioned_gradient!(
  backend::SparseJacobianMoiModelBackend{T},
  x::PartitionedVector{T},
  g::PartitionedVector{T},
) where {T}
  PartitionedVectors.build!(x)
  PartitionedVectors.set!(backend.translated_x, x.epv.v)
  real_size_x = backend.translated_x[1].vec
  evaluator = backend.evaluator
  sparse_jacobian = backend.sparse_jacobian
  MOI.eval_constraint_jacobian(evaluator, sparse_jacobian, real_size_x)
  set_pv_from_sparse_jacobian!(g, sparse_jacobian)
  return g
end

function objective(backend::SparseJacobianMoiModelBackend{T}, x::PartitionedVector{T}) where {T}
  PartitionedVectors.build!(x)
  PartitionedVectors.set!(backend.translated_x, x.epv.v)
  real_size_x = backend.translated_x[1].vec
  evaluator = backend.evaluator
  return MOI.eval_objective(evaluator, real_size_x)
end
