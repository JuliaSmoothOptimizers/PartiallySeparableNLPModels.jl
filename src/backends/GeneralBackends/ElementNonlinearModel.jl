export ElementMOIModelBackend

"""
    ElementMOIModelBackend{T}

Composed of:
- `vec_element_evaluators::Vector{MOI.Nonlinear.Evaluator{MOI.Nonlinear.ReverseAD.NLPEvaluator}}`, M distinct element function `MOI.Nonlinear.Model`;
- `index_element_tree::Vector{Int}`, from which any of the N element function may associate a gradient tape from `vec_element_gradient_tapes`.
Each `MOI.Nonlinear.Evaluator{MOI.Nonlinear.ReverseAD.NLPEvaluator}` accumulates the element-function's contribution in a element-vector of a `PartitionedVector`.
"""
mutable struct ElementMOIModelBackend{T} <: PartitionedBackend{T}
  vec_element_evaluators::Vector{MOI.Nonlinear.Evaluator{MOI.Nonlinear.ReverseAD.NLPEvaluator}}
  index_element_tree::Vector{Int}
end

"""
    backend = ElementMOIModelBackend(vec_elt_expr_tree::Vector, index_element_tree::Vector{Int}; type=Float64)

Return an `ElementMOIModelBackend` from a `Vector` of expression trees
(supported by [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl))
of size `length(vec_elt_expr_tree)=M` and `index_element_tree` which redirects each element function `i`
 to its corresponding expression tree (1 ≤ `index_element_tree[i]` ≤ M, 1 ≤ i ≤ N).
"""
function ElementMOIModelBackend(vec_elt_expr_tree::Vector, index_element_tree::Vector{Int}; type=Float64)
  vec_element_evaluators = ExpressionTreeForge.non_linear_JuMP_model_evaluator.(vec_elt_expr_tree)
  ElementMOIModelBackend{type}(vec_element_evaluators, index_element_tree)
end 

function partitioned_gradient!(backend::ElementMOIModelBackend{T},
  x::PartitionedVector{T},
  g::PartitionedVector{T}
  ) where T
  epv_x = x.epv
  epv_g = g.epv
  index_element_tree = backend.index_element_tree
  N = length(index_element_tree)
  for i = 1:N
    evaluator = backend.vec_element_evaluators[index_element_tree[i]]
    Uix = PartitionedStructures.get_eev_value(epv_x, i)
    gi = PartitionedStructures.get_eev_value(epv_g, i)
    MOI.eval_objective_gradient(evaluator, gi, Uix)
  end
  return g
end

function objective(backend::ElementMOIModelBackend{T},
  x::PartitionedVector{T},
  ) where T
  epv_x = x.epv  
  index_element_tree = backend.index_element_tree
  N = length(index_element_tree)
  f = (T)(0)
  for i = 1:N
    evaluator = backend.vec_element_evaluators[index_element_tree[i]]
    Uix = PartitionedStructures.get_eev_value(epv_x, i)
    f += MathOptInterface.eval_objective(evaluator, Uix)
  end
  return f
end