export ElementReverseForwardHprod

"""
    ElementReverseForwardHprod{T,G}

Compute the partitioned Hessian-product with `partitioned_hessian_prod!` by applying successively ReverseDiff and ForwardDiff.
It is composed of:
- `vec_elt_complete_expr_tree::Vector{G}`, the expression trees of the distinct element functions (of size M);
- `index_element_tree::Vector{Int}`, reffering to index in `vec_elt_complete_expr_tree` of any of the N element function;
"""
mutable struct ElementReverseForwardHprod{T,G} <: AbstractHprodBackend{T}
  vec_elt_complete_expr_tree::Vector{G} # of size M
  index_element_tree::Vector{Int} # of size N
end

"""
    hprod_backend = ElementReverseForwardHprod(complete_trees::Vector, index_element_tree::Vector{Int}; type=Float64)

Return an `ElementReverseForwardHprod` from a `Vector` of expression trees
(supported by [ExpressionTreeForge.jl](https://github.com/JuliaSmoothOptimizers/ExpressionTreeForge.jl))
`vec_elt_expr_tree` of size `length(vec_elt_expr_tree)=M` and `index_element_tree` which redirects each element function `i`
 to its corresponding expression tree (1 ≤ `index_element_tree[i]` ≤ M, 1 ≤ i ≤ N).
"""
function ElementReverseForwardHprod(vec_elt_expr_tree::Vector{G},
  index_element_tree::Vector{Int};
  type=Float64) where G
  ElementReverseForwardHprod{type,G}(vec_elt_expr_tree, index_element_tree)
end

function partitioned_hessian_prod!(backend::ElementReverseForwardHprod{T,G},
  x::PartitionedVector{T},
  v::PartitionedVector{T},
  Hv::PartitionedVector{T};
  obj_weight=(T)(1.)
  ) where {G,T}
  epv_x = x.epv
  epv_v = v.epv
  epv_Hv = Hv.epv

  index_element_tree = backend.index_element_tree
  N = length(index_element_tree)
  ∇f(x; f) = ReverseDiff.gradient(f, x)
  ∇²fv!(x, v, Hv; f) = ForwardDiff.derivative!(Hv, t -> ∇f(x + t * v; f), 0)

  for i = 1:N
    expr_tree = backend.vec_elt_complete_expr_tree[index_element_tree[i]]
    elf_fun = ExpressionTreeForge.evaluate_expr_tree(expr_tree)

    Uix = PartitionedStructures.get_eev_value(epv_x, i)
    Uiv = PartitionedStructures.get_eev_value(epv_v, i)
    Hvi = PartitionedStructures.get_eev_value(epv_Hv, i)
    ∇²fv!(Uix, Uiv, Hvi; f = elf_fun)
  end
  Hv .*= obj_weight
  return Hv
end