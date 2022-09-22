using SparseArrays
using ReverseDiff, ForwardDiff
using PartitionedStructures, ExpressionTreeForge

export get_n,
  get_N,
  get_vec_elt_fun,
  get_M,
  get_vec_elt_complete_expr_tree,
  get_element_expr_tree_table,
  get_index_element_tree,
  get_vec_compiled_element_gradients
export get_x, get_v, get_s, get_pg, get_pv, get_py, get_ps, get_phv, get_pB, get_fx
export set_n!,
  set_N!,
  set_vec_elt_fun!,
  set_M!,
  set_vec_elt_complete_expr_tree!,
  set_element_expr_tree_table!,
  set_index_element_tree!,
  set_vec_compiled_element_gradients!
export set_x!,
  set_v!,
  set_s!,
  set_pg!,
  set_pv!,
  set_ps!,
  set_pg!,
  set_pv!,
  set_py!,
  set_ps!,
  set_phv!,
  set_pB!,
  set_fx!

export product_part_data_x, evaluate_obj_part_data, evaluate_grad_part_data
export product_part_data_x!,
  evaluate_obj_part_data!, evaluate_y_part_data!, evaluate_grad_part_data!
export update_nlp!

@inline get_n(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.n

@inline get_N(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.N

@inline get_vec_elt_fun(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.vec_elt_fun

@inline get_M(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.M

@inline get_vec_elt_complete_expr_tree(psnlp::AbstractPartiallySeparableNLPModel) =
  psnlp.vec_elt_complete_expr_tree

@inline get_vec_elt_complete_expr_tree(psnlp::AbstractPartiallySeparableNLPModel, i::Int) =
  psnlp.vec_elt_complete_expr_tree[i]

@inline get_element_expr_tree_table(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.element_expr_tree_table

@inline get_index_element_tree(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.index_element_tree

@inline get_vec_compiled_element_gradients(psnlp::AbstractPartiallySeparableNLPModel) =
  psnlp.vec_compiled_element_gradients

@inline get_vec_compiled_element_gradients(psnlp::AbstractPartiallySeparableNLPModel, i::Int) =
  psnlp.vec_compiled_element_gradients[i]

@inline get_x(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.x
@inline get_v(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.v
@inline get_s(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.s
@inline get_pg(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.pg
@inline get_pv(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.pv
@inline get_py(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.py
@inline get_ps(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.ps
@inline get_phv(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.phv
@inline get_pB(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.pB
@inline get_fx(psnlp::AbstractPartiallySeparableNLPModel) = psnlp.fx

@inline set_n!(psnlp::AbstractPartiallySeparableNLPModel, n::Int) = psnlp.n = n
@inline set_N!(psnlp::AbstractPartiallySeparableNLPModel, N::Int) = psnlp.N = N
@inline set_vec_elt_fun!(psnlp::AbstractPartiallySeparableNLPModel, vec_elt_fun::Vector{ElementFunction}) =
  psnlp.vec_elt_fun .= vec_elt_fun

@inline set_M!(psnlp::AbstractPartiallySeparableNLPModel, M::Int) = psnlp.M = M

@inline set_vec_elt_complete_expr_tree!(
  psnlp::AbstractPartiallySeparableNLPModel,
  vec_elt_complete_expr_tree::Vector{G},
) where {G} = psnlp.vec_elt_complete_expr_tree .= vec_elt_complete_expr_tree

@inline set_element_expr_tree_table!(
  psnlp::AbstractPartiallySeparableNLPModel,
  element_expr_tree_table::Vector{Vector{Int}},
) = psnlp.element_expr_tree_table .= element_expr_tree_table

@inline set_index_element_tree!(psnlp::AbstractPartiallySeparableNLPModel, index_element_tree::Vector{Int}) =
  psnlp.index_element_tree .= index_element_tree

@inline set_vec_compiled_element_gradients!(
  psnlp::AbstractPartiallySeparableNLPModel,
  vec_compiled_element_gradients::Vector{ReverseDiff.CompiledTape},
) = psnlp.vec_compiled_element_gradients = vec_compiled_element_gradients

@inline set_x!(psnlp::AbstractPartiallySeparableNLPModel, x::AbstractVector{Y}) where {Y <: Number} =
  psnlp.x .= x

@inline set_v!(psnlp::AbstractPartiallySeparableNLPModel, v::AbstractVector{Y}) where {Y <: Number} =
  psnlp.v .= v

@inline set_s!(psnlp::AbstractPartiallySeparableNLPModel, s::AbstractVector{Y}) where {Y <: Number} =
  psnlp.s .= s

@inline set_pg!(
  psnlp::AbstractPartiallySeparableNLPModel,
  pg::PartitionedStructures.Elemental_pv{Y},
) where {Y <: Number} = PartitionedStructures.epv_from_epv!(psnlp.pg, pg)

@inline set_pv!(
  psnlp::AbstractPartiallySeparableNLPModel,
  pv::PartitionedStructures.Elemental_pv{Y},
) where {Y <: Number} = PartitionedStructures.epv_from_epv!(psnlp.pv, pv)

@inline set_py!(
  psnlp::AbstractPartiallySeparableNLPModel,
  py::PartitionedStructures.Elemental_pv{Y},
) where {Y <: Number} = PartitionedStructures.epv_from_epv!(psnlp.py, py)

@inline set_ps!(
  psnlp::AbstractPartiallySeparableNLPModel,
  ps::PartitionedStructures.Elemental_pv{Y},
) where {Y <: Number} = PartitionedStructures.epv_from_epv!(psnlp.ps, ps)

@inline set_pg!(psnlp::AbstractPartiallySeparableNLPModel, x::AbstractVector{Y}) where {Y <: Number} =
  PartitionedStructures.epv_from_v!(psnlp.pg, x)

@inline set_pv!(psnlp::AbstractPartiallySeparableNLPModel, v::AbstractVector{Y}) where {Y <: Number} =
  PartitionedStructures.epv_from_v!(psnlp.pv, v)

@inline set_py!(psnlp::AbstractPartiallySeparableNLPModel, y::AbstractVector{Y}) where {Y <: Number} =
  PartitionedStructures.epv_from_v!(psnlp.py, y)

@inline set_ps!(psnlp::AbstractPartiallySeparableNLPModel, s::AbstractVector{Y}) where {Y <: Number} =
  PartitionedStructures.epv_from_v!(psnlp.ps, s)

@inline set_phv!(psnlp::AbstractPartiallySeparableNLPModel, s::AbstractVector{Y}) where {Y <: Number} =
  PartitionedStructures.epv_from_v!(psnlp.phv, s)

@inline set_pB!(
  psnlp::AbstractPartiallySeparableNLPModel,
  pB::PartitionedStructures.Elemental_pm{Y},
) where {Y <: Number} = psnlp.pB = pB

@inline set_fx!(psnlp::AbstractPartiallySeparableNLPModel, fx::Y) where {Y <: Number} = psnlp.fx = fx

update_nlp!(psnlp::AbstractPartiallySeparableNLPModel) = @error("Should not be called")

"""
    Bx = product_part_data_x(psnlp::AbstractPartiallySeparableNLPModel, v::AbstractVector{Y}) where {Y <: Number}

Return the product between the partitioned matrix `psnlp.pB` and `v`.
"""
function product_part_data_x(psnlp::AbstractPartiallySeparableNLPModel, v::AbstractVector{Y}) where {Y <: Number}
  res = similar(v)
  product_part_data_x!(res, psnlp, v)
  return res
end

"""
    product_part_data_x!(res::AbstractVector{Y}, psnlp::AbstractPartiallySeparableNLPModel, v::AbstractVector{Y}) where {Y <: Number}
    product_part_data_x!(epv_res::PartitionedStructures.Elemental_pv{Y}, psnlp::AbstractPartiallySeparableNLPModel, epv::PartitionedStructures.Elemental_pv{Y}) where {Y <: Number}
    product_part_data_x!(epv_res::PartitionedStructures.Elemental_pv{Y}, pB::T, epv::PartitionedStructures.Elemental_pv{Y}) where {Y <: Number, T <: PartitionedStructures.Part_mat{Y}}

Store the product between the partitioned matrix `psnlp.pB` and the vector `v` in `res`.
The computation of every element matrix-vector product require two partitioned vectors.
The first partitioned vector `epv` is created from `v` and the results are stored in the second partitioned vector `epv_res` which builds the value store by `g`.
"""
function product_part_data_x!(
  res::AbstractVector{Y},
  psnlp::AbstractPartiallySeparableNLPModel{Y,S},
  v::AbstractVector{Y},
) where {Y <: Number, S}
  pB = get_pB(psnlp)
  epv = get_ps(psnlp) # a first temporary partitioned vector
  PartitionedStructures.epv_from_v!(epv, v)
  epv_res = get_py(psnlp) # a second temporary partitioned vector
  product_part_data_x!(epv_res, pB, epv)
  PartitionedStructures.build_v!(epv_res)
  res .= PartitionedStructures.get_v(epv_res)
  return res
end

@inline product_part_data_x!(
  epv_res::PartitionedStructures.Elemental_pv{Y},
  psnlp::AbstractPartiallySeparableNLPModel{Y,S},
  epv::PartitionedStructures.Elemental_pv{Y},
) where {Y <: Number, S} = PartitionedStructures.mul_epm_epv!(epv_res, get_pB(psnlp), epv)

@inline product_part_data_x!(
  epv_res::PartitionedStructures.Elemental_pv{Y},
  pB::PartitionedStructures.Part_mat{Y},
  epv::PartitionedStructures.Elemental_pv{Y},
) where {Y <: Number} =
  PartitionedStructures.mul_epm_epv!(epv_res, pB, epv)

"""
    fx = evaluate_obj_part_data(psnlp::AbstractPartiallySeparableNLPModel, x::AbstractVector{Y}) where {Y <: Number}

Return the partially separable objective at `x`.
"""
function evaluate_obj_part_data(
  psnlp::AbstractPartiallySeparableNLPModel{Y,S},
  x::AbstractVector{Y},
) where {Y <: Number, S}
  set_x!(psnlp, x)
  evaluate_obj_part_data!(psnlp)
  return get_fx(psnlp)
end

"""
    evaluate_obj_part_data!(psnlp::AbstractPartiallySeparableNLPModel)

Compute the partially separable objective function, as a sum of element functions, at `psnlp.x`.
Its objective value is stored in `psnlp.fx` .
"""
function evaluate_obj_part_data!(psnlp::AbstractPartiallySeparableNLPModel)
  set_pv!(psnlp, get_x(psnlp))
  index_element_tree = get_index_element_tree(psnlp)
  N = get_N(psnlp)
  acc = 0
  for i = 1:N
    elt_expr_tree = get_vec_elt_complete_expr_tree(psnlp, index_element_tree[i])
    fix = ExpressionTreeForge.evaluate_expr_tree(
      elt_expr_tree,
      PartitionedStructures.get_eev_value(get_pv(psnlp), i),
    )
    acc += fix
  end
  set_fx!(psnlp, acc)
  return get_fx(psnlp)
end

"""
    evaluate_y_part_data!(psnlp::AbstractPartiallySeparableNLPModel, x::AbstractVector{Y}, s::AbstractVector{Y}) where {Y <: Number}
    evaluate_y_part_data!(psnlp::AbstractPartiallySeparableNLPModel, s::AbstractVector{Y}) where {Y <: Number}

Compute element gradients differences ∇̂fᵢ(x+s)-∇̂fᵢ(x) for each element function. 
Store the results in `psnlp.py`.
When `x` is ommited, consider `psnlp.x` as `x` and `psnlp.pg` as the partitioned gradient at `x`.
"""
function evaluate_y_part_data!(
  psnlp::AbstractPartiallySeparableNLPModel{Y,S},
  x::AbstractVector{Y},
  s::AbstractVector{Y},
) where {Y <: Number, S}
  set_x!(psnlp, x)
  evaluate_grad_part_data!(psnlp)
  evaluate_y_part_data!(psnlp, s)
  return get_py(psnlp)
end

function evaluate_y_part_data!(psnlp::AbstractPartiallySeparableNLPModel{Y,S}, s::AbstractVector{Y}) where {Y <: Number, S}
  set_s!(psnlp, s)
  set_py!(psnlp, get_pg(psnlp))
  PartitionedStructures.minus_epv!(get_py(psnlp))
  set_x!(psnlp, get_x(psnlp) + s)
  evaluate_grad_part_data!(psnlp)
  PartitionedStructures.add_epv!(get_pg(psnlp), get_py(psnlp))
  return get_py(psnlp)
end

"""
    gradient = evaluate_grad_part_data(psnlp::AbstractPartiallySeparableNLPModel, x::AbstractVector{Y}) where {Y <: Number}

Return the gradient vector `g` at `x`.
After the computation of the element gradients (stored in `psnlp.pg`), `g` is built by accumulating their contributions.
"""
function evaluate_grad_part_data(
  psnlp::AbstractPartiallySeparableNLPModel{Y,S},
  x::AbstractVector{Y},
) where {Y <: Number, S}
  g = similar(x)
  evaluate_grad_part_data!(g, psnlp, x)
  return g
end

"""
    evaluate_grad_part_data!(g::AbstractVector{Y}, psnlp::AbstractPartiallySeparableNLPModel, x::AbstractVector{Y}) where {Y <: Number}
    evaluate_grad_part_data!(psnlp::AbstractPartiallySeparableNLPModel)

Evaluate the gradient at `x` in place.
It computes and store the element gradients in `psnlp.pg` and accumulate their contributions in `g`.
When `g` and `x` are omitted, consider that `psnlp.g` and `psnlp.x` are `g` and `x`, respectively.
"""
function evaluate_grad_part_data!(
  g::AbstractVector{Y},
  psnlp::AbstractPartiallySeparableNLPModel{Y,S},
  x::AbstractVector{Y},
) where {Y <: Number, S}
  x != get_x(psnlp) && set_x!(psnlp, x)
  evaluate_grad_part_data!(psnlp)
  g .= PartitionedStructures.get_v(get_pg(psnlp))
  return g
end

function evaluate_grad_part_data!(psnlp::AbstractPartiallySeparableNLPModel)
  set_pv!(psnlp, get_x(psnlp))
  pg = get_pg(psnlp)
  index_element_tree = get_index_element_tree(psnlp)
  N = get_N(psnlp)
  for i = 1:N
    compiled_tape = get_vec_compiled_element_gradients(psnlp, index_element_tree[i])
    Uix = PartitionedStructures.get_eev_value(get_pv(psnlp), i)
    gi = PartitionedStructures.get_eev_value(get_pg(psnlp), i)
    ReverseDiff.gradient!(gi, compiled_tape, Uix)
  end
  PartitionedStructures.build_v!(pg)
  return pg
end