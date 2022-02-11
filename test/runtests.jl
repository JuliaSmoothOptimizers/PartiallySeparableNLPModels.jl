using Test
using CalculusTreeTools, PartiallySeparableNLPModels

using JuMP, MathOptInterface, LinearAlgebra, SparseArrays

include("premier_test.jl")
include("compare_MOI_JuMP.jl")
include("partitoned_data/_include.jl")