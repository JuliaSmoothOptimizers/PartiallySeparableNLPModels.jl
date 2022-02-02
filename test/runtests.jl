using Test
using PartiallySeparableNLPModel

using JuMP, MathOptInterface, LinearAlgebra, SparseArrays
using CalculusTreeTools

include("premier_test.jl")
include("compare_MOI_JuMP.jl")
