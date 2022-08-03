using Test
using PartiallySeparableNLPModels
using LinearAlgebra

using JuMP, MathOptInterface
using ADNLPModels, NLPModels, NLPModelsJuMP
using OptimizationProblems, OptimizationProblems.ADNLPProblems
using LinearOperators

include("partitoned_data/_include.jl")
