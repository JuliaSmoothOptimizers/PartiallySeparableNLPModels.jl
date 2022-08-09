using Test
using PartiallySeparableNLPModels
using LinearAlgebra

using ExpressionTreeForge, PartitionedStructures
using ADNLPModels, NLPModels, NLPModelsJuMP
using OptimizationProblems, OptimizationProblems.ADNLPProblems, OptimizationProblems.PureJuMP
using LinearOperators

include("partitoned_data/_include.jl")
