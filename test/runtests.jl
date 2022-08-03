using Test
using PartiallySeparableNLPModels
using LinearAlgebra

using ExpressionTreeForge, PartitionedStructures
using JuMP, MathOptInterface
using ADNLPModels, NLPModels, NLPModelsJuMP
using OptimizationProblems, OptimizationProblems.ADNLPProblems, OptimizationProblems.PureJuMP

include("partitoned_data/_include.jl")
