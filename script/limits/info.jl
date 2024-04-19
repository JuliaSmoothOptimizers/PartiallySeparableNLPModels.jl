using Statistics
using ADNLPModels
using NLPModels, NLPModelsModifiers, ExpressionTreeForge, PartitionedStructures
using PartiallySeparableNLPModels
using PrettyTables

include("function.jl")

squared_nmin = 6
squared_nmax = 100
range = (len -> len^2).([6,25,50,100])

limit_problems = map(n -> ADNLPModel(limit, start_limit(n); name="limit$(n)"), range)
lenprob = length(limit_problems)
header = ["name", "n", "N", "M", "constant", "linear", "quadratic", "cubic", "general", "convex", "concave", "general", "mininimal element dimension", "mean element dimension", "maximal element dimension", "minimal elemental contribution (for 1 variables)", "mean elemental contribution (for 1 variables)", "maximal elemental contribution (for 1 variables)"]
data = Matrix{Any}(undef, lenprob, length(header))

for (index,nlp) in enumerate(limit_problems)
  psnlp = PSNLPModel(nlp)
  n = nlp.meta.nvar
  name = nlp.meta.name
  println("name:", name)
  n = psnlp.n
  N = psnlp.N
  M = psnlp.M

  element_functions = psnlp.vec_elt_fun

  element_function_types = (elt_fun -> elt_fun.type).(element_functions)
  constant = count(is_constant, element_function_types)
  linear = count(is_linear, element_function_types)
  quadratic = count(is_quadratic, element_function_types)
  cubic = count(is_cubic, element_function_types)
  general_type = count(is_more, element_function_types)

  element_function_convexity_status = (elt_fun -> elt_fun.convexity_status).(element_functions)
  convex = count(is_convex, element_function_convexity_status)
  concave = count(is_concave, element_function_convexity_status)
  general_convex = count(is_unknown, element_function_convexity_status)

  length_element_functions = (elt_fun -> length(elt_fun.variable_indices)).(element_functions)
  mean_length_element_functions = mean(length_element_functions)
  min_length_element_functions = minimum(length_element_functions)
  max_length_element_functions = maximum(length_element_functions)

  pv = psnlp.meta.x0.epv
  component_list = PartitionedStructures.get_component_list(pv)
  length_by_variable = (elt_list_var -> length(elt_list_var)).(component_list)
  mean_length_variable = mean(length_by_variable)
  min_length_variable = minimum(length_by_variable)
  max_length_variable = maximum(length_by_variable)

  data[index,:] .= [name, n, N, M, constant, linear, quadratic, cubic, general_type, convex, concave, general_convex, min_length_element_functions, mean_length_element_functions, max_length_element_functions, min_length_variable, mean_length_variable, max_length_variable]
end

path = pwd()*"/script/limits/results/"
io = open(path*"infos.tex", "w+")
pretty_table(io, data, backend = Val(:latex), header = header)
close(io)

io = open(path*"infos.txt", "w+")
pretty_table(io, data, header = header)
close(io)