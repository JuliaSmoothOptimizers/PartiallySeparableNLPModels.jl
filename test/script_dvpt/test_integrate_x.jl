using BenchmarkTools

struct st_test{N}
    x :: Vector{Float64}
    v :: Array{SubArray{Float64,1,Array{Float64,1},N,false},1}
end


function empty_st_test(x :: Vector{Float64})
    n = length(x)-2
    st_view = Vector{SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}}(undef,n)
    for i in 1:n
        st_view[i] = view(x, [i:(i+2);])
    end
    return st_view
end


function empt_st_struct(x :: Vector{Float64})
    st_view = empty_st_test(x)
    return st_test(x,st_view)
end

set_st(st :: st_test{N}, x :: Vector{Float64}) where N = st.x .= x

n = 1000000
x1 = rand(n)
def_st = empt_st_struct(x1)
# bench1 = @benchmark empt_st_struct(x1)
x2 = ones(n)
# bench2 = @benchmark set_st(def_st,x2)
# @show def_st
