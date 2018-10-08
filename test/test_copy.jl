using LinearAlgebra
using BenchmarkTools

function return_reshape(A,Tuple)
    copy(reshape(A,Tuple))
end

function return_reshape_prealloc(A,Tuple)
    B = zeros(Float64,Tuple)
    copyto!(reshape(A,Tuple),B)
    B
end

A = rand(300,300)

@btime return_reshape(A,(60,1500))
@btime return_reshape_prealloc(A,(60,1500))
