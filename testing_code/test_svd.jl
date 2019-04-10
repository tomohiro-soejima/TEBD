using LinearAlgebra
using BenchmarkTools

A = rand(50,50)
B = rand(50,50,50,50)

@btime svd(A)
@btime svd(B[1,1,:,:])
@btime svd(copy(B[1,1,:,:]))
