using LinearAlgebra
using BenchmarkTools

A = rand(100,100,100)
B = rand(100,100,100)

function multiple(A,B)
    A2 = PermutedDimsArray(reshape(A,10000,100),(2,1))
    B2 = PermutedDimsArray(reshape(B,100,10000),(2,1))
    B2*A2
end

function multiple2(A,B)
    A2 = permutedims(reshape(A,10000,100),(2,1))
    B2 = permutedims(reshape(B,100,10000),(2,1))
    B2*A2
end

A2 = rand(100,10000)
B2 = rand(10000,100)

@btime $B2*$A2
@btime multiple2($A,$B)
@btime multiple($A,$B)
