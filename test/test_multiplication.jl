using LinearAlgebra
using BenchmarkTools

function multiply(A,B,d,D1,D2)
    A2 = reshape(A,d,D2*D1^2)
    B2 = reshape(PermutedDimsArray(B,(1,3,5,2,4,6)),D2*D1^2,D2*D1^2)
    A2*B2
end

function multiply_copy(A,B,d,D1,D2)
    A2 = copy(reshape(A,d,D2*D1^2))
    B2 = copy(reshape(PermutedDimsArray(B,(1,3,5,2,4,6)),D2*D1^2,D2*D1^2))
    A2*B2
end

function multiply_copy2(A,B,d,D1,D2)
    A2 = Array{Float64}(undef,d,D2*D1^2)
    A2[:,:] = copy(reshape(A,d,D2*D1^2))
    B2 = Array{Float64}(undef,D2*D1^2,D2*D1^2)
    B2[:,:] = copy(reshape(PermutedDimsArray(B,(1,3,5,2,4,6)),D2*D1^2,D2*D1^2))
    A2*B2
end

multiply(2,4,4)
multiply_copy(2,4,4)
d = 1
D1 = 40
D2 = 4
A = rand(Float64,d,D2,D1,D1)
B = rand(Float64,D1,D1,D2,D2,D1,D1)
@btime multiply($A,$B,d,D1,D2)
@btime multiply_copy($A,$B,d,D1,D2)
@btime multiply_copy2($A,$B,d,D1,D2)
