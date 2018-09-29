include("./VidalTEBD.jl")
using .VidalTEBD
using LinearAlgebra

N = 100
U = LinearAlgebra.Diagonal([0,0,1,0])
U2 = ones(4,N)
U1 = reshape(U*U2,2,2,N)
P = [exp(-(i-50)^2/50) for i in 1:100]
MPO = VidalTEBD.make_superpositionMPO(U1,P)
