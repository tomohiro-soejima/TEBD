include("./VidalTEBD.jl")
using .VidalTEBD
using LinearAlgebra

#test make_superpositionMPO
N = 10
U = LinearAlgebra.Diagonal([0,1,0,0])
U2 = ones(4,N)
U1 = reshape(U*U2,2,2,N)
P = [1 for i in 1:N]
MPO = VidalTEBD.make_superpositionMPO(U1,P)

#make all up product state
zeros(Complex{Float64},2,2)
PS = zeros(Float64,(2,10))
PS[1,:] = ones(Float64,10)
MPS = VidalTEBD.make_productVidalMPS(PS,2)

#test do_MPOonMPS
MPS2 = VidalTEBD.do_MPOonMPS(MPS,MPO)

#test convert_to_Vidal
MPS3 = VidalTEBD.convert_to_Vidal(MPS2)
