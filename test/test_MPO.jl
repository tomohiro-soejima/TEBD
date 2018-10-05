include("./VidalTEBD.jl")
using .VidalTEBD
using LinearAlgebra

#test make_superpositionMPO
N = 4
U = LinearAlgebra.Diagonal([0,1,0,0])
U2 = ones(4,N)
U1 = reshape(U*U2,2,2,N)
P = [1 for i in 1:N]
MPO = VidalTEBD.make_superpositionMPO(U1,P)

#make all up product state
zeros(Complex{Float64},2,2)
PS = zeros(Float64,(2,N))
PS[1,:] = ones(Float64,N)
MPS = VidalTEBD.make_productVidalMPS(PS,2)

#test do_MPOonMPS
MPS2 = VidalTEBD.do_MPOonMPS(MPS,MPO)

#test convert_to_Vidal
MPS3 = VidalTEBD.convert_to_leftorthogonal(MPS2)
MPS4 = VidalTEBD.convert_to_Vidal(MPS3)
