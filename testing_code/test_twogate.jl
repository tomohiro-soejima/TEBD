include("../XYmodel_dynamics/XYmodel_dynamics.jl")
using .dynamics
using Plots
using Printf
using Profile
using ProfileView
using LinearAlgebra
using Traceur

#initialize
N = 101
x0 = 51
sigma = 5
D = 50
T = pi
Nt = 1
h_list = zeros(Float64,N)
a = 1

H = dynamics.xymodel_Hamiltonian(h_list,a)

U = LinearAlgebra.Diagonal([0,1,0,0]) #spin down ladder operator
U2 = ones(4,N)
U1 = reshape(U*U2,2,2,N)
P = [exp(-(xi-x0)^2/(2*sigma^2)) for xi in 1:N]
MPO = dynamics.VidalTEBD.make_superpositionMPO(U1,P)
#make all up product state
zeros(Complex{Float64},2,2)
PS = zeros(Float64,(2,N))
PS[1,:] = ones(Float64,N)
MPS2 = dynamics.VidalTEBD.make_productVidalMPS(PS,D)
#Apply MPO
MPS3 = dynamics.VidalTEBD.do_MPOonMPS(MPS2,MPO)
MPS = dynamics.VidalTEBD.convert_to_Vidal(MPS3)

Profile.clear()
dynamics.VidalTEBD.TEBD!(MPS,H,T,Nt)
@profile expvalues = dynamics.VidalTEBD.TEBD!(MPS,H,T,Nt)
ProfileView.view()
