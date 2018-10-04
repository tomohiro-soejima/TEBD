module dynamics

include("../VidalTEBD.jl")
using .VidalTEBD
using LinearAlgebra

function xymodel_dynamics(N,x0,sigma,D,T,Nt,h_list,alpha,O)
    #implements the dynamics of a Hamiltonian H = h_i S_i + (Sx_i Sx_i+1 + Sy_i Sy_i+1 + alpha Sz_i Sz_i+1)
    H = xymodel_Hamiltonian(h_list,alpha)
    MPS = create_excited_state(N,x0,sigma,D)
    VidalTEBD.getTEBDexpvalue!(MPS,H,T,Nt,O)
end

function xymodel_Hamiltonian(h_list,a)
    #create Hamiltonian Hamiltonian H = h_i S_i + (Sx_i Sx_i+1 + Sy_i Sy_i+1 + alpha Sz_i Sz_i+1)
    N = size(h_list)[1]
    OneSite = zeros(Float64,4,N)
    OneSite[4,:] = h_list #h_i at each site
    TwoSite = zeros(Float64,3,3,N)
    TwoSite[1,1,1:N-1] = ones(Float64,N-1) #SxSx
    TwoSite[2,2,1:N-1] = ones(Float64,N-1) #SySy
    TwoSite[3,3,1:N-1] = a .* ones(Float64,N-1) #SzSz
    H = VidalTEBD.NNSpinHalfHamiltonian(OneSite,TwoSite)
    VidalTEBD.makeNNQuadH(H)
end

function ising_Hamiltonian(h_list,J_list)
    #create Hamiltonian Hamiltonian H = h_i S_i + J_i Sz_i Sz_i+1
    N = size(h_list)[1]
    OneSite = zeros(Float64,4,N)
    OneSite[4,:] = h_list #h_i at each site
    TwoSite = zeros(Float64,3,3,N)
    TwoSite[3,3,1:N-1] = J_list #SzSz
    H = VidalTEBD.NNSpinHalfHamiltonian(OneSite,TwoSite)
    VidalTEBD.makeNNQuadH(H)
end

function create_excited_state(N,x0,sigma,D)
    #create an MPO
    U = LinearAlgebra.Diagonal([0,1,0,0]) #spin down ladder operator
    U2 = ones(4,N)
    U1 = reshape(U*U2,2,2,N)
    P = [exp(-(xi-x0)^2/(2*sigma^2)) for xi in 1:N]
    MPO = VidalTEBD.make_superpositionMPO(U1,P)
    #make all up product state
    zeros(Complex{Float64},2,2)
    PS = zeros(Float64,(2,N))
    PS[1,:] = ones(Float64,N)
    MPS = VidalTEBD.make_productVidalMPS(PS,D)
    #Apply MPO
    MPS2 = VidalTEBD.do_MPOonMPS(MPS,MPO)
    VidalTEBD.convert_to_Vidal(MPS2)
end

function excited_state_dynamics(N,x0,sigma, D,T,Nt,H,O)
    MPS = create_excited_state(N,x0,sigma,D)
    VidalTEBD.getTEBDexpvalue!(MPS,H,T,Nt,O)
end

#this end is for module
end
