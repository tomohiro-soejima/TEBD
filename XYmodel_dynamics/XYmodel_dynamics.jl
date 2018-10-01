include("./VidalTEBD.jl")
using .VidalTEBD
using LinearAlgebra

function excited_state_dynamics(N,D,T,Nt,H,U)
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
    MPS3 = VidalTEBD.convert_to_Vidal(MPS2)

    VidalTEBD.getTEBDexpvalue!(MPS3,H,T,Nt,U)
end

function xymodel_dynamics(N,D,T,Nt,h_list,alpha,U)
    #implements the dynamics of a Hamiltonian H = h_i S_i + (Sx_i Sx_i+1 + Sy_i Sy_i+1 + alpha Sz_i Sz_i+1)
    H = xymodel_Hamiltonian(h_list,alpha)
    excited_state_dynamics(N,D,T,Nt,H,U)
end

function xymodel_Hamiltonian(h_list,alpha)
    N = size(h_list)[1]
    OneSite = zeros(Float64,4,N)
    OneSite[4,:] = h_list
    TwoSite = zeros(Float64,4,4,N) #h_i at each site
    TwoSite[2,2,1:N-1] = ones(Float64,N-1) #SxSx
    TwoSite[3,3,1:N-1] = ones(Float64,N-1) #SySy
    TwoSite[4,4,1:N-1] = alpha.*ones(Float64,N-1) #SzSz
    H = VidalTEBD.NNSpinHalfHamiltonian(OneSite,TwoSite)
    VidalTEBD.makeNNQuadH(H)
end
