#=
This is to reproduce the result of DMT paper by Gil Rafael et al.
=#

include("../VidalTEBD.jl")
using .VidalTEBD
using Plots
using Printf
using Profile
using ProfileView
using LinearAlgebra
using Traceur
using BenchmarkTools

filename = "TFIM_TEBD_E_D_16_stochastic"


#initialize
N = 64
#these values ensure chaos, following Gil Rafael's convention
hx = 0.9045
hz = 0.8090

D = 16
T = 200
Nt = round(Int64,T/0.0625)

#make Ising Hamiltonian
function make_TFIM_H(hx,hz,N)
    OneSite = zeros(Float64,4,N)
    OneSite[2,:] = hx/2 .* ones(Float64,N)
    OneSite[4,:] = hz/2 .* ones(Float64,N) #h_i at each site
    TwoSite = zeros(Float64,3,3,N)
    TwoSite[3,3,1:N-1] = ones(Float64,N-1) #SzSz
    H = VidalTEBD.NNSpinHalfHamiltonian(OneSite,TwoSite)
    VidalTEBD.makeNNQuadH(H)
end
H = make_TFIM_H(hx,hz,N)

#make all up product state
#there is a mistake in the paper. how should I fix it?
function initialize_state(N,D)
    PS = zeros(Complex{Float64},2,N)
    down_site = [im*0.9, 1]/sqrt(0.9^2 + 1)
    up_site = [im*1.1, 1]/sqrt(1.1^2 + 1)
    for i in 1:N
        if i%8 in [1,2,7,0]
            PS[:,i] = down_site
        elseif i%8 in [3,4,5,6]
            PS[:,i] = up_site
        end
    end
    VidalTEBD.make_productVidalMPS(PS,D)
end

#=
#filename
filename = "TFIM_TEBD"
MPS = initialize_state(N,D)
renyivalue = @time VidalTEBD.TEBDwithRenyi!(MPS,H,T,Nt,32,2)
x = 1:(Nt+1)
plot(x,renyivalue)
savefig(filename*".png")
=#

function makeMPOforTHIM(hx,hz,N)
    d = 2
    M1 = [1,0,0]
    Mend = [0,0,1]
    M = zeros(Complex{Float64},3,3,d,d)
    M[1,1,:,:] = Matrix{Complex{Float64}}(I,d,d)
    M[1,2,:,:] = [1/2 0;0 -1/2]
    M[1,3,:,:] = hx/2*[0 1/2;1/2 0]+hz/2*[1/2 0;0 -1/2]
    M[2,3,:,:] = [1/2 0;0 -1/2]
    M[3,3,:,:] = Matrix{Complex{Float64}}(I,d,d)
    Mall = zeros(Complex{Float64},3,3,d,d,N)
    for i in 1:N
        Mall[:,:,:,:,i] = M
    end
    VidalTEBD.MatrixProductOperator(M1,Mall,Mend)
end




MPS = initialize_state(N,D)
MPO = makeMPOforTHIM(hx,hz,N)
Profile.clear()
energyvalue = @profile VidalTEBD.stochasticTEBD!(MPS,H,T,Nt,MPO = MPO)
x = (1:(Nt+1))*0.0625
plot(x,real(energyvalue))
savefig(filename*".png")
