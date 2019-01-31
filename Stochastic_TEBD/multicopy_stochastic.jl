include("../VidalTEBD.jl")
using .VidalTEBD
using Plots
using Printf
using Profile
using ProfileView
using LinearAlgebra
using Traceur
using BenchmarkTools



#initialize
N = 16
#these values ensure chaos, following Gil Rafael's convention
hx = 0.9045
hz = 0.8090

D = 16
T = 4pi
Nt = round(Int64,T/0.0625)
number_of_data_points = 20
number_of_copies = 10

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

filename = "TFIM_TEBD_Renyi_D_64_2"

MPS = initialize_state(N,D)
data_list = @profile VidalTEBD.stochasticTEBD_multicopy(MPS,H,number_of_copies,T,Nt,number_of_data_points)

#=
ProfileView.view()
x = (1:(Nt+1))*0.0625
plot(x,renyivalue)
savefig(filename*".png")

filename = "TFIM_TEBD_Renyi_D_64_stochastic_2"
=#


#=
MPS = initialize_state(N,D)
Profile.clear()
renyivalue = @profile VidalTEBD.stochasticTEBDwithRenyi!(MPS,H,T,Nt,32,2)
ProfileView.view()
x = (1:(Nt+1))*0.0625
plot(x,renyivalue)
savefig(filename*".png")
=#
