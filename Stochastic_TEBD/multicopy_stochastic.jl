println("including VidalTEBD.jl")
@time include("../VidalTEBD.jl")
using .VidalTEBD
using Plots
using Printf
using Profile
using ProfileView
using LinearAlgebra
using Traceur
using BenchmarkTools
println("finished including different packages")


#initialize
N = 16
cut_position = [7,10]
#these values ensure chaos, following Gil Rafael's convention
hx = 0.9045
hz = 0.8090


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
println("creating the Hamiltonian")
@time H = make_TFIM_H(hx,hz,N)

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


D = 4
T = 10pi
Nt = 100
number_of_data_points = 20
number_of_copies = 20

filename = "TFIM_renyi_multicopy_1.png"

println("creating the MPS")
@time MPS = initialize_state(N,D)
println("calculating the Renyi entropy")
@time data_list = VidalTEBD.stochasticTEBD_multicopy(MPS,H,number_of_copies,T,Nt,number_of_data_points,cut_position)
