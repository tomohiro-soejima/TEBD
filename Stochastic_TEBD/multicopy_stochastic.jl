println("including VidalTEBD.jl")
@time include("../VidalTEBD.jl")
using .VidalTEBD
using PyPlot
using LinearAlgebra
using Random
println("finished including different packages")


#initialize
N = 64
cut_position = [25,40]
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
T = 20pi
Nt = round(Int64,T/0.0625)
number_of_data_points = 20
number_of_copies = [1,1,4,16,64]
data_list = zeros(Float64,number_of_data_points+1,5)


for (index, value) in enumerate(number_of_copies)
    println("creating the MPS")
    @time global MPS = initialize_state(N,D)
    println("calculating the Renyi entropy with $value copies")
    Random.seed!(5545343)
    @time data_list[:,index] = VidalTEBD.stochasticTEBD_multicopy(MPS,H,value,T,Nt,number_of_data_points,cut_position)[1]
end

#filename = "TFIM_renyi_multicopy_5.png"


println("plotting figures")
figure()
for (index,value) in enumerate(number_of_copies)
    @time plot((0:number_of_data_points)*T/number_of_data_points,data_list[:,index],marker="o",label = "#copies = $value")
end
xlabel("Time")
ylabel("Renyi 2 entropy (base e)")
axhline(log(4^2),color="k")
axhline(log(2^16),color="k")
legend()
title("D = $D")
#@time savefig(filename)
