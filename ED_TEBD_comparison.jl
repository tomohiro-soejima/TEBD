include("./VidalTEBD.jl")
include("./ED.jl")

using .VidalTEBD
using .ED

N = 8
t = 30
J = 1
hx = 0.9045
hz = 0.8090
Nt = 3000
D = 64

#prepare the initial state used by Gil Rafael
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

MPS = VidalTEBD.make_productVidalMPS(PS,D)

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

VidalTEBD.TEBD!(MPS,H,t,Nt)
final_state_TEBD = VidalTEBD.create_vector(MPS)
final_state_ED = ED.Ising_time_evolve(J,hx/2,hz/2,PS,t)

print(final_state_TEBD'*final_state_ED)
