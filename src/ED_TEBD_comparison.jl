include("./VidalTEBD.jl")
include("./ED.jl")

using .VidalTEBD
using .ED
using LinearAlgebra
using Plots
pyplot()

N = 32
t = 24pi
J = 1
hx = 0.9045
hz = 0.8090
Nt = 200
D = 16

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

#Compare the overlap between ED eigenstate and VidalTEBD state
#=
H = make_TFIM_H(hx,hz,N)
VidalTEBD.TEBD!(MPS,H,t,Nt)
final_state_TEBD = VidalTEBD.create_vector(MPS)
final_state_ED = ED.Ising_time_evolve(J,hx/2,hz/2,PS,t)
print(final_state_TEBD'*final_state_ED)
=#

#=
H = make_TFIM_H(hx,hz,N)
VidalTEBD.TEBD!(MPS,H,t,Nt)
final_state_ED = ED.Ising_time_evolve(J,hx/2,hz/2,PS,t)
println(MPS.Lambda[:,3])
#=
ρ = ED.create_density_matrix(final_state_ED,2,3)
F = eigen(ρ)
println(F.values)
=#
S = ED.do_svd(final_state_ED,2,3)
println(S.S)
=#

function ED_TEBD_compare(N,t,hx,hz,PS,Nt)
    MPS = VidalTEBD.make_productVidalMPS(PS,D)
    H = make_TFIM_H(hx,hz,N)
    plot()
    x = 0:4pi/20:4pi
    ED_renyi = Float64[0]
    TEBD_renyi = Float64[0]

    for i in 1:20
        VidalTEBD.TEBD!(MPS,H,t/20,Nt/20)
        final_state_ED = ED.Ising_time_evolve(J,hx/2,hz/2,PS,i*t/20)
        ED_lambda = ED.do_svd(final_state_ED,2,4).S
        push!(ED_renyi,-log(sum(ED_lambda.^4)))
        TEBD_lambda = MPS.Lambda[:,5]
        push!(TEBD_renyi,-log(sum(TEBD_lambda.^4)))
    end
    plot!(x,ED_renyi,label = "ED")
    plot!(x,TEBD_renyi,label="TEBD")
    xlabel!("Time")
    ylabel!("Renyi entropy")
    savefig("Renyi_ED_TEBD_2.png")
end

#ED_TEBD_compare(N,t,hx,hz,PS,Nt)

function TEBD_renyi(N,t,hx,hz,PS,Nt)
    MPS = VidalTEBD.make_productVidalMPS(PS,D)
    H = make_TFIM_H(hx,hz,N)
    plot()
    x = 0:12pi/40:12pi
    TEBD_renyi = Float64[0]
    #renyivalue = VidalTEBD.TEBDwithRenyi!(MPS,H,t,Nt,4,2)
    #t = 0:4pi/Nt:4pi
    for i in 1:40
        VidalTEBD.TEBD!(MPS,H,t/40,Nt/40)
        TEBD_lambda = MPS.Lambda[:,17]
        push!(TEBD_renyi,-log(sum(TEBD_lambda.^4)))
    end
    plot!(x,TEBD_renyi,label="TEBD")
    xlabel!("Time")
    ylabel!("Renyi entropy")
    savefig("Renyi_TEBD_4.png")
    println(TEBD_renyi)

    #=plot(t,renyivalues,label="TEBD")
    xlabel!("Time")
    ylabel!("Renyi entropy")
    savefig("Renyi_TEBD_11.png")
    println(renyivalues)
    =#
end

@time TEBD_renyi(N,t,hx,hz,PS,Nt)
