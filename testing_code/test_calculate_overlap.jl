println("including VidalTEBD.jl")
include("../VidalTEBD.jl")
using .VidalTEBD
using Profile
using ProfileView
using LinearAlgebra
using Random
println("finished including different packages")


#initialize
N = 8
D = 16


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

println("Creating MPS")
MPS = initialize_state(N,D)

println("Calculating overlap")
@time VidalTEBD.calculate_overlap(MPS,MPS,[2,6])
@time VidalTEBD.calculate_overlap(MPS,MPS,[2,6])
Profile.clear()
function run_overlap(MPS)
    for i in 1:50
        VidalTEBD.calculate_overlap(MPS,MPS,[2,6])
    end
end
@time run_overlap(MPS)
@profile run_overlap(MPS)
ProfileView.view()
