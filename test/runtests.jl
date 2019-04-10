using Test

include("../VidalTEBD.jl")
using .VidalTEBD
using LinearAlgebra

@testset "VidalMPS" begin

    #test normalization
    D = 20
    N = 20
    PS = zeros(Complex{Float64},2,N)
    PS[2,:] = Ones(eltype(PS),N)
    VidalTEBD.make_productVidalMPS(PS,D)
    

end
