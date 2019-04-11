using Test

include("../VidalTEBD.jl")
using .VidalTEBD
using LinearAlgebra

@testset "VidalMPS" begin
    #test normalization
    D = 20
    N = 20
    d = 2
    PS = zeros(Complex{Float64},d,N)
    PS[2,:] = ones(eltype(PS),N)
    MPS = VidalTEBD.make_productVidalMPS(PS,D)
    @test VidalTEBD.isnormalized(MPS)

    #test normalization with complex numbers
    D = 20
    N = 20
    PS = zeros(Complex{Float64},2,N)
    PS[2,:] = im*ones(eltype(PS),N)
    MPS = VidalTEBD.make_productVidalMPS(PS,D)
    @test VidalTEBD.isnormalized(MPS)
    @test VidalTEBD.isleftorthogonal(MPS, 10)
    @test VidalTEBD.isleftorthogonal(MPS, 2)
    @test VidalTEBD.isrightorthogonal(MPS, 10)
    @test VidalTEBD.isrightorthogonal(MPS, 18)
end


@testset "TEBD on VidalMPS" begin
    D = 4
    N = 20
    hx = 0.5
    hz = 1.0
    T = pi
    Nt = 5
    PS = zeros(Complex{Float64},2,N)
    PS[2,:] = im*ones(eltype(PS),N)
    MPS = VidalTEBD.make_productVidalMPS(PS,D)
    H = VidalTEBD.make_TFIM_H(hx,hz,N)
    VidalTEBD.TEBD!(MPS, H, T, Nt)
    @test VidalTEBD.isnormalized(MPS)
    @test VidalTEBD.isleftorthogonal(MPS, 10)
    @test VidalTEBD.isleftorthogonal(MPS, 2)
    @test VidalTEBD.isrightorthogonal(MPS, 10)
    @test VidalTEBD.isrightorthogonal(MPS, 18)
end
