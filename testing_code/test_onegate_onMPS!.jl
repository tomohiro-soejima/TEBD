include("../VidalTEBD.jl")
using .VidalTEBD
using BenchmarkTools

zeros(Complex{Float64},2,2)
PS = zeros(Float64,(2,N))
PS[1,:] = ones(Float64,N)
MPS = VidalTEBD.make_productVidalMPS(PS,20)

@btime VidalTEBD.onegate_onMPS1!(MPS,[1.0+0.0im 0;0 -1],3)

zeros(Complex{Float64},2,2)
PS = zeros(Float64,(2,N))
PS[1,:] = ones(Float64,N)
MPS = VidalTEBD.make_productVidalMPS(PS,20)

@btime VidalTEBD.onegate_onMPS2!(MPS,[1.0+0.0im 0;0 -1],3)
