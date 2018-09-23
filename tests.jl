include("./normalizeMPS.jl")
using VidalMPSfunctions

zeros(Complex{Float64},2,2)

PS = zeros(Float64,(2,10))
PS[1,:] = ones(Float64,10)
MPS = ProductVidalMPS(PS,10)
