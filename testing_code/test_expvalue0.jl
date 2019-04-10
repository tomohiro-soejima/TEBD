#=
Test  whether my contract is slow or not =#

include("../VidalTEBD.jl")
using .VidalTEBD
using LinearAlgebra
using BenchmarkTools
using Profile
using Traceur
using Plots

A = rand(2,300,300,2)
B = rand(2,2,150,5,60,2)

function contract_ex(A,B)
    A2 = reshape(PermutedDimsArray(A,(1,3,4,2)),1200,300)
    B2 = reshape(PermutedDimsArray(B,(2,3,1,4,5,6)),300,1200)
    reshape(A2*B2,2,300,2,2,5,60,2)
end

VidalTEBD.contract(A,[2],B,[2,3])
@profile VidalTEBD.contract(A,[2],B,[2,3])
@time VidalTEBD.contract(A,[2],B,[2,3])
@code_warntype VidalTEBD.contract(A,[2],B,[2,3])
@trace VidalTEBD.contract(A,[2,],B,[2,3])
#@btime VidalTEBD.contract($A,[2],$B,[2,3])
contract_ex(A,B)
#@btime contract_ex($A,$B)

function contract_ex2(A,B)
    A2 = reshape(A,1,360000)
    B2 = reshape(B,360000,1)
    A2*B2[1]
end

#@btime VidalTEBD.contract($A,[1,2,3,4],$B,[1,2,3,4,5,6])
contract_ex2(A,B)
#@btime contract_ex2($A,$B)
