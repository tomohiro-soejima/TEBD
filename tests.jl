include("./VidalTEBD.jl")
using .VidalTEBD
using LinearAlgebra
using BenchmarkTools

function Ha(J,h,N)
    OneSite = zeros(Float64,2,2,N)
    for i in 1:N
        OneSite[:,:,i] = h*Sz
    end
    TwoSite =zeros(Float64,4,4,N)
    for i in 1:N-1
        TwoSite[:,:,i] = J*SzSz
    end
    NNQuadHamiltonian(OneSite,TwoSite)
end

zeros(Complex{Float64},2,2)
PS = zeros(Float64,(2,10))
PS[1,:] = ones(Float64,10)
MPS = make_productVidalMPS(PS,10)
Sz = [1 0; 0 -1]
SzSz = LinearAlgebra.Diagonal([1/4,-1/4,-1/4,1/4])

@time TEBD(MPS,Ha(1,0,10),1,10)
@time TEBD(MPS,Ha(1,0,10),1,10)
@time TEBD(MPS,Ha(1,0,10),1,100)
@btime TEBD($MPS,Ha(1,0,10),1,10)
