include("./VidalTEBD.jl")
using .VidalTEBD

#=test for Updates
PS = zeros(Float64,2,4)
PS[1,:] = ones(Float64,4)
MPS = ProductVidalMPS(PS,5)

#one site operation
U = [1 1;1 -1]/sqrt(2)
OneGateOnMPS(MPS,U,2)

#two site operation
U2 = [1/sqrt(2) 0 0 1/sqrt(2);0 1 0 0; 0 0 1 0; 1/sqrt(2) 0 0 -1/sqrt(2)]
TwoGateOnMPS(MPS,U2,3)
=#

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
SzSz = Diagonal([1/4,-1/4,-1/4,1/4])

@time TEBD(MPS,Ha(1,0,10),1,10)
@time TEBD(MPS,Ha(1,0,10),1,10)
