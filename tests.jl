include("./VidalTEBD.jl")
using .VidalTEBD
using LinearAlgebra
using BenchmarkTools
using Profile
using Traceur
using Plots

function Ha(J,h,N)
    Sz = [1 0; 0 -1]
    SzSz = LinearAlgebra.Diagonal([1/4,-1/4,-1/4,1/4])
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
PS[:,5] = [1/sqrt(2),1/sqrt(2)]
MPS = make_productVidalMPS(PS,10)

TEBD!(MPS,Ha(1,0,10),1,10)
@time TEBD!(MPS,Ha(1,0,10),1,10)
@time TEBD!(MPS,Ha(1,0,10),10*pi,100)

Sz = [0 1;1 0]
U = zeros(Float64,2,2,10)
for i in 1:10
    U[:,:,i] = Sz
end



#=
expvalue =  @time getTEBDexpvalue!(MPS,Ha(1,0,10),10*pi,1000,U)
x = 1:1001
plot(x,real(expvalue))
savefig("Isingplot.png")=#

MPS = make_productVidalMPS(PS,1000)
@time onesite_expvalue1(MPS,U[:,:,1],1)
@time onesite_expvalue2(MPS,U[:,:,1],1)
#removing for loop from onesite_expvalue made it slightly faster
