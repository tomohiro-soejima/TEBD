using BenchmarkTools
using LinearAlgebra

struct VidalMPS
    Gamma::Array{Float64,4}
    Lambda::Array{Float64,2}
end

#=
struct VidalMPS{D::Int64,d::Int64,N::Int64}
    Gamma::Array{Float64,4}(D,D,d,N)
    Lambda::Array{Float64,3}(D,N+1)
end
=#

function ProductVidalMPS(ProductState,D)
    d, N = size(ProductState)
    Gamma = zeros(Float64,D,D,d,N)
    Lambda = zeros(Float64,D,N+1)
    Gamma[1,1,:,:] = ProductState
    Lambda[1,:] = ones(Float64,N+1)
    VidalMPS(Gamma,Lambda)
end

function OneGateOnMPS(MPS::VidalMPS,U,loc)
    D,D2,d,N = size(MPS.Gamma)
    R = PermutedDimsArray(reshape(view(MPS.Gamma,:,:,:,loc),D^2,d),(2,1))
    R[:,:] = U*R
end

function OneGateOnMPSCopy(MPS::VidalMPS,U,loc)
    #To figure out whether in place multiplication is faster
    D,D2,d,N = size(MPS.Gamma)
    R = PermutedDimsArray(reshape(view(MPS.Gamma,:,:,:,loc),D^2,d),(2,1))
    R[:,:] = U*R
end

function TwoGateOnMPS(MPS::VidalMPS,U,loc)
    D,D2,d,N = size(MPS.Gamma)
    L1 = view(MPS.Lambda,:,loc)
    L2 = view(MPS.Lambda,:,loc+1)
    L3 = view(MPS.Lambda,:,loc+2)
    Gamma1 = view(MPS.Gamma,:,:,:,loc)
    Gamma2 = view(MPS.Gamma,:,:,:,loc+1)
    S1 = zeros(Float64,d*D,D)
    S2 = zeros(Float64,D,d*D)
    for i in 1:d
        S1[(i-1)*D+1:i*D,:] = Diagonal(L1)*view(Gamma1,:,:,i)*Diagonal(L2)
    end
    for j in 1:d
        S2[:,(j-1)*D+1:j*D] = view(Gamma2,:,:,j)*Diagonal(L3)
    end
    theta = reshape(PermutedDimsArray(reshape(S1*S2,D,d,D,d),(3,1,4,2)),d^2,D^2)
    thetaNew = reshape(PermutedDimsArray(reshape(U*theta,d,d,D,D),(2,4,1,3)),(d*D,d*D))
    F = LinearAlgebra.svd(copy(thetaNew))
    @views GL1 = PermutedDimsArray(reshape(F.U[:,1:D],D,d,D),(1,3,2))
    @views GL2 = reshape(F.Vt[1:D,:],D,D,d)
    print(GL1)
    for i in 1:D
        if L1[i] != 0
            Gamma1[i,:,:] = GL1[i,:,:]/L1[i]
        end
    end
    @views L2[:] = F.S[1:D]/sum(F.S[1:D])
    for j in 1:D
        if L3[j] != 0
            Gamma2[:,j,:] = GL2[:,j,:]/L3[j]
        end
    end
    F.U
end


#test
PS = zeros(Float64,2,4)
PS[1,:] = ones(Float64,4)
MPS = ProductVidalMPS(PS,2)

#one site operation
U = [1 1;1 -1]/sqrt(2)
OneGateOnMPS(MPS,U,2)

#two site operation
U2 = [1/sqrt(2) 0 0 1/sqrt(2);0 1 0 0; 0 0 1 0; 1/sqrt(2) 0 0 -1/sqrt(2)]
U = TwoGateOnMPS(MPS,U2,3)
