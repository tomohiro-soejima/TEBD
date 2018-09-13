using BenchmarkTools
using LinearAlgebra

struct VidalMPS
    Gamma::Array{Complex{Float64},4}
    Lambda::Array{Float64,2}
end

struct NNQuadHamiltonian
    #OneSite[i] is on site term at site i
    OneSite::Array{Hermitian{Complex{Float64},Array{Complex{Float64},2}},1}
    #TwoSite[i] is two-site term at i and i+1
    TwoSite::Array{Hermitian{Complex{Float64},Array{Complex{Float64},2}},1}
end

struct NNQuadUnitary
    #OneSite[:,:,i] is on site term at site i
    OneSite::Array{Complex{Float64},3}
    #TwoSite[i] is two-site term at i and i+1
    TwoSite::Array{Complex{Float64},3}
end

struct NNSpinHalfHamiltonian
    OneSite::Array{Float64,2}
    TwoSite::Array{Float64,3}
end

function ProductVidalMPS(ProductState,D)
    d, N = size(ProductState)
    Gamma = zeros(Complex{Float64},D,D,d,N)
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
    #turns out it is not much faster
    D,D2,d,N = size(MPS.Gamma)
    R = PermutedDimsArray(reshape(view(MPS.Gamma,:,:,:,loc),D^2,d),(2,1))
    R[:,:] = U*R
end

function OneSiteExpValue(MPS::VidalMPS,U,loc)
    #not working yet!
    D,D2,d,N = size(MPS.Gamma)
    L1 = view(MPS.Lambda,:,loc)
    L2 = view(MPS.Lambda,:,loc+1)
    Gamma1 = view(MPS.Gamma,:,:,:,loc)
    S = zeros(Complex{Float64},D,D,d)
    for i in 1:d
        S[:,:,i] = Diagonal(L1)*Gamma1[:,:,i]*Diagonal(L2)
    end
    K = PermutedDimsArray(reshape(S,D^2,d),(2,1))
    L = U*K
    sum(L .* conj.(K))
end

function TwoGateOnMPS(MPS::VidalMPS,U,loc)
    thetaNew = Theta_ij(MPS,U,loc)
    F = LinearAlgebra.svd(copy(thetaNew))
    UpdateMPSafterTwoGate(MPS,F,loc)
end

function Theta_ij(MPS::VidalMPS,U,loc)
    D,D2,d,N = size(MPS.Gamma)
    L1 = view(MPS.Lambda,:,loc)
    L2 = view(MPS.Lambda,:,loc+1)
    L3 = view(MPS.Lambda,:,loc+2)
    Gamma1 = view(MPS.Gamma,:,:,:,loc)
    Gamma2 = view(MPS.Gamma,:,:,:,loc+1)
    S1 = zeros(Complex{Float64},d*D,D)
    S2 = zeros(Complex{Float64},D,d*D)
    for i in 1:d
        S1[(i-1)*D+1:i*D,:] = Diagonal(L1)*view(Gamma1,:,:,i)*Diagonal(L2)
    end
    for j in 1:d
        S2[:,(j-1)*D+1:j*D] = view(Gamma2,:,:,j)*Diagonal(L3)
    end
    theta = reshape(PermutedDimsArray(reshape(S1*S2,D,d,D,d),(2,4,1,3)),d^2,D^2)
    reshape(PermutedDimsArray(reshape(U*theta,d,d,D,D),(3,1,4,2)),(d*D,d*D))
end

function UpdateMPSafterTwoGate(MPS::VidalMPS,F,loc)
    D,D2,d,N = size(MPS.Gamma)
    L1 = view(MPS.Lambda,:,loc)
    L2 = view(MPS.Lambda,:,loc+1)
    L3 = view(MPS.Lambda,:,loc+2)
    Gamma1 = view(MPS.Gamma,:,:,:,loc)
    Gamma2 = view(MPS.Gamma,:,:,:,loc+1)
    @views GL1 = PermutedDimsArray(reshape(F.U[:,1:D],D,d,D),(1,3,2))
    @views GL2 = reshape(F.Vt[1:D,:],D,D,d)
    for i in 1:D
        if L1[i] != 0
            Gamma1[i,:,:] = GL1[i,:,:]/L1[i]
        end
    end
    @views L2[:] = F.S[1:D]/sqrt(sum(F.S[1:D].^2))
    for j in 1:D
        if L3[j] != 0
            Gamma2[:,j,:] = GL2[:,j,:]/L3[j]
        end
    end
end

function TEBD(MPS::VidalMPS,H::NNQuadHamiltonian,T,N)
    del = T/N
    U = makeNNQuadUnitary(H,del::Float64)
    for i in 1:N
        OddSiteUpdate(MPS,U)
        EvenSiteUpdate(MPS,U)
    end
end

function OddSiteUpdate(MPS::VidalMPS,U::NNQuadUnitary)
    D,D2,d,N = size(MPS.Gamma)
    for loc in 1:2:N-1
        OneGateOnMPS(MPS,U.OneSite[:,:,loc],loc)
        TwoGateOnMPS(MPS,U.TwoSite[:,:,loc],loc)
    end
    if N%2 == 1
        OneGateOnMPS(MPS,U.OneSite[:,:,N],N)
    end
end

function EvenSiteUpdate(MPS::VidalMPS,U::NNQuadUnitary)
    D,D2,d,N = size(MPS.Gamma)
    for loc in 2:2:N-1
        OneGateOnMPS(MPS,U.OneSite[:,:,loc],loc)
        TwoGateOnMPS(MPS,U.TwoSite[:,:,loc],loc)
    end
    if N%2 == 0
        OneGateOnMPS(MPS,U.OneSite[:,:,N],N)
    end
end

function makeNNQuadUnitary(H::NNQuadHamiltonian,del::Float64)
    N = size(H.OneSite)[1]
    d = size(H.OneSite[1])[1]
    OneSite = zeros(Complex{Float64},d,d,N)
    TwoSite = zeros(Complex{Float64},d^2,d^2,N)
    for i in 1:N
        OneSite[:,:,i] = exp(-im*del*H.OneSite[i])
    end
    for j in 1:N-1
        TwoSite[:,:,j] = exp(-im*del*H.TwoSite[j])
    end
    NNQuadUnitary(OneSite,TwoSite)
end

function makeNNQuadH(H::NNSpinHalfHamiltonian)
    N = size(H.OneSite)[1]
    I = [1 0;0 1]
    Sx = [0 1;1 0]
    Sy = [0 -im;im 0]
    Sz = 1/2*[1 0, 0 -1]
    Ops = [I,Sx,Sy,Sz]
    OneSiteOp = zeros(Complex{Float64},2,2,4)
    for i in 1:4
        OneSiteOp[:,:,i] = Ops[i]
    end
    OneSiteOpVec = PermutedDimsArray(reshape(OneSiteOp,4,4),(2,1))
    SS = zeros(Complex{Float64},2,2,2,2,3,3)
    SS[:,:,:,:,:,:] = [OneSiteOp[i1,i2,m]*OneSiteOp[j1,j2,n]
                    for i1 in 1:2, j1 in 1:2, i2 in 1:2, j2 in 1:2
                        m in 2:4, n in 2:4]
    TwoSiteOpVec = PermutedDimsArray(reshape(SS,16,9),(2,1))
    OneSite = reshape(PermutedDimsArray(transpose(H.OneSite)*OneSiteOpVec,(2,1)),2,2,N)
    TwoSite2 = reshape(H.TwoSite, 16, N)
    TwoSite = reshape(PermutedDimsArray(transpose(TwoSite2)*TwoSiteOpVec,(2,1)),4,4,N)
    NNQuadHamiltonian(OneSite,TwoSite)
end


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

zeros(Complex{Float64},2,2)

PS = zeros(Float64,(2,10))
PS[1,:] = ones(Float64,10)
MPS = ProductVidalMPS(PS,10)

Sz = [1 0; 0 -1]
SzSz = Diagonal([1/4,-1/4,-1/4,1/4])
Ha(J,h,N) = NNQuadHamiltonian([Hermitian(h*Sz) for i in 1:N],
                            [Hermitian(J*SzSz) for i in 1:N-1])
TEBD(MPS,Ha(1,0,10),1,10)
