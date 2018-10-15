module VidalTEBD

export VidalMPS, NNSpinHalfHamiltonian, NNQuadHamiltonian, NNQuadUnitary
export make_productVidalMPS, onesite_expvalue,onesite_expvalue1,onesite_expvalue2, TEBD!, makeNNQuadH, getTEBDexpvalue!,getTEBDexpvaluecopy!
export contract

using BenchmarkTools
using LinearAlgebra

GammaMat = Array{Complex{Float64},4}

struct VidalMPS
    #MPS in Vidal canonical form
    Gamma::Array{Complex{Float64},4} #dim = D,D,d,N
    Lambda::Array{Float64,2} #dim = D,N
end
struct OrthogonalMPS
    #MPS in Left orthogonal and right orthogonal form. The location of orthogonality center is specified.
    Gamma::Array{Complex{Float64},4} #dim = D,D,d,N
    Loc_OrtCenter::Int
end
struct LeftOrthogonalMPS
    #MPS in Left orthogonal
    Gamma::Array{Complex{Float64},4} #dim = D,D,d,N
end
struct GenericMPS
    # most generic MPS without any orthogonality condition imposed
    Gamma::Array{Complex{Float64},4} #dim =D,D,d,N
end
struct NNQuadHamiltonian
    #OneSite[i] is on site term at site i
    OneSite::Array{Complex{Float64},3} #dim = d,d,N
    #TwoSite[i] is two-site term at i and i+1
    TwoSite::Array{Complex{Float64},3} #dim = d^2,d^2,N
end
struct NNQuadUnitary
    #OneSite[:,:,i] is on site term at site i
    OneSite::Array{Complex{Float64},3} #dim =d,d,N
    #TwoSite[i] is two-site term at i and i+1
    TwoSite::Array{Complex{Float64},3} #dim = d^2,d^2,N
end
struct NNSpinHalfHamiltonian
    OneSite::Array{Float64,2}
    TwoSite::Array{Float64,3}
end
struct MatrixProductOperator
    M1::Array{Complex{Float64},1} #dim = D
    M::Array{Complex{Float64},5} #dim = D,D,d,d,N
    Mend::Array{Complex{Float64},1} #dim = D
end

function make_productVidalMPS(ProductState,D)
    d, N = size(ProductState)
    Gamma = zeros(Complex{Float64},D,D,d,N)
    Lambda = zeros(Float64,D,N+1)
    Gamma[1,1,:,:] = ProductState
    Lambda[1,:] = ones(Float64,N+1)
    VidalMPS(Gamma,Lambda)
end
function make_biggerMPS(MPS::VidalMPS,D_new)
    #enlarge the bond dimension
    #=Maybe it is better to append zeros to original lattice?
    so that it can modify the original MPS?=#
    D_old, D_old2, d, N = size(MPS.Gamma)
    Gamma = MPS.Gamma[:,:,:,:]
    Lambda = MPS.Lambda[:,:]
    GammaNew = zeros(Complex{Float64},D_new,D_new,d,N)
    GammaNew[1:D_old,1:D_old,d,N] = Gamma
    LambdaNew = zeros(Complex{Float64}, D_new,N)
    LambdaNew[1:D_old,N] = Lambda
    VidalMPS(GammaNew,LambdaNew)
end

function onegate_onMPS!(MPS::VidalMPS,U,loc)
    onegate_onMPS2!(MPS::VidalMPS,U,loc)
end

function onegate_onMPS1!(MPS::VidalMPS,U,loc)
    D,D2,d,N = size(MPS.Gamma)
    R = PermutedDimsArray(reshape(view(MPS.Gamma,:,:,:,loc),D^2,d),(2,1))
    R[:,:] = U*R
end

function onegate_onMPS2!(MPS::VidalMPS,U::Array{Complex{Float64},2},loc::Int64)
    D,D2,d,N = size(MPS.Gamma)
    R = permutedims(reshape(view(MPS.Gamma,:,:,:,loc),D^2,d),(2,1))
    MPS.Gamma[:,:,:,loc] = reshape(PermutedDimsArray(U*R,(2,1)),D,D,d)
    #=Lambdap = MPS.Lambda
    Gammap = Array{Complex{Float64},2}(undef,D^2,d)
    mul!(Gammap,U,R)
    Gammap2 = reshape(Gammap,D,D,d)

    return VidalMPS(Gammap,Lambdap)=#
end

function onesite_expvalue(MPS::VidalMPS,U,loc)
    onesite_expvalue2(MPS,U,loc)
end
function onesite_expvalue1(MPS::VidalMPS,U,loc)
    #not working yet!
    @views D,D2,d,N = size(MPS.Gamma)
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
function onesite_expvalue2(MPS::VidalMPS,U,loc)
    @views D,D2,d,N = size(MPS.Gamma)
    L1 = view(MPS.Lambda,:,loc)
    L2 = view(MPS.Lambda,:,loc+1)
    Gamma1 = view(MPS.Gamma,:,:,:,loc)
    Gamma1 = Diagonal(L1) * reshape(Gamma1,D,D*d)
    Gamma1 = Diagonal(L2) * reshape(PermutedDimsArray(reshape(Gamma1,D,D,d),(2,1,3)),D,D*d)
    K = PermutedDimsArray(reshape(Gamma1,D^2,d),(2,1))
    L = U*K
    sum(L .* conj.(K))
end
function twogate_onMPS!(MPS::VidalMPS,U,loc)
    thetaNew = theta_ij(MPS,U,loc)
    F = LinearAlgebra.svd(copy(thetaNew))
    updateMPSafter_twogate!(MPS,F,loc)
end
function theta_ij(MPS::VidalMPS,U,loc)
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
function updateMPSafter_twogate!(MPS::VidalMPS,F,loc)
    D,D2,d,N = size(MPS.Gamma)
    L1 = view(MPS.Lambda,:,loc)
    L2 = view(MPS.Lambda,:,loc+1)
    L3 = view(MPS.Lambda,:,loc+2)
    Gamma1 = view(MPS.Gamma,:,:,:,loc)
    Gamma2 = view(MPS.Gamma,:,:,:,loc+1)
    @views GL1 = PermutedDimsArray(reshape(F.U[:,1:D],D,d,D),(1,3,2))
    @views GL2 = reshape(F.Vt[1:D,:],D,D,d)

    L1_inv = zero(L1)
    for i in 1:D
        if L1[i] > 10^-6
            L1_inv[i] = 1/L1[i]
        end
    end
    Gamma1[:,:,:] = contract(Diagonal(L1_inv),[2],GL1,[1])

    S = zeros(Float64,D)
    for i in 1:D
        if F.S[i] > 10^-6
            S[i] = F.S[i]
        else
            S[i] = 0 #somehow if I make this 10^-6 my code explods
        end
    end
    @views L2[:] = S[:]/sqrt(sum(S[:].^2))
    #@views L2[:] = F.S[1:D]/sqrt(sum(F.S[1:D].^2))
    L3_inv = zero(L3)
    for i in 1:D
        if L3[i] >10^-6
            L3_inv[i] = 1/L3[i]
        end
    end
    Gamma2[:,:,:]= permutedims(contract(GL2,[2],Diagonal(L3_inv),[1]),(1,3,2))
end
function TEBD!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N)
    del = T/N
    U = makeNNQuadUnitary(H,del::Float64)
    for i in 1:N
        update_oddsite!(MPS,U)
        update_evensite!(MPS,U)
    end
end
function update_oddsite!(MPS::VidalMPS,U::NNQuadUnitary)
    D,D2,d,N = size(MPS.Gamma)
    for loc in 1:2:N-1
        onegate_onMPS!(MPS,U.OneSite[:,:,loc],loc)
        twogate_onMPS!(MPS,U.TwoSite[:,:,loc],loc)
    end
    if N%2 == 1
        onegate_onMPS!(MPS,U.OneSite[:,:,N],N)
    end
end
function update_evensite!(MPS::VidalMPS,U::NNQuadUnitary)
    D,D2,d,N = size(MPS.Gamma)
    for loc in 2:2:N-1
        onegate_onMPS!(MPS,U.OneSite[:,:,loc],loc)
        twogate_onMPS!(MPS,U.TwoSite[:,:,loc],loc)
    end
    if N%2 == 0
        onegate_onMPS!(MPS,U.OneSite[:,:,N],N)
    end
end
function makeNNQuadUnitary(H::NNQuadHamiltonian,del::Float64)
    d,d1,N = size(H.OneSite)
    OneSite = zeros(Complex{Float64},d,d,N)
    TwoSite = zeros(Complex{Float64},d^2,d^2,N)
    for i in 1:N
        OneSite[:,:,i] = exp(-im*del*Hermitian(H.OneSite[:,:,i]))
    end
    for j in 1:N-1
        TwoSite[:,:,j] = exp(-im*del*Hermitian(H.TwoSite[:,:,j]))
    end
    NNQuadUnitary(OneSite,TwoSite)
end
function makeNNQuadH(H::NNSpinHalfHamiltonian)
    d, N = size(H.OneSite)
    I = [1 0;0 1]
    Sx = 1/2*[0 1;1 0]
    Sy = 1/2*[0 -im;im 0]
    Sz = 1/2*[1 0; 0 -1]
    Ops = [I,Sx,Sy,Sz]
    OneSiteOp = zeros(Complex{Float64},2,2,4)
    for i in 1:4
        OneSiteOp[:,:,i] = Ops[i]
    end
    OneSiteOpVec = PermutedDimsArray(reshape(OneSiteOp,4,4),(2,1))
    SS = zeros(Complex{Float64},2,2,2,2,3,3)
    SS[:,:,:,:,:,:] = [OneSiteOp[i1,i2,m]*OneSiteOp[j1,j2,n]
                    for i1 in 1:2, j1 in 1:2, i2 in 1:2, j2 in 1:2,
                        m in 2:4, n in 2:4 ]
    TwoSiteOpVec = PermutedDimsArray(reshape(SS,16,9),(2,1))
    OneSite = reshape(PermutedDimsArray(transpose(H.OneSite)*OneSiteOpVec,(2,1)),2,2,N)
    TwoSite2 = reshape(H.TwoSite, 9, N)
    TwoSite = reshape(PermutedDimsArray(transpose(TwoSite2)*TwoSiteOpVec,(2,1)),4,4,N)
    NNQuadHamiltonian(copy(OneSite),copy(TwoSite))
end
function getTEBDexpvalue!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N,A)
    d,d2,N_site = size(H.OneSite)
    del = T/N
    U = makeNNQuadUnitary(H,del::Float64)
    expvalue = zeros(Complex{Float64},N+1,size(H.OneSite)[3])
    for j in 1:N_site
        expvalue[1,j] = onesite_expvalue(MPS,A[:,:,j],j)
        if real(expvalue[1,j]) > 1
            println("expvalue at site $(j) at time step 1 is $(expvalue[1,j])",)
        end
    end
    for i in 1:N
        update_oddsite!(MPS,U)
        update_evensite!(MPS,U)
        for j in 1:N_site
            expvalue[i+1,j] = onesite_expvalue(MPS,A[:,:,j],j)
            if real(expvalue[i+1,j]) > 1
                println("expvalue at site $(j) at time step $(i+1) is $(expvalue[i+1,j])",)
            end
        end
    end
    expvalue
end

function TEBDwithRenyi!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N,loc,α)
    d,d2,N_site = size(H.OneSite)
    del = T/N
    U = makeNNQuadUnitary(H,del::Float64)
    renyivalue = zeros(Float64,N+1)
    renyivalue[1] = getRenyi(MPS,loc,α)
    for i in 1:N
        update_oddsite!(MPS,U)
        update_evensite!(MPS,U)
        renyivalue[i+1] = getRenyi(MPS,loc,α)
    end
    return renyivalue
end


function make_superpositionMPO(U,P)
    d,d2,N = size(U)
    M = zeros(Complex{Float64},2,2,d,d,N)
    for i in 1:N
        M[1,1,:,:,i] = Matrix{Complex{Float64}}(I,d,d)
        M[1,2,:,:,i] = P[i]*U[:,:,i]
        M[2,2,:,:,i] = Matrix{Complex{Float64}}(I,d,d)
    end
    M1 = [1,0]
    Mend = [0,1]
    MatrixProductOperator(M1,M,Mend)
end
function do_MPOonMPS(MPS::VidalMPS,MPO::MatrixProductOperator)
    D,Dp,d,N = size(MPS.Gamma)
    D2,D2p,d2,d2p,N2 =  size(MPO.M)
    Gamma = zeros(Complex{Float64},D*D2,D*D2,d,N)

    @views M1 = contract(MPO.M1,[1],MPO.M[:,:,:,:,1],[1]) #get (D2,d,d)
    @views G1 = contract(MPS.Lambda[:,1],[1],MPS.Gamma[:,:,:,1],[1]) #get (D,d)
    Gamma[1,:,:,1] = reshape(PermutedDimsArray(contract(M1,[3],G1,[2]),(3,1,2)),D*D2,d)

    for i in 2:N-1
        @views Mi = MPO.M[:,:,:,:,i] #get (D2,D2,d,d)
        @views Gi = contract(LinearAlgebra.Diagonal(MPS.Lambda[:,i]),[2], MPS.Gamma[:,:,:,i], [1]) #get (D,D,d)
        Gamma[:,:,:,i] = reshape(PermutedDimsArray(contract(Mi,[4],Gi,[3]),(4,1,5,2,3)),D*D2,D*D2,d)
    end

    @views Mend = contract(MPO.Mend,[1],MPO.M[:,:,:,:,1],[2]) #get (D2,d,d)
    @views Gend = contract(LinearAlgebra.Diagonal(MPS.Lambda[:,N]),[2],MPS.Gamma[:,:,:,N],[1]) #get (D,D,d)
    @views Gend = contract(Gend,[2],MPS.Lambda[:,N+1],[1]) #get (D,d)
    Gamma[:,1,:,N] = reshape(PermutedDimsArray(contract(Mend,[3],Gend,[2]),(3,1,2)),D*D2,d)
    GenericMPS(Gamma)
end

function convert_to_Vidal(MPS::GenericMPS)
    MPS2 = convert_to_leftorthogonal(MPS)
    convert_to_Vidal(MPS2)
end

function convert_to_Vidal(MPS::LeftOrthogonalMPS)
    #need to do it in reverse order!
    D,D2,d,N = size(MPS.Gamma)
    GammaNew = zeros(Complex{Float64},D,D,d,N)
    LambdaNew = zeros(Float64,D,N+1)

    LambdaNew[1,N+1] = 1
    F = svd(MPS.Gamma[:,1,:,N])
    GammaNew[1:minimum([d,D]),1,:,N] = F.Vt
    LambdaNew[1:minimum([d,D]),N] = F.S
    GammaNew[:,1:minimum([d,D]),:,N-1] = PermutedDimsArray(contract(MPS.Gamma[:,:,:,N-1],[2],F.U*Diagonal(F.S),[1]),(1,3,2))

    for i in reverse(2:N-1)
        F = svd(Matrix(reshape(GammaNew[:,:,:,i],D,D*d)))
        LambdaNew[:,i] = F.S
        LambdaInv = zeros(Float64,D)
        for j in 1:D
            if LambdaNew[j,i] > 10^-13
                LambdaInv[j] = 1/LambdaNew[j,i+1]
            else
                LambdaInv[j] = 0
            end
        end
        GammaNew[:,:,:,i] = PermutedDimsArray(contract(reshape(F.Vt,D,D,d),[2],Diagonal(LambdaInv),[1]),(1,3,2))
        GammaNew[:,:,:,i-1] = PermutedDimsArray(contract(MPS.Gamma[:,:,:,i-1],[2],F.U*Diagonal(F.S),[1]),(1,3,2))
    end
    LambdaNew[1,1] = 1
    VidalMPS(GammaNew,LambdaNew)
end

function convert_to_leftorthogonal(MPS::GenericMPS)
    #takes an (unnormalized) Generic MPS and returns a normalized VidalMPS
    D,D2,d,N = size(MPS.Gamma)
    GammaNew = zeros(Complex{Float64},D,D,d,N)
    #qr decomposition
    F = LinearAlgebra.qr(PermutedDimsArray(MPS.Gamma[1,:,:,1],(2,1)))
    GammaNew[1,1:size(F.Q)[2],:,1] = PermutedDimsArray(Matrix(F.Q),(2,1))
    GammaNew[1:size(F.R)[1],:,:,2] = contract(F.R,[2],MPS.Gamma[:,:,:,2],[1])
    for i in 2:N-1
        F = LinearAlgebra.qr(reshape(PermutedDimsArray(GammaNew[:,:,:,i],(1,3,2)),D*d,D))
        GammaNew[:,:,:,i] = PermutedDimsArray(reshape(Matrix(F.Q),D,d,D),(1,3,2))
        GammaNew[:,:,:,i+1] = contract(F.R,[2],MPS.Gamma[:,:,:,i+1],[1])
    end

    A = contract(GammaNew[:,:,:,N],[1,2,3],conj(GammaNew[:,:,:,N]),[1,2,3])[1]
    GammaNew[:,:,:,N] = 1/sqrt(A)*GammaNew[:,:,:,N]
    B = contract(GammaNew[:,:,:,N],[1,2,3],conj(GammaNew[:,:,:,N]),[1,2,3])[1]
    LeftOrthogonalMPS(GammaNew)
end

function contract(M,loc1,Gamma,loc2)
    #contract an index
    #=
    loc1,loc2 are arrays of index to be contracted
    Make sure prod(size1[loc1]) = prod(size2[loc2])
    =#
    size1 = collect(size(M))
    dim1 = size(size1)[1]
    size2 = collect(size(Gamma))
    dim2 = size(size2)[1]
    index1 = filter(p->p∉loc1,collect(1:dim1))
    index2 = filter(p->p∉loc2,collect(1:dim2))
    if size(loc1)[1] == dim1
        M2 = copy(reshape(M,1,prod(size1[loc1])))
    else
        M2 = copy(reshape(PermutedDimsArray(M,Tuple(vcat(index1,loc1))),prod(size1[index1]),prod(size1[loc1])))
    end

    if size(loc2)[1] == dim2
        Gamma2 = copy(reshape(Gamma,prod(size2[loc2])))
    else
        Gamma2 = copy(reshape(PermutedDimsArray(Gamma,Tuple(vcat(loc2,index2))),prod(size2[loc2]),prod(size2[index2])))
    end
    reshape(M2*Gamma2,Tuple(vcat(size1[index1],size2[index2])))
end

function normalization_test(MPS::VidalMPS,index,side)
    D,D1,d, N  = size(MPS.Gamma)
    dummy = 0
    if side == "left"
        dummy = 0
    elseif side == "right"
        dummy = 1
    else
        println("say left or right")
    end

    if index == 1
        a = contract(MPS.Gamma[1,:,:,1],[2],MPS.Gamma[1,:,:,1],[2])
        @views println(contract(MPS.Gamma[1,:,:,1],[2],MPS.Gamma[1,:,:,1],[2]))
    elseif index == N
        a = contract(MPS.Gamma[:,1,:,N],[2],MPS.Gamma[:,1,:,N],[2])
        @views println(contract(MPS.Gamma[:,1,:,N],[2],MPS.Gamma[:,1,:,N],[2]))
    else
        G = contract(MPS.Gamma[:,:,:,index],[dummy+1],Diagonal(MPS.Lambda[:,N+dummy]),[2-dummy])
        a = contract(G,[2,3],G,[2,3])
        println(contract(G,[2,3],G,[2,3]))
    end
    a
end

function getRenyi(MPS::VidalMPS,loc,α)
    """
    get αth Renyi entropy for a cut between site loc and loc+1
    """
    Lambda = MPS.Lambda[:,loc+1]
    return 1/(1-α) log(sum(Lambda .^ α))
end

#this end is for the module
end
