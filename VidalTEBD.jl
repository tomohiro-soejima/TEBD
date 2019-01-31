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
    #faster than MPS1!
    D,D2,d,N = size(MPS.Gamma)
    R = permutedims(reshape(view(MPS.Gamma,:,:,:,loc),D^2,d),(2,1))
    MPS.Gamma[:,:,:,loc] = reshape(PermutedDimsArray(U*R,(2,1)),D,D,d)
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
    A = reshape(PermutedDimsArray(reshape(U*theta,d,d,D,D),(3,1,4,2)),(d*D,d*D))
    return A
end
function updateMPSafter_twogate!(MPS::VidalMPS,F::SVD,loc)
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
        if L1[i] > 10^-10
            L1_inv[i] = 1/L1[i]
        end
    end
    Gamma1[:,:,:] = contract(Diagonal(L1_inv),[2],GL1,[1])

    S = zeros(Float64,D)
    for i in 1:D
        if F.S[i] > 10^-10
            S[i] = F.S[i]
        else
            S[i] = 0 #somehow if I make this 10^-6 my code explods
        end
    end
    @views L2[:] = S[:]/sqrt(sum(S[:].^2))
    #@views L2[:] = F.S[1:D]/sqrt(sum(F.S[1:D].^2))
    L3_inv = zero(L3)
    for i in 1:D
        if L3[i] >10^-10
            L3_inv[i] = 1/L3[i]
        end
    end
    Gamma2[:,:,:]= permutedims(contract(GL2,[2],Diagonal(L3_inv),[1]),(1,3,2))
end
function TEBD!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N;operator = nothing, α = nothing,loc = nothing, MPO = nothing)
    #initializing for different options
    operator != nothing ? expvalues = zeros(Complex{Float64},N+1,size(H.OneSite)[3]) : false
    if operator != nothing
        expvalues = zeros(Complex{Float64},N+1,size(H.OneSite)[3])
        for j in 1:N_site
            expvalue[1,j] = onesite_expvalue(MPS,A[:,:,j],j)
        end
    end
    if α != nothing
        renyivalue = zeros(Float64,N+1)
        renyivalue[1] = getRenyi(MPS,loc,α)
    end
    if MPO != nothing
        MPOvalue = zeros(Complex{Float64},N+1)
        MPOvalue[1] = getMPOexpvalue(MPS,MPO)
    end

    #initalizing the unitary operators
    del = T/N
    U = makeNNQuadUnitary(H,del::Float64)
    for i in 1:N
        try
            update_oddsite!(MPS,U)
            update_evensite!(MPS,U)
        catch y
            println("This happened after ",i," th time step")
            error(y)
        end

        if operator != nothing
            for j in 1:N_site
                expvalue[i+1,j] = onesite_expvalue(MPS,A[:,:,j],j)
            end
        end
        α != nothing ? renyivalue[i+1] = getRenyi(MPS,loc,α) : false
        MPO != nothing ? MPOvalue[i+1] = getMPOexpvalue(MPS,MPO) : false
    end
    result = Dict()
    operator != nothing ? result["expvalue"] = expvalue : false
    α != nothing ? result["renyivalue"] = renyivalue : false
    MPO != nothing ? result["MPOvalue"] = MPOvalue : false
    return result
end

function TEBD_traverse!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N;operator = nothing, α = nothing,loc = nothing, MPO = nothing)
    #initializing for different options
    operator != nothing ? expvalues = zeros(Complex{Float64},N+1,size(H.OneSite)[3]) : false
    if operator != nothing
        expvalues = zeros(Complex{Float64},N+1,size(H.OneSite)[3])
        for j in 1:N_site
            expvalue[1,j] = onesite_expvalue(MPS,A[:,:,j],j)
        end
    end
    if α != nothing
        renyivalue = zeros(Float64,N+1)
        renyivalue[1] = getRenyi(MPS,loc,α)
    end
    if MPO != nothing
        MPOvalue = zeros(Complex{Float64},N+1)
        MPOvalue[1] = getMPOexpvalue(MPS,MPO)
    end

    #initalizing the unitary operators
    del = T/2N #it traverses twice
    U = makeNNQuadUnitary(H,del::Float64)
    for i in 1:N
        try
            update_traverse!(MPS,U)
        catch y
            println("This happened after ",i," th time step")
            error(y)
        end

        if operator != nothing
            for j in 1:N_site
                expvalue[i+1,j] = onesite_expvalue(MPS,A[:,:,j],j)
            end
        end
        α != nothing ? renyivalue[i+1] = getRenyi(MPS,loc,α) : false
        MPO != nothing ? MPOvalue[i+1] = getMPOexpvalue(MPS,MPO) : false
    end

    result = Dict()
    operator != nothing ? result["expvalue"] = expvalue : false
    α != nothing ? result["renyivalue"] = renyivalue : false
    MPO != nothing ? result["MPOvalue"] = MPOvalue : false
    return result
end

"""
deprecated
"""
function TEBD_simple!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N)
    del = T/N
    U = makeNNQuadUnitary(H,del::Float64)
    for i in 1:N
        try
            update_oddsite!(MPS,U)
            update_evensite!(MPS,U)
        catch y
            println("This happened after ",i," th time step")
            error(y)
        end
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
function update_traverse!(MPS::VidalMPS,U::NNQuadUnitary)
    D,D2,d,N = size(MPS.Gamma)
    #first iteration of trotter gates
    for loc in 1:N-1
        onegate_onMPS!(MPS,U.OneSite[:,:,loc],loc)
        twogate_onMPS!(MPS,U.TwoSite[:,:,loc],loc)
    end
    onegate_onMPS!(MPS,U.OneSite[:,:,N],N)

    #second iteration
    onegate_onMPS!(MPS,U.OneSite[:,:,N],N)
    for loc in reverse(1:N-1)
        twogate_onMPS!(MPS,U.TwoSite[:,:,loc],loc)
        onegate_onMPS!(MPS,U.OneSite[:,:,loc],loc)
    end
end
function stochastic_update_traverse!(MPS::VidalMPS,U::NNQuadUnitary)
    D,D2,d,N = size(MPS.Gamma)
    #first iteration of trotter gates
    for loc in 1:N-1
        onegate_onMPS!(MPS,U.OneSite[:,:,loc],loc)
        stochastic_twogate_onMPS!(MPS,U.TwoSite[:,:,loc],loc)
    end
    onegate_onMPS!(MPS,U.OneSite[:,:,N],N)

    #second iteration
    onegate_onMPS!(MPS,U.OneSite[:,:,N],N)
    for loc in reverse(1:N-1)
        stochastic_twogate_onMPS!(MPS,U.TwoSite[:,:,loc],loc)
        onegate_onMPS!(MPS,U.OneSite[:,:,loc],loc)
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
"""
deprecated
"""
function getTEBDexpvalue!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N,A)
    d,d2,N_site = size(H.OneSite)
    del = T/N
    U = makeNNQuadUnitary(H,del::Float64)
    expvalue = zeros(Complex{Float64},N+1,size(H.OneSite)[3])
    for j in 1:N_site
        expvalue[1,j] = onesite_expvalue(MPS,A[:,:,j],j)
        if real(expvalue[1,j]) > 1.1
            println("expvalue at site $(j) at time step 1 is $(expvalue[1,j])",)
        end
    end
    for i in 1:N
        update_oddsite!(MPS,U)
        update_evensite!(MPS,U)
        for j in 1:N_site
            expvalue[i+1,j] = onesite_expvalue(MPS,A[:,:,j],j)
            if real(expvalue[i+1,j]) > 1.1
                println("expvalue at site $(j) at time step $(i+1) is $(expvalue[i+1,j])",)
            end
        end
    end
    expvalue
end
"""
deprecated
"""
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
"""
deprecated
"""
function TEBDwithMPO!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N,MPO::MatrixProductOperator)
    d,d2,N_site = size(H.OneSite)
    del = T/N
    U = makeNNQuadUnitary(H,del::Float64)
    MPOvalue = zeros(Complex{Float64},N+1)
    MPOvalue[1] = getMPOexpvalue(MPS,MPO)
    for i in 1:N
        update_oddsite!(MPS,U)
        update_evensite!(MPS,U)
        MPOvalue[i+1] = getMPOexpvalue(MPS,MPO)
    end
    return MPOvalue
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

"""
Given a VidalMPS, create a vector representation of that MPS using product states as basis
"""
function create_vector(MPS::VidalMPS)
    D,D1,d,N = size(MPS.Gamma)
    coeff = Array{Complex{Float64},1}(undef,d^N)
    for i in 1:2^N
        dit = string(i-1,base = d,pad=N)
        M = MPS.Lambda[:,1]
        for j in 1:N
            index = parse(Int64,dit[N+1-j])+1
            M = contract(M,[1],MPS.Gamma[:,:,index,j],[1])
            M = contract(M,[1],Diagonal(MPS.Lambda[:,j+1]),[1])
        end
        coeff[i] = sum(M)
    end
    return coeff
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
    size1 = size(M)
    dim1 = length(size1)
    size2 = size(Gamma)
    dim2 = length(size2)
    index1 = filter(p->p∉loc1,collect(1:dim1))
    index2 = filter(p->p∉loc2,collect(1:dim2))
    dim_M2_1 = prod(size1[index1])
    dim_M2_2 = prod(size1[loc1])
    dim_G2_2 = prod(size2[index2])
    dim_G2_1 = prod(size2[loc2])


    if isa(M,Diagonal) & length(loc1) == 1
        M2 = M
    elseif size(loc1)[1] == dim1
        M2 = reshape(M,1,dim_M2_2)
    else
        M2 = reshape(permutedims(M,Tuple(vcat(index1,loc1))),dim_M2_1,dim_M2_2)
    end

    if isa(Gamma,Diagonal) & length(loc2) == 1
        Gamma2 = Gamma
    elseif size(loc2)[1] == dim2
        Gamma2 = reshape(Gamma,dim_G2_1)
    else
        Gamma2 = (reshape(permutedims(Gamma,Tuple(vcat(loc2,index2))),dim_G2_1,dim_G2_2))
    end
    reshape(M2*Gamma2,(size1[index1]...,size2[index2]...))

end

function normalization_test(MPS::VidalMPS,index,side)
    # there is a doubt about the correctness of this algorithm. Shouldn't it have complex conjugate?
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
    #=
    get αth Renyi entropy for a cut between site loc and loc+1
    =#
    Lambda = MPS.Lambda[:,loc+1]
    return 1/(1-α)*log(sum(Lambda .^ (2*α)))
end

function getMPOexpvalue(MPS::VidalMPS,MPO::MatrixProductOperator)
    D1,D11, d,N = size(MPS.Gamma)
    D2,D22, dx,dy,Nx = size(MPO.M)
    #the first site
    @views begin
    M1 = contract(MPO.M1,[1],MPO.M[:,:,:,:,1],[1])#D2,d,d
    G1 = MPS.Gamma[1,:,:,1] #D1,d
    G11 = contract(M1,[3],G1,[2])#D2,d,D1
    T = contract(conj.(G1),[2],G11,[2])#D1,D2,D1
    end
    for i in 2:N-1
        @views Mi = MPO.M[:,:,:,:,i]#D2,D2,d,d
        @views Gi = contract(Diagonal(MPS.Lambda[:,i]),[2],MPS.Gamma[:,:,:,i],[1])#D1,D1,d
        T1 = contract(T,[3],Gi,[1])#D1,D2,D1,d
        T2 = contract(Mi,[1,4],T1,[2,4]) #D2,d,D1,D1
        T[:,:,:] = contract(conj.(Gi),[3,1],T2,[2,3])
    end
    MN = contract(MPO.M[:,:,:,:,N],[2],MPO.Mend,[1])#D2,d,d
    GN = contract(Diagonal(MPS.Lambda[:,N]),[2],MPS.Gamma[:,1,:,N],[1])#D1,d
    T1 = contract(T,[3],GN,[1])#D1,D2,d
    T2 = contract(T1,[2,3],MN,[1,3]) #D1,d
    return contract(T2,[1,2],conj.(GN),[1,2])[1]
end

function stochasticTEBD!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N;operator = nothing, α = nothing,loc = nothing, MPO = nothing)
    #initializing for different options
    operator != nothing ? expvalues = zeros(Complex{Float64},N+1,size(H.OneSite)[3]) : false
    if operator != nothing
        expvalues = zeros(Complex{Float64},N+1,size(H.OneSite)[3])
        for j in 1:N_site
            expvalue[1,j] = onesite_expvalue(MPS,A[:,:,j],j)
        end
    end
    if α != nothing
        renyivalue = zeros(Float64,N+1)
        renyivalue[1] = getRenyi(MPS,loc,α)
    end
    if MPO != nothing
        MPOvalue = zeros(Complex{Float64},N+1)
        MPOvalue[1] = getMPOexpvalue(MPS,MPO)
    end

    #initalizing the unitary operators
    del = T/N #it traverses twice
    U = makeNNQuadUnitary(H,del::Float64)
    for i in 1:N
        try
            stochastic_update_oddsite!(MPS,U)
            stochastic_update_evensite!(MPS,U)
        catch y
            println("This happened after ",i," th time step")
            error(y)
        end

        if operator != nothing
            for j in 1:N_site
                expvalue[i+1,j] = onesite_expvalue(MPS,A[:,:,j],j)
            end
        end
        α != nothing ? renyivalue[i+1] = getRenyi(MPS,loc,α) : false
        MPO != nothing ? MPOvalue[i+1] = getMPOexpvalue(MPS,MPO) : false
    end
    result = Dict()
    operator != nothing ? result["expvalue"] = expvalue : false
    α != nothing ? result["renyivalue"] = renyivalue : false
    MPO != nothing ? result["MPOvalue"] = MPOvalue : false
    return result
end

function stochasticTEBD_traverse!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N;operator = nothing, α = nothing,loc = nothing, MPO = nothing)
    #initializing for different options
    operator != nothing ? expvalues = zeros(Complex{Float64},N+1,size(H.OneSite)[3]) : false
    if operator != nothing
        expvalues = zeros(Complex{Float64},N+1,size(H.OneSite)[3])
        for j in 1:N_site
            expvalue[1,j] = onesite_expvalue(MPS,A[:,:,j],j)
        end
    end
    if α != nothing
        renyivalue = zeros(Float64,N+1)
        renyivalue[1] = getRenyi(MPS,loc,α)
    end
    if MPO != nothing
        MPOvalue = zeros(Complex{Float64},N+1)
        MPOvalue[1] = getMPOexpvalue(MPS,MPO)
    end

    #initalizing the unitary operators
    del = T/2N #it traverses twice
    U = makeNNQuadUnitary(H,del::Float64)
    for i in 1:N
        try
            stochastic_update_traverse!(MPS,U)
        catch y
            println("This happened after ",i," th time step")
            error(y)
        end

        if operator != nothing
            for j in 1:N_site
                expvalue[i+1,j] = onesite_expvalue(MPS,A[:,:,j],j)
            end
        end
        α != nothing ? renyivalue[i+1] = getRenyi(MPS,loc,α) : false
        MPO != nothing ? MPOvalue[i+1] = getMPOexpvalue(MPS,MPO) : false
    end
    result = Dict()
    operator != nothing ? result["expvalue"] = expvalue : false
    α != nothing ? result["renyivalue"] = renyivalue : false
    MPO != nothing ? result["MPOvalue"] = MPOvalue : false
    return result
end

function stochasticTEBD_simple!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N)
    del = T/N
    U = makeNNQuadUnitary(H,del::Float64)
    for i in 1:N
        try
            stochastic_update_oddsite!(MPS,U)
            stochastic_update_evensite!(MPS,U)
        catch y
            println("This happened after ",i," th time step")
            error(y)
        end
    end
end

function getStochasticTEBDexpvalue!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N,A)
    d,d2,N_site = size(H.OneSite)
    del = T/N
    U = makeNNQuadUnitary(H,del::Float64)
    expvalue = zeros(Complex{Float64},N+1,size(H.OneSite)[3])
    for j in 1:N_site
        expvalue[1,j] = onesite_expvalue(MPS,A[:,:,j],j)
        if real(expvalue[1,j]) > 1.1
            println("expvalue at site $(j) at time step 1 is $(expvalue[1,j])",)
        end
    end
    for i in 1:N
        stochastic_update_oddsite!(MPS,U)
        stochastic_update_evensite!(MPS,U)
        for j in 1:N_site
            expvalue[i+1,j] = onesite_expvalue(MPS,A[:,:,j],j)
            if real(expvalue[i+1,j]) > 1.1
                println("expvalue at site $(j) at time step $(i+1) is $(expvalue[i+1,j])",)
            end
        end
    end
    expvalue
end

function stochasticTEBDwithRenyi!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N,loc,α)
    d,d2,N_site = size(H.OneSite)
    del = T/N
    U = makeNNQuadUnitary(H,del::Float64)
    renyivalue = zeros(Float64,N+1)
    renyivalue[1] = getRenyi(MPS,loc,α)
    for i in 1:N
        stochastic_update_oddsite!(MPS,U)
        stochastic_update_evensite!(MPS,U)
        renyivalue[i+1] = getRenyi(MPS,loc,α)
    end
    return renyivalue
end

function stochasticTEBDwithMPO!(MPS::VidalMPS,H::NNQuadHamiltonian,T,N,MPO::MatrixProductOperator)
    d,d2,N_site = size(H.OneSite)
    del = T/N
    U = makeNNQuadUnitary(H,del::Float64)
    MPOvalue = zeros(Complex{Float64},N+1)
    MPOvalue[1] = getMPOexpvalue(MPS,MPO)
    for i in 1:N
        stochastic_update_oddsite!(MPS,U)
        stochastic_update_evensite!(MPS,U)
        MPOvalue[i+1] = getMPOexpvalue(MPS,MPO)
    end
    return MPOvalue
end


function stochastic_update_oddsite!(MPS::VidalMPS,U::NNQuadUnitary)
    D,D2,d,N = size(MPS.Gamma)
    for loc in 1:2:N-1
        onegate_onMPS!(MPS,U.OneSite[:,:,loc],loc)
        stochastic_twogate_onMPS!(MPS,U.TwoSite[:,:,loc],loc)
    end
    if N%2 == 1
        onegate_onMPS!(MPS,U.OneSite[:,:,N],N)
    end
end
function stochastic_update_evensite!(MPS::VidalMPS,U::NNQuadUnitary)
    D,D2,d,N = size(MPS.Gamma)
    for loc in 2:2:N-1
        onegate_onMPS!(MPS,U.OneSite[:,:,loc],loc)
        stochastic_twogate_onMPS!(MPS,U.TwoSite[:,:,loc],loc)
    end
    if N%2 == 0
        onegate_onMPS!(MPS,U.OneSite[:,:,N],N)
    end
end

function stochastic_twogate_onMPS!(MPS::VidalMPS,U,loc)
    thetaNew = theta_ij(MPS,U,loc)
    F = LinearAlgebra.svd(copy(thetaNew))
    stochastic_updateMPSafter_twogate!(MPS,F,loc)
end

function stochastic_updateMPSafter_twogate!(MPS::VidalMPS,F::SVD,loc)
    D,D2,d,N = size(MPS.Gamma)
    L1 = view(MPS.Lambda,:,loc)
    L2 = view(MPS.Lambda,:,loc+1)
    L3 = view(MPS.Lambda,:,loc+2)
    Gamma1 = view(MPS.Gamma,:,:,:,loc)
    Gamma2 = view(MPS.Gamma,:,:,:,loc+1)

    index1 = chooseN(F.S .^2,D)
    if length(unique(index1))!=D
        error("stop")
    end
    @views GL1 = PermutedDimsArray(reshape(F.U[:,index1],D,d,D),(1,3,2))
    @views GL2 = reshape(F.Vt[index1,:],D,D,d)

    L1_inv = zero(L1)
    for i in 1:D
        if L1[i] > 10^-10
            L1_inv[i] = 1/L1[i]
        end
    end
    Gamma1[:,:,:] = contract(Diagonal(L1_inv),[2],GL1,[1])

    S = zeros(Float64,D)
    for i in 1:D
        if F.S[index1[i]] > 10^-10
            S[i] = F.S[index1[i]]
        else
            S[i] = 0 #somehow if I make this 10^-6 my code explods
        end
    end
    @views L2[:] = S[:]/sqrt(sum(S[:].^2))
    #@views L2[:] = F.S[1:D]/sqrt(sum(F.S[1:D].^2))
    L3_inv = zero(L3)
    for i in 1:D
        if L3[i] >10^-10
            L3_inv[i] = 1/L3[i]
        end
    end
    Gamma2[:,:,:]= permutedims(contract(GL2,[2],Diagonal(L3_inv),[1]),(1,3,2))
end

function chooseN(list::Array{Float64,1},N::Int64)
    a = copy(list)
    b = Array{Int64,1}(undef,N)
    index_list = collect(1:length(list))
    for i in 1:N
        bi = choose1(a)
        if sum(a) == sum(list)
            error("Something went wrong")
        end
        b[i] = index_list[bi]
        filter!(!isequal(b[i]),index_list)
    end
    return b
end

function choose1(list::Array{Float64,1})
    p = rand(Float64)
    for i in 1:length(list)
        if p < sum(list[1:i])/sum(list)
            deleteat!(list,i)
            return i
        end
    end

    #if everthing is zero, choose a value randomly
    i = ceil(Int64,p*length(list))
    deleteat!(list,i)
    return i
end

function stochasticTEBD_multicopy(initial_MPS,number_of_copies,Time,Nt,number_of_data_points))
    MPS_list = Array{VidalMPS,1}(undef,number_of_copies)
    for index in 1:number_of_copies
        MPS_list[i] = initial_MPS
    end

    dt = T/Nt
    raw_Renyi_list = zeros(Float64,number_of_copies)
    raw_Renyi = zeros(Float64,number_of_data_points)
    mutual_Renyi = zeros(Float64,number_of_data_points)
    for iteration in 1:number_of_data_points
        for index in 1:number_of_copies
            stochasticTEBD!(MPS_list[i],H,T,Nt/number_of_data_points)
            raw_Renyi[i] = getRenyi(MPS,(left_cut,right_cut),alpha) #define this
        end

        raw_Renyi[iteration+1] = sum(raw_Renyi_list)
        mutual_Renyi = calculate_mutual_renyi2(MPS_list,[left_cut,right_cut])
    end

    Renyi_entropy = 1/number_of_copies*raw_Renyi + 1/(number_of_copies^2)*mutual_Renyi
    Renyi_entropy[1] = getRenyi(MPS,[left_cut,right_cut],alpha) # this is the fastest way to calculate the Renyi entropy before any operation is done

    return Renyi_entropy,raw_Renyi,mutual_Renyi
end

function getRenyi(MPS,cut_position::Vector{Int},alpha)
    if alpha != 2
        error("the value for alpha must be 2. Other methods for calculating Renyi entropy for double cuts have not been implemented")
    end

    rho1 = Diagonal(MPS.Lambda[left])*Diagonal(MPS.Lambda[left])
    rho2 = Diagonal(MPS.Lambda[right+1])*Diagonal(MPS.Lambda[right+1])

    return -log(tr(rho1*rho1)*tr(rho2*rho2)) #prove this!
end

function calculate_overlap(MPS1,MPS2,cut_position::Vector{Int})
    D,D1,d,N = size(MPS1)
    T = Array{Complex{Float64},2}(undef,D,D)
    T2 = Array{Complex{Float64},2}(undef,D,d,D)
    left = cut_position[1]
    right= cut_position[2]
    U1 = contract(Diagonal(MPS1.Lambda[:,left]),[2],MPS1.Gamma[:,:,:,left],[1])
    U2 = contract(Diagonal(MPS2.Lambda[:,left]),[2],MPS1.Gamma[:,:,:,right],[1])
    T[:,:] = contract(conjugate.(U1),[1,3],U2,[1,3]) #dim(D,D)
    for index in (left+1):right
        U1[:,:] = contract(Diagonal(MPS1.Lambda[:,index]),[2],MPS1.Gamma[:,:,:,index],[1])
        U2[:,:] = contract(Diagonal(MPS2.Lambda[:,index]),[2],MPS1.Gamma[:,:,:,index],[1])
        T2[:,:] = contract(conjugate(U1),[1],T,[1])#dim(D,d,D)
        T[:,:] = contract(T,[2,3],U2,[1,2]) #dim(D,D)
    end

    T[:,:] = contract(Diagonal(MPS1.Lambda[:,right+1]),[1],T,[1])#dim(D,D)
    T[:,:] = contract(T,[2],Diagonal(MPS2.Lambda[:,right+1]),[1])#dim(D,D)

    return tr(T)
end

function calculate_mutual_renyi2(MPS_list,cut_position::Vector{Int})
    M = length(MPS_list)
    values = Array{Float64,2}(undef,M,M)
    for raw in 1:M
        for column in raw+1:M
            value[raw,column]= norm(calculate_overlap(MPS_list[raw],MPS[column],cut_position))^2
        end
    end

    mutual_renyi = 2*sum(values) # a factor of two because we only calculated half of the matrix

    return mutual_renyi
end

#this end is for the module
end
