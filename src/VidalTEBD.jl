module VidalTEBD

export VidalMPS, NNSpinHalfHamiltonian, NNQuadHamiltonian, NNQuadUnitary
export make_productVidalMPS, onesite_expvalue,onesite_expvalue1,onesite_expvalue2, TEBD!, makeNNQuadH, getTEBDexpvalue!,getTEBDexpvaluecopy!
export contract

using BenchmarkTools
using LinearAlgebra
using Random
using TensorOperations

GammaMat = Array{Complex{Float64},4}

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

include("./VidalMPSoperations.jl")
include("./TEBDoperations.jl")

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

"""
contract an index
loc1,loc2 are arrays of index to be contracted
Make sure prod(size1[loc1]) = prod(size2[loc2])
"""
function contract(M,loc1,Gamma,loc2)
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

function getRenyi(MPS::VidalMPS,loc::Int,α::Int)
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

function get_eigvals_squared(MPS::VidalMPS,cut_position::Vector{Int},alpha::Int)
    if alpha != 2
        error("the value for alpha must be 2. Other methods for calculating Renyi entropy for double cuts have not been implemented")
    end

    left = cut_position[1]
    right = cut_position[2]

    rho1 = Diagonal(MPS.Lambda[:,left])*Diagonal(MPS.Lambda[:,left])
    rho2 = Diagonal(MPS.Lambda[:,right+1])*Diagonal(MPS.Lambda[:,right+1])

    eigvals_squared = tr(rho1*rho1)*tr(rho2*rho2) #prove this!
    if eigvals_squared<0
        println("eigvals_squared is less than 0")
    end
    return eigvals_squared
end

function getRenyi(MPS::VidalMPS,cut_position::Vector{Int},alpha::Int)
    eigs_squared = get_eigvals_squared(MPS,cut_position,alpha)

    if eigs_squared<0
        println("eigs_squared = ", eigs_squared)
    end

    return -log(eigs_squared)
end

function calculate_overlap(MPS1,MPS2,cut_position::Vector{Int})
    #this algorithm is not optimized. It runs ot O(chi^8) time,whereas the optimal algorithm can run at O(chi^4).
    D,D1,d,N = size(MPS1.Gamma)
    T = Array{Complex{Float64},4}(undef,D,D,D,D)
    T2 = Array{Complex{Float64},5}(undef,D,d,D,D,D)
    left = cut_position[1]
    right= cut_position[2]

    U1 = contract(Diagonal(MPS1.Lambda[:,left]),[2],MPS1.Gamma[:,:,:,left],[1])
    U2 = contract(Diagonal(MPS2.Lambda[:,left]),[2],MPS2.Gamma[:,:,:,left],[1])
    T[:,:,:,:] = contract(conj.(U1),[3],U2,[3]) #dim(D,D,D,D)
    for index in (left+1):right
        U1[:,:,:] = contract(Diagonal(MPS1.Lambda[:,index]),[2],MPS1.Gamma[:,:,:,index],[1])
        U2[:,:,:] = contract(Diagonal(MPS2.Lambda[:,index]),[2],MPS2.Gamma[:,:,:,index],[1])
        T2[:,:,:,:,:] = contract(conj(U1),[1],T,[2])#dim(D,d,D,D,D)
        T[:,:,:,:] = PermutedDimsArray(contract(T2,[2,5],U2,[3,1]),(2,1,3,4)) #dim(D,D)
    end

    T[:,:,:,:] = contract(Diagonal(MPS1.Lambda[:,right+1]),[1],T,[2])#dim(D,D,D,D)
    T[:,:,:,:] = contract(T,[4],Diagonal(MPS2.Lambda[:,right+1]),[1])#dim(1)

    overlap = contract(T,[1,2,3,4],conj.(T),[1,2,3,4])[1]

    overlap = convert(Float64,overlap)

    if 0>overlap
        println("overlap = ",overlap)
    elseif overlap>1
        println("overlap = ", overlap)
    end

    return overlap

end
"""
preallocated version of calculate_overlap. This however, doesn't actually make things faster
"""
function calculate_overlap!(MPS1,MPS2,cut_position::Vector{Int},T,T2,U1,U2)
    #this algorithm is not optimized. It runs ot O(chi^8) time,whereas the optimal algorithm can run at O(chi^4).
    D,D1,d,N = size(MPS1.Gamma)
    if size(T) != (D,D,D,D)
        println("the dimensions of T does not match")
    end
    if size(T2) != (D,d,D,D,D)
        println("the dimensions of T2 does not match")
    end
    if size(U1) != (D,D,d)
        println("the dimensions of U1 does not match")
    end
    if size(U2) != (D,D,d)
        println("the dimensions of U2 does not match")
    end
    left = cut_position[1]
    right= cut_position[2]
    U1[:,:,:] = contract(Diagonal(MPS1.Lambda[:,left]),[2],MPS1.Gamma[:,:,:,left],[1])
    U2[:,:,:] = contract(Diagonal(MPS2.Lambda[:,left]),[2],MPS2.Gamma[:,:,:,left],[1])
    T[:,:,:,:] = contract(conj.(U1),[3],U2,[3]) #dim(D,D,D,D)
    for index in (left+1):right
        U1[:,:,:] = contract(Diagonal(MPS1.Lambda[:,index]),[2],MPS1.Gamma[:,:,:,index],[1])
        U2[:,:,:] = contract(Diagonal(MPS2.Lambda[:,index]),[2],MPS2.Gamma[:,:,:,index],[1])
        T2[:,:,:,:,:] = contract(conj(U1),[1],T,[2])#dim(D,d,D,D,D)
        T[:,:,:,:] = PermutedDimsArray(contract(T2,[2,5],U2,[3,1]),(2,1,3,4)) #dim(D,D)
    end

    T[:,:,:,:] = contract(Diagonal(MPS1.Lambda[:,right+1]),[1],T,[2])#dim(D,D,D,D)
    T[:,:,:,:] = contract(T,[4],Diagonal(MPS2.Lambda[:,right+1]),[1])#dim(1)

    overlap = contract(T,[1,2,3,4],conj.(T),[1,2,3,4])[1]

    overlap = convert(Float64,overlap)

    if 0>overlap
        println("overlap = ",overlap)
    elseif overlap>1
        println("overlap = ", overlap)
    end

    return overlap

end
"""
receives a preallocated array dim(values) = (M,M)
"""
function calculate_mutual_overlap!(MPS_list,cut_position::Vector{Int},values)
    M = length(MPS_list)
    if (M,M) != size(values)
        println("the preallocated array has a wrong size")
    end
    for raw in 1:M
        values[raw,raw] = 0
        for column in raw+1:M
            println("calculate_overlap")
            @time values[raw,column]= calculate_overlap(MPS_list[raw],MPS_list[column],cut_position)
            if values[raw,column]>1
                println("overlap = ", values[raw,column])
            end
            values[column,raw] = values[raw,column]
        end
    end

    mutual_renyi = sum(values)
    if isnan(mutual_renyi)
        println("there are NaN in mutual_renyi")
    end

    if mutual_renyi>M^2
        println("mutual_renyi = ",mutual_renyi)
        println("filtered values = ", values[values.>1])
    end

    return mutual_renyi
end

include("./hamiltonian_constructor.jl")

#this end is for the module
end
