# Define basic VidalMPS operations

struct VidalMPS
    #MPS in Vidal canonical form
    Gamma::Array{Complex{Float64},4} #dim = D,D,d,N
    Lambda::Array{Float64,2} #dim = D,N
    rank::Array{Int64,1}
end

"""
Tests whether a VidalMPS is normalized or not.
"""
function isnormalized(MPS::VidalMPS)
    norm = calculate_norm(MPS)
    return norm ≈ 1.0
end

"""
Check if the VidalTEbD is leftorthogonal or not. Note that for VidalTEBD,
the function does not need to be strictly left orthogonal. Instead, it needs
to be an isometry, where the size of the projector is the same as the schmidt
rank of the lambda matrix at the corresponding cut.
"""
function isleftorthogonal(MPS::VidalMPS, loc)
    D,D2,d,N = size(MPS.Gamma)
    T = get_leftoperator(MPS,loc)
    R = rank(Diagonal(MPS.Lambda[:,loc+1]))
    R2 = rank(Diagonal(MPS.Lambda[:,loc]))
    Rmax = max(R, R2)
    Rmin = min(R, R2)
    Rm = min(Rmin*d, Rmax)
    if Rm < D
        I_partial = zeros(Int64,D,D)
        I_partial[1:Rm, 1:Rm] = Matrix{Int64}(I, Rm, Rm)
    else
        I_partial = Matrix{Int64}(I, D, D)
    end
    println(norm(T-I_partial))
    return isapprox(T, I_partial, rtol = 10^-7)
end

"""
Check if the VidalTEbD is rightorthogonal or not. Note that for VidalTEBD,
the function does not need to be strictly right orthogonal. Instead, it needs
to be an isometry, where the size of the projector is the same as the schmidt
rank of the lambda matrix at the corresponding cut.
"""
function isrightorthogonal(MPS::VidalMPS, loc)
    D,D2,d,N = size(MPS.Gamma)
    T = get_rightoperator(MPS, N-loc)
    R = rank(Diagonal(MPS.Lambda[:,loc+2]))
    R2 = rank(Diagonal(MPS.Lambda[:,loc+1]))
    Rmax = max(R,R2)
    Rmin = min(R, R2)
    Rm = min(Rmin*d, Rmax)
    if Rm < D
        I_partial = zeros(Int64,D,D)
        I_partial[1:Rm, 1:Rm] = Matrix{Int64}(I, Rm, Rm)
    else
        I_partial = Matrix{Int64}(I, D, D)
    end
    #@show diag(T)
    println("print the norm ", norm(T-I_partial))
    return isapprox(T, I_partial, rtol = 10^-7)
end

"""
Check whether the rank attribute corresponds to the actual rank of the matrix
"""
function iscorrectrank(MPS::VidalMPS)
    D, D2, d, N = size(MPS.Gamma)
    for loc in 1:(N+1)
        r = rank(Diagonal(MPS.Lambda[:,loc]))
        if r != MPS.rank[loc]
            println("loc = $loc")
            println("r = $r")
            println("rank = $(MPS.rank[loc])")
            return false
        end
    end
    return true
end

function calculate_norm(MPS::VidalMPS)
    D,D2,d,N = size(MPS.Gamma)
    T = zeros(eltype(MPS.Gamma),D,D)
    temp = zero(T)
    @views Lambda = Diagonal(MPS.Lambda[:,1])
    @views Gamma = MPS.Gamma[:,:,:,1]
    A = zero(Gamma)
    @tensor A[α,β,i] = Lambda[α,γ]*Gamma[γ,β,i]
    @tensor T[α,β] = conj(A[γ,α,i])*A[γ,β,i]
    for loc in 2:N
        @views Lambda = Diagonal(MPS.Lambda[:,loc])
        @views Gamma = MPS.Gamma[:,:,:,loc]
        @tensor A[α,β,i] = Lambda[α,γ]*Gamma[γ,β,i]
        @tensor temp[α, β] = T[γ, ρ]*conj(A[γ, α, i])*A[ρ, β, i]
        T .= temp
    end
    @views Lambda = Array(Diagonal(MPS.Lambda[:, N+1]))
    @tensor norm = T[α, β]*Lambda[α, γ]*Lambda[β, γ]
    return norm
end


"""
calculates the operator to left. Can be used to check if it is left orthogonal.
n is the number of matrices to the left.
"""
function get_leftoperator(MPS::VidalMPS, n)
    D,D2,d,N = size(MPS.Gamma)
    T = zeros(eltype(MPS.Gamma),D,D)
    temp = zero(T)
    @views Lambda = Diagonal(MPS.Lambda[:,1])
    @views Gamma = MPS.Gamma[:,:,:,1]
    A = zero(Gamma)
    @tensor T[α,β] = Lambda[a, b] * Gamma[b, α, i] * Lambda[a, c] * conj(Gamma[c, β, i])
    for loc in 2:n
        @views Lambda = Diagonal(MPS.Lambda[:,loc])
        @views Gamma = MPS.Gamma[:,:,:,loc]
        @tensor A[α,β,i] = Lambda[α,γ]*Gamma[γ,β,i]
        @tensor temp[α, β] = T[γ, ρ]*conj(A[γ, α, i])*A[ρ, β, i]
        T .= temp
    end
    return T
end

"""
calculates the operator to right. Can be used to check if it is right orthogonal.
n is the number of matrices to the right.
"""
function get_rightoperator(MPS::VidalMPS, n)
    D,D2,d,N = size(MPS.Gamma)
    T = zeros(eltype(MPS.Gamma),D,D)
    temp = zero(T)
    @views Lambda = Diagonal(MPS.Lambda[:,N+1])
    @views Gamma = MPS.Gamma[:,:,:,N]
    A = zero(Gamma)
    @tensor A[α,β,i] = Lambda[γ,β]*Gamma[α,γ,i]
    @tensor T[α,β] = conj(A[α,γ,i])*A[β,γ,i]
    for loc in reverse((N-n+1):N-1)
        @views Lambda = Diagonal(MPS.Lambda[:,loc+1])
        @views Gamma = MPS.Gamma[:,:,:,loc]
        @tensor A[α,β,i] = Lambda[γ,β]*Gamma[α,γ,i]
        @tensor temp[α, β] = conj(A[α,γ,i])*A[β,ρ,i]*T[γ, ρ]
        T .= temp
    end
    return T
end

"""
Make a VidalMPS starting from a product state
"""
function make_productVidalMPS(ProductState,D)
    d, N = size(ProductState)
    Gamma = zeros(Complex{Float64},D,D,d,N)
    Lambda = zeros(Float64,D,N+1)
    Gamma[1,1,:,:] = ProductState
    A = zeros(eltype(Gamma), d, d+1)
    Lambda[1,:] = ones(Float64,N+1)
    rank = ones(Int64,N+1)
    VidalMPS(Gamma, Lambda, rank)
end

"""
Enlarge the bond dimension
"""
function make_biggerMPS(MPS::VidalMPS,D_new)
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


function onesite_expvalue(MPS::VidalMPS,U,loc)
    onesite_expvalue2(MPS,U,loc) #using this version, which seems to be faster
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
