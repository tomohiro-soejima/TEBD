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

function twogate_onMPS!(MPS::VidalMPS,U,loc)
    thetaNew = theta_ij(MPS,U,loc)
    F = LinearAlgebra.svd(copy(thetaNew))
    updateMPSafter_twogate!(MPS,F,loc)
end

function theta_ij(MPS::VidalMPS,U,loc)
    D,D2,d,N = size(MPS.Gamma)
    #look at the rank of each Lambda matrices
    R1 = MPS.rank[loc]
    R2 = MPS.rank[loc+1]
    R3 = MPS.rank[loc+2]
    #use the rank to get smaller L1, L2, L3
    L1 = Diagonal(view(MPS.Lambda,1:R1,loc))
    L2 = Diagonal(view(MPS.Lambda,1:R2,loc+1))
    L3 = Diagonal(view(MPS.Lambda,1:R3,loc+2))
    G1 = view(MPS.Gamma,1:R1,1:R2,:,loc)
    G2 = view(MPS.Gamma,1:R2,1:R3,:,loc+1)
    theta = Array{eltype(G1),4}(undef, d, d, R1, R3)
    @tensor theta[i, j, α, β] = L1[α,γ1]*G1[γ1,γ2,i]*L2[γ2,γ3]*G2[γ3,γ4,j]*L3[γ4,β]
    theta2 = U*reshape(theta,d^2,R1*R3)
    A = reshape(PermutedDimsArray(reshape(theta2,d,d,R1,R3),(3,1,4,2)),(R1*d,R3*d))
    return A
end

function updateMPSafter_twogate!(MPS::VidalMPS,F::SVD,loc)
    D,D2,d,N = size(MPS.Gamma)
    ϵ = 10^-10
    R1 = MPS.rank[loc]
    R2 = MPS.rank[loc+1]
    R3 = MPS.rank[loc+2]
    L1 = view(MPS.Lambda,1:R1,loc)
    L2 = view(MPS.Lambda,:,loc+1)
    L3 = view(MPS.Lambda,1:R3,loc+2)
    Gamma1 = view(MPS.Gamma,:,:,:,loc)
    Gamma2 = view(MPS.Gamma,:,:,:,loc+1)

    Rd = min(R1*d,R3*d,D)
    S = zeros(Float64,D)
    for i in 1:Rd
        if F.S[i] > ϵ
            S[i] = F.S[i]
        else
            S[i] = 0 #somehow if I make this 10^-6 my code explods
            break
        end
    end
    @views L2[:] = S[:]/sqrt(sum(S[:].^2))
    R2 = rank(Diagonal(L2))

    @views GL1 = PermutedDimsArray(reshape(F.U[:,1:R2],R1,d,R2),(1,3,2))
    @views GL2 = reshape(F.Vt[1:R2,:],R2,R3,d)

    L1_inv = zero(L1)
    for i in 1:R1
        L1_inv[i] = 1/L1[i]
    end
    Gamma1[:,:,:] = zero(Gamma1)
    Gamma1[1:R1,1:R2,:] = contract(Diagonal(L1_inv),[2],GL1,[1])

    #@views L2[:] = F.S[1:D]/sqrt(sum(F.S[1:D].^2))
    L3_inv = zero(L3)
    for i in 1:R3
        L3_inv[i] = 1/L3[i]
    end
    Gamma2[:,:,:]= zero(Gamma2)
    Gamma2[1:R2,1:R3,:]= permutedims(contract(GL2,[2],Diagonal(L3_inv),[1]),(1,3,2))
    MPS.rank[loc+1] = R2
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
        update_oddsite!(MPS,U)
        update_evensite!(MPS,U)

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

include("./improved_chooseN.jl")

function stochasticTEBD_multicopy(initial_MPS,Hamiltonian,number_of_copies,Time,Nt,number_of_data_points,cut_position)
    MPS_list = Array{VidalMPS,1}(undef,number_of_copies)
    D,D2,d,N = size(initial_MPS.Gamma)
    left_cut = cut_position[1]
    right_cut = cut_position[2]
    for index in 1:number_of_copies
        MPS_list[index] = VidalMPS(deepcopy(initial_MPS.Gamma),deepcopy(initial_MPS.Lambda))
    end
    #preallocate arrays
    values = zeros(Float64,number_of_copies,number_of_copies)

    eigs_squared_list = zeros(Float64,number_of_copies)
    eigs_squared = zeros(Float64,number_of_data_points+1)
    mutual_overlap = zeros(Float64,number_of_data_points+1)
    for iteration in 1:number_of_data_points
        for index in 1:number_of_copies
            stochasticTEBD_traverse!(MPS_list[index],Hamiltonian,Time/number_of_data_points,Nt/number_of_data_points)
            eigs_squared_list[index] = get_eigvals_squared(MPS_list[index],[left_cut,right_cut],2)
        end

        eigs_squared[iteration+1] = sum(eigs_squared_list)
        mutual_overlap[iteration+1] = calculate_mutual_overlap!(MPS_list,[left_cut,right_cut],values)
    end

    Renyi_entropy = -log.(1/number_of_copies^2 .*eigs_squared + 1/(number_of_copies^2).*mutual_overlap)
    Renyi_entropy[1] = getRenyi(initial_MPS,[left_cut,right_cut],2) # this is the fastest way to calculate the Renyi entropy before any operation is done
    eigs_squared[1] = get_eigvals_squared(initial_MPS,[left_cut,right_cut],2)*number_of_copies
    mutual_overlap[1] = get_eigvals_squared(initial_MPS,[left_cut,right_cut],2)*(number_of_copies^2-number_of_copies)

    return Renyi_entropy,eigs_squared,mutual_overlap,MPS_list
end
