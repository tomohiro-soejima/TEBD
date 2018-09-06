mutable struct VidalMPS
    Gamma::Array
    Lambda::Array
end

function ProductVidalMPS(ProductState,D)
    N = length(ProductState)
    d = length(ProductState[1])
    Gamma = Vector{Array{Array{Float64},1}}(undef,N)
    Lambda = Vector{Array{Float64,2}}(undef,N-1)
    for i in 1:N
        v = ProductState[i]
        if i == 1
            Gammai = Vector{Vector{Float64}}(undef,d)
            for j in 1:d
                vec = zeros(Float64,dim)
                vec[1] = v[j]
                Gammai[j] = vec'
            end
            Gamma[i] = Gammai
        elseif i == N
            Gammai = Vector{Vector{Float64}}(undef,d)
            for j in 1:d
                vec = zeros(Float64,D)
                vec[1] = v[j]
                Gammai[j] = vec
            end
            Gamma[i] = Gammai
        else
            Gammai = Vector{Array{Float64,2}}(undef,d)
            for j in 1:d
                vec = zeros(Float64,D,D)
                vec[1,1] = v[j]
                Gammai[j] = vec
            end
            Gamma[i] = Gammai
        end
    end

    for i in 1:N-1
        lambda = zero(Float64, D,D)
        lambda[1][1] = 1
        Lambda[i] = lambda
    end

    VidalMPS(Gamma,Lambda)
end


"function VidalMPStoCoefficients(MPS,ProductState)
    N = length(ProductState)
    Gamma = MPS.Gamma
    Lambda = MPS.Lambda
    a = ProductState[1]'*Gamma[1]
    for i in 1:N-1
        a = a*(Lambda[i]*(ProductState[i+1]'Gamma[i+1])
"

function OneGateOnMPS(MPS::VidalMPS,U,loc)
    Gamma = MPS.Gamma[loc]
    Gamma = U*Gamma
    MPS.Gamma[loc] = Gamma
end

function TwoGateOnMPS(MPS::VidalMPS,U,loc)
    if loc == 1
        TwoGateOnMPSLeftEdge(MPS::VidalMPS,U)
    elseif loc == length(MPS.Gamma)-1
        TwoGateOnMPSRightEdge(MPS::VidalMPS,U)
    else
        TwoGateOnMPSMiddle(MPS::VidalMPS,U,loc)
    end
end

function TwoGateOnMPSMiddle(MPS::VidalMPS,U,loc)
    Gamma1 = MPS.Gamma[loc]
    Gamma2 = MPS.Gamma[loc+1]
    lambda1 = MPS.Lambda[loc-1]
    lambda2 = MPS.Lambda[loc]
    lambda3 = MPS.Lambda[loc+1]
    Hd = length(Hd)
    D  = size(lambda1,1)
    V = Array{Array{Float64,2},1}(undef,Hd^2)

    ind = 1
    for i in 1:Hd
        for j in 1:Hd
            V[ind] = lambda1*Gamma1[i]*lambda2*Gamma2[j]*lambda3
            ind += 1
        end
        ind += 1
    end

    Theta = permutedims(reshape(U*V, (Hd,Hd)))
    Gamma12 = Array{Float64,2}{Hd*D,Hd*D}
    ind_x = 1
    ind_y = 1
    for i in 1:Hd
        for a in 1:D
            for j in 1:Hd
                for b in 1:D
                    if lambda1[a,a] == 0 or lambda3[b,b] == 0
                        ThetaD[ind_x,ind_y] = 0
                    else
                        ThetaD[ind_x,ind_y] = Theta[i,j][a,b]/(lambda1[a,a]*lambda3[b,b])
                    end
                    ind_y += 1
                end
                ind_y += 1
            end
            ind_x += 1
        end
        ind_x += 1
    end

    F = svd(ThetaD)
