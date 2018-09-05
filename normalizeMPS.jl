mutable struct VidalMPS
    Gamma::Array
    Lambda::Array
end

function ProductVidalMPS(ProductState,dim)
    N = length(ProductState)
    Hd = length(ProductState[1])
    Gamma = Vector{Array{Array{Float64},1}}(undef,N)
    Lambda = Vector{Array{Float64,2}}(undef,N-1)
    for i in 1:N
        v = ProductState[i]
        if i == 1
            Gammai = Vector{Vector{Float64}}(undef,Hd)
            for j in 1:Hd
                vec = zeros(Float64,dim)
                vec[1] = v[j]
                Gammai[j] = vec'
            end
            Gamma[i] = Gammai
        elseif i == N
            Gammai = Vector{Vector{Float64}}(undef,Hd)
            for j in 1:Hd
                vec = zeros(Float64,dim)
                vec[1] = v[j]
                Gammai[j] = vec
            end
            Gamma[i] = Gammai
        else
            Gammai = Vector{Array{Float64,2}}(undef,Hd)
            for j in 1:Hd
                vec = zeros(Float64,dim,dim)
                vec[1,1] = v[j]
                Gammai[j] = vec
            end
            Gamma[i] = Gammai
        end
    end

    for i in 1:N-1
        lambda = zero(Float64, dim,dim)
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
