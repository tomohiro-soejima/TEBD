mutable struct VidalMPS
    Gamma::Array
    Lambda::Array
end

function ProductVidalMPS(ProductState,dim)
    N = length(ProductState)
    Hd = length(ProductState[1])
    Gamma = Vector{Array{Array{Float64},1}}(N)
    Lambda = Vector{Vector{Float64}(2)}(N-1)
    for i in 1:N
        v = ProductState[i]
        if i == 1
            Gammai = Vector{Vector{Float64}(dim)}(Hd)
            for j in 1:Hd
                vec = zeros(Float64,dim)
                vec[1] = v[j]
                Gammai[j] = vec
            Gamma[i] = Gammai
        elseif i == N
            Gammai = Vector{Vector{Float64}(dim)}(Hd)
            for j in 1:Hd
                vec = zeros(Float64,dim)
                vec[1] = v[j]
                Gammai[j] = vec
            Gamma[i] = Gammai
        else
            Gammai = Vector{Array{Float64,2}(dim,dim)}(Hd)
            for j in 1:Hd
                vec = zeros(Float64,dim,dim)
                vec[1,1] = v[j]
                Gammai[j] = vec
            Gamma[i] = Gammai

        for i in 1:N-1
            lambda = [1,0]
            Lambda[i] = [lambda,lambda]
