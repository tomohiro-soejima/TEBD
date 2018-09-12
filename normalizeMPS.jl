using BenchmarkTools

struct VidalMPS
    Gamma::Array{Float64,4}
    Lambda::Array{Float64,3}
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
    Lambda = zeros(Float64,D,D,N+1)
    Gamma[1,1,:,:] = ProductState
    Lambda[1,:] = ones(Float64,N+1)
    VidalMPS(Gamma,Lambda)
end

function OneGateOnMPS(MPS::VidalMPS{D,d,N},U,loc)
    R = PermutedDimsArray(reshape(view(MPS.Gamma,:,:,:,loc),D^2,d),(2,1))
    R[:,:] = U*R
end

function OneGateOnMPSCopy(MPS::VidalMPS{D,d,N},U,loc)
    #To figure out whether in place multiplication is faster
    R = PermutedDimsArray(reshape(view(MPS.Gamma,:,:,:,loc),D^2,d),(2,1))
    R[:,:] = U*R
end

function TwoGateOnMPS(MPS::VidalMPS,U,loc)
end
