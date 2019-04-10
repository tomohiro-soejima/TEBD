include("../VidalTEBD.jl")
using .VidalTEBD
using BenchmarkTools

function contract(M,loc1,Gamma,loc2,dim::Val{N}) where N
    #contract an index
    #=
    loc1,loc2 are arrays of index to be contracted
    Make sure prod(size1[loc1]) = prod(size2[loc2])
    =#
    size1 = collect(size(M))
    dim1 = ndims(M)
    size2 = collect(size(Gamma))
    dim2 = ndims(Gamma)
    index1 = filter(p->p∉loc1,collect(1:dim1))
    index2 = filter(p->p∉loc2,collect(1:dim2))
    V1 = vcat(index1,loc1)
    T1 = ntuple(i->V1[i],ndims(M))

    if size(loc1)[1] == dim1
        M2 = copy(reshape(M,1,prod(size1[loc1])))
    else
        V1 = vcat(index1,loc1)
        T1 = ntuple(i->V1[i],ndims(M))
        copy(M2 = reshape(PermutedDimsArray(M,T1),prod(size1[index1]),prod(size1[loc1])))
    end
    if size(loc2)[1] == dim2
        Gamma2 = copy(reshape(Gamma,prod(size2[loc2])))
    else
        V2 = vcat(index2,loc2)
        T2 = ntuple(i->V2[i],ndims(Gamma))
        Gamma2 = copy(reshape(PermutedDimsArray(Gamma,T2),prod(size2[loc2]),prod(size2[index2])))
    end

    V3 = vcat(size1[index1],size2[index2])
    T3 = ntuple(i->V3[i],dim)
    Array{Complex{Float64},N}(reshape(M2*Gamma2,T3))
end

function contract_2(M,loc1,Gamma,loc2)
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


A = rand(400,20,5)
B = rand(8000,5,3)
C = rand(8,5)
D = rand(40)

zeros(Complex{Float64},2,2)
PS = zeros(Float64,(2,N))
PS[1,:] = ones(Float64,N)
MPS = VidalTEBD.make_productVidalMPS(PS,20)

@btime VidalTEBD.contract($A,[1,2],$B,[1])
@btime contract_2($A,[1,2],$B,[1])
