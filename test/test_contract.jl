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

function testing(b::Val{N}) where N
    b
end


A = rand(2,4,5)
B = rand(8,5,3)
C = rand(8,5)
D = rand(40)
