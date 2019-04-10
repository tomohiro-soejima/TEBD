using LinearAlgebra
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

function contract2(M,loc1,Gamma,loc2)
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

    @views begin
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
    end
    reshape(M2*Gamma2,(size1[index1]...,size2[index2]...))

end

A = rand(1000,10,100)
B = rand(100,10,1000)
M2 = zeros(100,100)

println("testing contract")
@time contract(A,[1,2],B,[2,3])
@time M = contract(A,[1,2],B,[2,3])
@time M2[:,:] = contract(A,[1,2],B,[2,3])

println("testing contrac2")
@time contract2(A,[1,2],B,[2,3])
@time M = contract2(A,[1,2],B,[2,3])
@time M2[:,:] =contract2(A,[1,2],B,[2,3])
