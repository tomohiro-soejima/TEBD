include("../VidalTEBD.jl")

using .VidalTEBD

function stochasticTEBD(initial_MPS,number_of_copies,Time,Nt,number_of_data_points))
    MPS_list = Array{VidalMPS,1}(undef,number_of_copies)
    for index in 1:number_of_copies
        MPS_list[i] = initial_MPS
    end

    dt = T/Nt
    raw_Renyi_list = zeros(Float64,number_of_copies)
    raw_Renyi = zeros(Float64,number_of_data_points)
    mutual_Renyi = zeros(Float64,number_of_data_points)
    for iteration in 1:number_of_data_points
        for index in 1:number_of_copies
            stochasticTEBD!(MPS_list[i],H,T,Nt/number_of_data_points)
            raw_Renyi[i] = getRenyi(MPS,(left_cut,right_cut),alpha) #define this
        end

        raw_Renyi[iteration+1] = sum(raw_Renyi_list)
        mutual_Renyi = calculate_mutual_renyi2(MPS_list,[left_cut,right_cut])
    end

    Renyi_entropy = 1/number_of_copies*raw_Renyi + 1/(number_of_copies^2)*mutual_Renyi
    Renyi_entropy[1] = getRenyi(MPS,[left_cut,right_cut],alpha) # this is the fastest way to calculate the Renyi entropy before any operation is done

    return Renyi_entropy,raw_Renyi,mutual_Renyi
end

function getRenyi(MPS,cut_position::Vector{Int},alpha)
    if alpha != 2
        error("the value for alpha must be 2. Other methods for calculating Renyi entropy for double cuts have not been implemented")
    end

    rho1 = Diagonal(MPS.Lambda[left])*Diagonal(MPS.Lambda[left])
    rho2 = Diagonal(MPS.Lambda[right+1])*Diagonal(MPS.Lambda[right+1])

    return -log(tr(rho1*rho1)*tr(rho2*rho2)) #prove this!
end

function calculate_overlap(MPS1,MPS2,cut_position::Vector{Int})
    D,D1,d,N = size(MPS1)
    T = Array{Complex{Float64},2}(undef,D,D)
    T2 = Array{Complex{Float64},2}(undef,D,d,D)
    left = cut_position[1]
    right= cut_position[2]
    U1 = contract(Diagonal(MPS1.Lambda[:,left]),[2],MPS1.Gamma[:,:,:,left],[1])
    U2 = contract(Diagonal(MPS2.Lambda[:,left]),[2],MPS1.Gamma[:,:,:,right],[1])
    T[:,:] = contract(conjugate.(U1),[1,3],U2,[1,3]) #dim(D,D)
    for index in (left+1):right
        U1[:,:] = contract(Diagonal(MPS1.Lambda[:,index]),[2],MPS1.Gamma[:,:,:,index],[1])
        U2[:,:] = contract(Diagonal(MPS2.Lambda[:,index]),[2],MPS1.Gamma[:,:,:,index],[1])
        T2[:,:] = contract(conjugate(U1),[1],T,[1])#dim(D,d,D)
        T[:,:] = contract(T,[2,3],U2,[1,2]) #dim(D,D)
    end

    T[:,:] = contract(Diagonal(MPS1.Lambda[:,right+1]),[1],T,[1])#dim(D,D)
    T[:,:] = contract(T,[2],Diagonal(MPS2.Lambda[:,right+1]),[1])#dim(D,D)

    return tr(T)
end

function calculate_mutual_renyi2(MPS_list,cut_position::Vector{Int})
    M = length(MPS_list)
    values = Array{Float64,2}(undef,M,M)
    for raw in 1:M
        for column in raw+1:M
            value[raw,column]= norm(calculate_overlap(MPS_list[raw],MPS[column],cut_position))^2
        end
    end

    mutual_renyi = 2*sum(values) # a factor of two because we only calculated half of the matrix

    return mutual_renyi
end
