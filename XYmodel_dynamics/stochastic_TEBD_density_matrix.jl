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
        mutual_Renyi = calculate_mutual_renyi(MPS_list,[left_cut,right_cut])
    end

    Renyi_entropy = 1/number_of_copies*raw_Renyi + 1/(number_of_copies^2)*mutual_Renyi
    Renyi_entropy[1] = getRenyi(MPS,[left_cut,right_cut],alpha) # this is the fastest way to calculate the Renyi entropy before any operation is done

    return Renyi_entropy,raw_Renyi,mutual_Renyi
end
