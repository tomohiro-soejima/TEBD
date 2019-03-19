include("../VidalTEBD.jl")
using .VidalTEBD
using StatsBase

#= The goal of this code is to test chooseN.
In particular, I am interested in what the resulting probability distribution is. 

=#

List = [1,2,3,4,5,6,7,8,9,10]/55
list_of_indices = zeros(5*N,Int64)
N = 2000
for i in 1:N
    a = VidalTEBD.chooseN(List,5)
    list_of_indices[(N*5-4):N*5]
end
number_of_occurence = countmap(list_of_indices)
println(number_of_occurence)
