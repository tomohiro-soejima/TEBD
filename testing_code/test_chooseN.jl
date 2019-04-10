include("../VidalTEBD.jl")
using .VidalTEBD
using StatsBase

#= The goal of this code is to test chooseN.
In particular, I am interested in what the resulting probability distribution is.

=#

List = [1,2,3,4,5,6,7,8,9,10]/55
N = 110000
list_of_indices = zeros(Int64,5*N)
for i in 1:N
    a = VidalTEBD.chooseN(List,5)
    list_of_indices[(i*5-4):i*5] = a
end
number_of_occurence = countmap(list_of_indices)
println(number_of_occurence)
