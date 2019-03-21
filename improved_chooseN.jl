#=
The first version of chooseN was not able to capture the right statistics.
This code implements the better version of the algorithm.
=#

"""
Given a vector of N items with associated probability distribution Pi, return M items such that the probability for those M to be chosen is proportional to sum(Pi).
Check a pdf by me for more detailed explanation.
"""
function chooseN(Plist::Vector{Real},M::Int)
    if !(isproblist(Plist))
        println("The probability does not add up to 1")
    end

    N = length(Plist)
    #the following two sums are best explained in my pdf
    PI = 0 #sum of the probability of indices that was chosen
    PN = 0 #sum of the probability of indices that was not chosen
    I = 0 #number of items that was chosen
    N = 0 #number of items that was not chosen
    indices = Int64[]
    for index in 1:length(Plist)
        if ischosen(index,PI,PN)
            push!(indices,index)
            PI += Plist[index]
        else
            PN += Plist[index]
        end

        if length(indices) == M
            break
        end
    end

    return indices
end

function isproblist(List)
    return sum(List)≈1
end

function ischosen(index,PI,PN)
    return true
end


Plist = 0.1*ones(10)
M = 5
a = chooseN(Plist,M)
println(a)
