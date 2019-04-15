#=
The first version of chooseN was not able to capture the right statistics.
This code implements the better version of the algorithm.
=#

"""
Given a vector of N items with associated probability distribution Pi, return M items such that the probability for those M to be chosen is proportional to sum(Pi).
Check a pdf by me for more detailed explanation.
"""
function chooseN(Plist::Vector{<:Real},M::Int)
    if !(isproblist(Plist))
        println("The probability does not add up to 1")
    end

    L = length(Plist)
    #the following two sums are best explained in my pdf
    PI = 0 #sum of the probability of indices that was chosen
    PN = 0 #sum of the probability of indices that was not chosen
    I = 0 #number of items that was chosen
    N = 0 #number of items that was not chosen
    indices = Int64[]
    for (index,prob) in enumerate(Plist)
        if ischosen(index,prob,PI,PN,I,N,L,M)
            push!(indices,index)
            PI += prob
            I += 1
        else
            PN += prob
            N += 1
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

function ischosen(index,prob,PI,PN,I,N,L,M)
    if L-I-N-1 == 0
        P_index = float(M-I)
    else
        P_index = prob_chosen(prob, PI,PN,I,N,L,M)
    end
    return (rand()<P_index)
end

function prob_chosen(prob, PI, PN, I, N, L, M)
    (M-I)/(L-I-N)*(PI+prob+(1-PI-PN-prob)*(M-I-1)/(L-I-N-1))/(PI+(1-PI-PN)*(M-I)/(L-I-N))
end
