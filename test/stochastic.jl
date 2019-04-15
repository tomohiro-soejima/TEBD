#=
include("../src/VidalTEBD.jl")
using Test
using LinearAlgebra
using .VidalTEBD
=#

@testset "Stochastic processes" begin
    """
    tests chooseN
    """
    function is_correct_sum()
        Plist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]/55
        N = 100000
        result_list = Array{Int64, 1}(undef, 3N)
        for i in 1:N
            result_list[((i - 1) * 3 + 1):(i * 3)] = VidalTEBD.chooseN(Plist, 3)
        end

        count_list = [count(p -> p == i, result_list) for i in 1:10]/N

        expvaluelist = [prob_chosen(i/55, 0, 0, 0, 0, 10, 3) for i in 1:10]

        for i in 1:10
            if abs(count_list[i] - expvaluelist[i])/expvaluelist[i] > 0.01
                return false
            end
        end

        return true
    end

    function prob_chosen(prob, PI, PN, I, N, L, M)
        (M-I)/(L-I-N)*(PI+prob+(1-PI-PN-prob)*(M-I-1)/(L-I-N-1))/(PI+(1-PI-PN)*(M-I)/(L-I-N))
    end

    @test is_correct_sum()
end
