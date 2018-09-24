#= compare for loops and matrix multiplication
=#

function naive_for_loop(M,Gamma,D,D2,d)
    G = zeros(Float64,D,D,D2,D2,d)
    for i in 1:d,c4 in 1:D2,c3 in 1:D2, c2 in 1:D, c1 in 1:D
        G[c1,c2,c3,c4,i] = sum(M[c3,c4,i,:].*Gamma[C1,C2,:])
    end
    G
end

function naive_for_loop_reverse_order(M,Gamma,D,D2,d)
    G = zeros(Float64,D,D,D2,D2,d)
    for c1 in 1:D, c2 in 1:D,c3 in 1:D2,c4 in 1:D2,i in 1:d
        G[c1,c2,c3,c4,i] = sum(M[c3,c4,i,:].*Gamma[C1,C2,:])
    end
    G
end

function matrixForm(M,Gamma,D,D2,d)
    M2 = reshape(M,D2^2*d,d)
    Gamma2 = PermutedDimsArray(reshape(M,D^2,d),(2,1))
    G2 = M2*Gamma2
    G = PermutedDimsArray(reshape(G2,D2,D2,d,D,D),(4,5,1,2,3))
end

d = 2
D = 10
D2 = 2
M = rand(D2,D2,d,d) #MPO, D,D,d,d
Gamma = rand(D,D,d) #MPS D,D,d

A1 = @time  naive_for_loop(M,Gamma,D,D2,d)
A2 = @time  naive_for_loop_reverse_order(M,Gamma,D,D2,d)
A3 = @time matrixForm(M,Gamma,D,D2,d)
