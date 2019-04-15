using TensorOperations
using LinearAlgebra

Lambda = Diagonal(rand(2))
Gamma = rand(2,2,2)

#These two tensor operations work
@tensor Transfer[α, β] := Lambda[a, b] * Gamma[c, α, d] * Gamma[d, β, a] * Lambda[b, c]
@tensor Transfer[α, β] := Lambda[a, b] * Gamma[c, α, d] * Lambda[b, c] * Gamma[d, β, a]

#This has identical indexing as above two expressions, but somehow does not work
@tensor Transfer[α, β] := Lambda[a, b] * Lambda[b, c] * Gamma[c, α, d] * Gamma[d, β, a]
#Gives an error!

#It works after adding a parenthesis
@tensor Transfer[α, β] := Lambda[a, b] * (Lambda[b, c] * Gamma[c, α, d]) * Gamma[d, β, a]
