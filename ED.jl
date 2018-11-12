module ED
using SparseArrays

"""
construct a Hamiltonian matrix in Sz basis.
Note the order of basis vectors
index 1 = |++>
index 2 = |-+>
"""
function sparseIsing_Hamiltonian(J,hx,hz,N)
    H = zeros(Float64,2^N,2^N)
    state = zeros(Float64,N)
    I = Int64[]
    sizehint!(I,5*N)
    K = Int64[]
    sizehint!(K,5*N)
    R = Float64[]
    sizehint!(R,5*N)
    for i in 1:2^N
        convert_to_product(i,N,state)
        Hii = 0.0
        for j in 1:N-1
            Hii += J*state[j]*state[j+1]/4
        end
        for j in 1:N
            Hii += hz*state[j]/2
        end
        push!(I,i)
        push!(K,i)
        push!(R,Hii)

        for j in 1:N
            new_index = spin_flip(j,i,N)
            push!(I,new_index)
            push!(K,i)
            push!(R,hx)
        end
    end
    return sparse(I,K,R)
end

"""
given a linear index, returns a vector in |+1> |-1> tensor product notation
example for N = 3
i = 2 -> |-++> -> returns [-1,1,1]
"""
function convert_to_product(i,N,storage)
    bit = string(i-1,base=2,pad = N)
    for i in 1:N
        storage[i] = 1-2*parse(Int64,bit[N+1-i])
    end
end


"""
flips the spin at loc and returns a new index corresponding to that state.
"""
function spin_flip(loc,index,N)
    bit = string(index-1,base=2,pad =N)
    if bit[N+1-loc] == '0'
        str = '1'
    elseif bit[N+1-loc] == '1'
        str = '0'
    end
    new_bitstring = bit[1:N-loc]*str*bit[N+2-loc:end]
    new_index = parse(Int64,new_bitstring,base=2)+1
    return new_index
end
"""
convert a product state, given as an Array{Float64,2}(2,N) by a spin sz basis e.g. |++->
example
|x+x-> = 1/2(|++>+|-+>-|+->+|-->) = [1/2,1/2,-1/2,-1/2]
"""
function product_state_to_vector(PS)
    N = size(PS)[2]
    state = zeros(Float64,N)
    coeff = Array{Float64,1}(undef,2^N)
    for i in 1:2^N
        convert_to_product(i,N,state)
        ci = 1
        for j in 1:N
            index = state[j] == 1 ? 1 : 2
            ci *= PS[index,j]
        end
        coeff[i] = ci
    end
    return coeff
end

function Ising_time_evolve(J,hx,hz,PS,t)
    N = size(PS)[2]
    init_state = product_state_to_vector(PS)
    H = sparseIsing_Hamiltonian(J,hx,hz,N)
    U = exp(-im*H*t)
    return U*init_state
end

end #module
