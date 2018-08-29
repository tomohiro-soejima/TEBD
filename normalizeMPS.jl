mutable struct MPS
    Gamma::Array
    Lambda::Array
end

a = MPS([1,2,3],[4,5,6])

println(typeof(a))

println(a.Gamma)
