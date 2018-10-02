include("./XYmodel_dynamics.jl")
using Plots
using Printf

#initialize
N = 100
x0 = 50
sigma = 5
D = 10
T = 2*pi
Nt = 100
J = 1

#define which variables to measure
Sx = 1/2*[0 1;1 0]
O = zeros(Float64,2,2,N)
for i in 1:N
    O[:,:,i] = Sx
end

zeros(Complex{Float64},2,2)
PS = zeros(Float64,(2,N))
PS[1,:] = ones(Float64,N)
PS[:,50] = [1/sqrt(2),1/sqrt(2)]
MPS = VidalTEBD.make_productVidalMPS(PS,D)

H = ising_Hamiltonian(zeros(Float64,N),J .* ones(Float64,N-1))
expvalues = @time VidalTEBD.getTEBDexpvalue!(MPS,H,T,Nt,O)
x = 1:(Nt+1)
plot(x,real(expvalues))
savefig("isingplot.png")#

function plot_series(x,data,filename,title_name)
    u = maximum(data)
    k = 1
    for i in 1:size(data)[1]
        plot(x, real(data[i,:]),title = title_name,lw =3)
        xlabel!("Position")
        ylabel!("<Sx>")
        ylims!((-u*1.2,u*1.2))
        savefig(filename * @sprintf("_%04d",k))
        k += 1
    end
end

filename = "isingplot"
x = 1:N
plot_series(x,real(expvalues),filename,"Ising plot")



#=
run(`convert -delay 5 -loop 0 $(filename)_\*.png $filename.gif`)
run(`rm $(filename)_\*.png`)
=#
