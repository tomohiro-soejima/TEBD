include("./XYmodel_dynamics.jl")
using Plots
using Printf

#initialize
N = 100
x0 = 50
sigma = 5
D = 10
T = 2*pi
Nt = 10
J = 1

#define which variables to measure
Sz = 1/2*[1 0;0 -1]
O = zeros(Float64,2,2,N)
for i in 1:N
    O[:,:,i] = Sz
end

H = ising_Hamiltonian(zeros(Float64,N),J .* ones(Float64,N-1))
expvalues = @time excited_state_dynamics(N,x0,sigma,D,T,Nt,H,O)
x = 1:(Nt+1)
plot(x,real(expvalues))
savefig("isingplot.png")#

function plot_series(x,data,filename,title_name)
    u = maximum(data)
    k = 1
    for i in 1:size(data)[1]
        plot(x, real(data[i,:]),title = title_name,lw =3)
        xlabel!("Position")
        ylabel!("1/2 - <Sz>")
        ylims!((-u*1.2,u*1.2))
        savefig(filename * @sprintf("_%04d",k))
        global k += 1
    end
end

expvalues2 = -(expvalues .- 0.5)
filename = "isingplot"
x = 1:N
plot_series(x,real(expvalues2),filename,"Ising plot")



#=
run(`convert -delay 5 -loop 0 $(filename)_\*.png $filename.gif`)
run(`rm $(filename)_\*.png`)
=#
