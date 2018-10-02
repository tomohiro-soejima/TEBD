include("./XYmodel_dynamics.jl")
using Plots
using Printf

#initialize
N = 10
x0 = 5
sigma = 5
D = 10
T = 2*pi
Nt = 100
h_list = zeros(Float64,N)
alpha = 0

#define which variables to measure
Sz = 1/2*[1 0;0 -1]
O = zeros(Float64,2,2,N)
for i in 1:N
    O[:,:,i] = Sz
end

expvalues = @time xymodel_dynamics(N,x0,sigma,D,T,Nt,h_list,alpha,O)
x = 1:(Nt+1)
plot(x,real(expvalues))
savefig("XYplot.png")#

expvalues2 = -(expvalues .- 0.5)

filename = "xyplot"
x = 1:N
k = 1
for i in 1:size(expvalues)[1]
    plot(x, real(expvalues2[i,:]),title = "ExpValues",lw =3)
    xlabel!("Position")
    ylabel!("1/2 - <Sz>")
    ylims!((-0.5,0.5))
    savefig(filename * @sprintf("_%04d",k))
    k += 1
end

#=
run(`convert -delay 5 -loop 0 $(filename)_\*.png $filename.gif`)
run(`rm $(filename)_\*.png`)
=#
