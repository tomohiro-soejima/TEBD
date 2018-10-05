include("./XYmodel_dynamics.jl")
using .dynamics
using Plots
using Printf

#filename

#initialize
N = 101
x0 = 51
sigma = 5
D = 2
T = 10pi
Nt = 500
h_list = zeros(Float64,N)
a = 1

#define which variables to measure
Ob = -1/2*[1 0;0 -1] + 1/2*[1 0;0 1]
O = zeros(Float64,2,2,N)
for i in 1:N
    O[:,:,i] = Ob
end

H = dynamics.xymodel_Hamiltonian(h_list,a)
MPS = dynamics.create_excited_state(N,x0,sigma,D)

expvalues = @time dynamics.VidalTEBD.getTEBDexpvalue!(MPS,H,T,Nt,O)
x = 1:(Nt+1)
plot(x,real(expvalues))
savefig("xyplot_gaussian_101sites_1.png")#

function plot_series(x,data,filename,title_name)
    u = maximum(data)
    k = 1
    for i in 1:size(data)[1]
        plot(x, real(data[i,:]),title = title_name,lw =3)
        xlabel!("Position")
        ylabel!("1/2 - <Sz>")
        ylims!((-u*1.2,u*1.2))
        savefig(filename * @sprintf("_%04d",k))
        k += 1
    end
end

filename = "xyplot_gaussian_101sites_1"
x = 1:N
plot_series(x,real(expvalues),filename,"xymodel plot")
#=
run(`convert -delay 5 -loop 0 $(filename)_\*.png $filename.gif`)
run(`rm $(filename)_\*.png`)
=#
