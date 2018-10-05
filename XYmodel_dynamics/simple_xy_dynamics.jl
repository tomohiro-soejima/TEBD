include("./XYmodel_dynamics.jl")
using .dynamics
using Plots
using Printf

#initialize
N = 25
x0 = N >> 1
D = 5
a = 1
T = 0.1*pi
Nt = 5
filename = "simple_xyplot_heisenberg_try" * @sprintf("_%03d",N)

zeros(Complex{Float64},2,2)
PS = zeros(Float64,(2,N))
PS[1,:] = ones(Float64,N)
PS[:,5] = [0,1]
PS[:,10] = [0,1]
PS[:,13] = [0,1]
PS[:,17] = [0,1]
PS[:,20] = [0,1]
MPS = dynamics.VidalTEBD.make_productVidalMPS(PS,D)
H = dynamics.xymodel_Hamiltonian(zeros(Float64,N),a)

#define which variables to measure
Ob = -1/2*[1 0;0 -1] + 1/2*[1 0;0 1]
O = zeros(Float64,2,2,N)
for i in 1:N
    O[:,:,i] = Ob
end
expvalues = @time dynamics.VidalTEBD.getTEBDexpvalue!(MPS,H,T,Nt,O)
x = 1:(Nt+1)
#plot(x,real(expvalues))
#savefig(filename)#

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

x = 1:N
#@time plot_series(x,real(expvalues),filename,"Ising plot")
