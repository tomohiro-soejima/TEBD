include("./XYmodel_dynamics.jl")
using .dynamics
using Plots


#initialize
N = 20
x0 = N >> 1
sigma = 2
D = 10
J = 1
T = 0.5*pi
Nt = 100

MPS = dynamics.create_excited_state(N,x0,sigma,D)
H = dynamics.ising_Hamiltonian(zeros(Float64,N),J .* ones(Float64,N-1))

#define which variables to measure
Ob = -1/2*[1 0;0 -1] + 1/2*[1 0;0 1]
O = zeros(Float64,2,2,N)
for i in 1:N
    O[:,:,i] = Ob
end
expvalues = @time dynamics.VidalTEBD.getTEBDexpvalue!(MPS,H,T,Nt,O)
x = 1:(Nt+1)
plot(x,real(expvalues))
savefig("isingplot_gaussian_100.png")#

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

filename = "isingplot_gaussian_100"
x = 1:N
plot_series(x,real(expvalues),filename,"Ising plot")
