include("./XYmodel_dynamics.jl")
using .dynamics
using Profile
using Plots
using Printf

#compare xy and Heisenberg result

#filename
filename = "xy_Heisenberg"

#initialize
N = 101
x0 = 51
sigma = 5
D = 2
T = 10pi
Nt = 1000
h_list = zeros(Float64,N)
a = 1

#define which variables to measure
Ob = -1/2*[1 0;0 -1] + 1/2*[1 0;0 1]
O = zeros(Float64,2,2,N)
for i in 1:N
    O[:,:,i] = Ob
end

H_H = dynamics.xymodel_Hamiltonian(h_list,1)
H_xy = dynamics.xymodel_Hamiltonian(h_list,0)

MPS = dynamics.create_excited_state(N,x0,sigma,D)
values_H = @time dynamics.VidalTEBD.getTEBDexpvalue!(MPS,H_H,T,Nt,O)

MPS = dynamics.create_excited_state(N,x0,sigma,D)
values_xy = @time dynamics.VidalTEBD.getTEBDexpvalue!(MPS,H_xy,T,Nt,O)

x = 1:(Nt+1)

function plot_series(x,data,filename,title_name,sep=1)
    u = maximum(data)
    k = 1
    for i in 1:sep:size(data)[1]
        plot(x, real(data[i,:]),title = title_name,lw =3)
        full_title = @sprintf(" t = %.1f pi", 200*(i-1)/(size(data)[1]-1))
        xlabel!("Position")
        ylabel!("1/2 - <Sz>")
        ylims!((-u*1.2,u*1.2))
        title!(title_name*full_title)
        savefig(filename * @sprintf("_%04d",k))
        k += 1
    end
end

x = 1:N
@time plot_series(x,real(values_H),"Heisenberg","Heisenberg plot",1)
@time plot_series(x,real(values_xy),"XY_model","XY model plot",1)
@time plot_series(x,real(values_xy-values_H),"XY-Heisenberg","XY-Heisenberg",1)

#=
run(`convert -delay 5 -loop 0 $(filename)_\*.png $filename.gif`)
run(`rm $(filename)_\*.png`)
=#
