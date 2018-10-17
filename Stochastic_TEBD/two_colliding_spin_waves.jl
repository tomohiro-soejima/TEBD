include("../XYmodel_dynamics/XYmodel_dynamics.jl")
using .dynamics
using Plots
using Printf
using Profile
using LinearAlgebra
using Traceur

#filename
filename = "xy_colliding_2"

#initialize
N = 150
x1 = 60
x2 = 91
sigma = 5
k1 = 1/3
k2 = -1/3

D = 5 #this will be multiplied by 4

h_list = zeros(Float64,N)
a = 0

T = 50pi
Nt = 500

#define which variables to measure
Ob = -1/2*[1 0;0 -1] + 1/2*[1 0;0 1]
O = zeros(Float64,2,2,N)
for i in 1:N
    O[:,:,i] = Ob
end

H = dynamics.xymodel_Hamiltonian(h_list,a)

U = LinearAlgebra.Diagonal([0,1,0,0]) #spin down ladder operator
U2 = ones(4,N)
U1 = reshape(U*U2,2,2,N)

c1 = [exp(-(xi-x1)^2/(2*sigma^2)-im*xi*k1) for xi in 1:N]
MPO1 = dynamics.VidalTEBD.make_superpositionMPO(U1,c1)
c2 = [exp(-(xi-x2)^2/(2*sigma^2)-im*xi*k2) for xi in 1:N]
MPO2 = dynamics.VidalTEBD.make_superpositionMPO(U1,c2)
#make all up product state
zeros(Complex{Float64},2,2)
PS = zeros(Float64,(2,N))
PS[1,:] = ones(Float64,N)
MPS2 = dynamics.VidalTEBD.make_productVidalMPS(PS,D)
#Apply MPO
MPS3 = dynamics.VidalTEBD.do_MPOonMPS(MPS2,MPO1)
#MPS = dynamics.VidalTEBD.convert_to_Vidal(MPS3)
MPS4 = dynamics.VidalTEBD.convert_to_Vidal(MPS3)
MPS5 = dynamics.VidalTEBD.do_MPOonMPS(MPS4,MPO2)
MPS = dynamics.VidalTEBD.convert_to_Vidal(MPS5)

expvalues = @time dynamics.VidalTEBD.getStochasticTEBDexpvalue!(MPS,H,T,Nt,O)
x = 1:(Nt+1)
plot(x,real(expvalues))
savefig(filename*".png")#

function plot_series(x,data,filename,title_name,sep=1)
    u = maximum(data)
    k = 1
    for i in 1:sep:size(data)[1]
        plot(x, real(data[i,:]),title = title_name,lw =3)
        full_title = @sprintf(" t = %.2f pi", 50*(i-1)/(size(data)[1]-1))
        xlabel!("Position")
        ylabel!("1/2 - <Sz>")
        ylims!((-u*1.2,u*1.2))
        title!(title_name*full_title)
        savefig(filename * @sprintf("_%04d",k))
        k += 1
    end
end

x = 1:N
plot_series(x,real(expvalues),filename,"xymodel plot",1)

#=
run(`convert -delay 5 -loop 0 $(filename)_\*.png $filename.gif`)
run(`rm $(filename)_\*.png`)
=#
