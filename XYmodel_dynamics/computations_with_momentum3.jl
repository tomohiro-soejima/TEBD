include("./XYmodel_dynamics.jl")
using .dynamics
using Plots
using Printf
using Profile
using LinearAlgebra

#filename
filename = "xy_model_with_momentum3"

#initialize
N = 101
x0 = 51
sigma = 5
D = 20
T = pi
Nt = 5
h_list = zeros(Float64,N)
k = 1/10
a = 1

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
P = [exp(-(xi-x0)^2/(2*sigma^2)+im*xi*k) for xi in 1:N]
MPO = dynamics.VidalTEBD.make_superpositionMPO(U1,P)
#make all up product state
zeros(Complex{Float64},2,2)
PS = zeros(Float64,(2,N))
PS[1,:] = ones(Float64,N)
MPS2 = dynamics.VidalTEBD.make_productVidalMPS(PS,D)
#Apply MPO
MPS3 = dynamics.VidalTEBD.do_MPOonMPS(MPS2,MPO)
MPS = dynamics.VidalTEBD.convert_to_Vidal(MPS3)

@profile expvalues = dynamics.VidalTEBD.getTEBDexpvalue!(MPS,H,T,Nt,O)
x = 1:(Nt+1)
plot(x,real(expvalues))
savefig(filename*".png")#

function plot_series(x,data,filename,title_name,sep=1)
    u = maximum(data)
    k = 1
    for i in 1:sep:size(data)[1]
        plot(x, real(data[i,:]),title = title_name,lw =3)
        full_title = @sprintf(" t = %.2f pi", 10*(i-1)/(size(data)[1]-1))
        xlabel!("Position")
        ylabel!("1/2 - <Sz>")
        ylims!((-u*1.2,u*1.2))
        title!(title_name*full_title)
        savefig(filename * @sprintf("_%04d",k))
        k += 1
    end
end

x = 1:N
@time plot_series(x,real(expvalues),filename,"xymodel plot",1)
#=
run(`convert -delay 5 -loop 0 $(filename)_\*.png $filename.gif`)
run(`rm $(filename)_\*.png`)
=#
