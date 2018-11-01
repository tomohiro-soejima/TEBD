include("./THIM_script.jl")

using .THIM_TEBD

N = 64
#these hx and hz follows Gil Rafael's convention
hx = 0.9045
hz = 0.8090
#=
THIM_TEBD.THIM_TEBD_energy(hx,hz,N,12,400,"TFIM_energy_D_12_1")
THIM_TEBD.THIM_TEBD_energy(hx,hz,N,12,400,"TFIM_energy_stochastic_D_12_1",stochastic=true)

THIM_TEBD.THIM_TEBD_energy(0.5,1,N,12,400,"TFIM_off_energy_D_12_1")
THIM_TEBD.THIM_TEBD_energy(0.5,1,N,12,400,"TFIM__off_energy_stochastic_D_12_1",stochastic=true)
=#

THIM_TEBD.THIM_TEBD_energy(hx,hz,N,24,100,"TFIM_energy_D_24_1")
#=
THIM_TEBD.THIM_TEBD_energy(hx,hz,N,24,400,"TFIM_energy_stochastic_D_24_1",stochastic=true)

THIM_TEBD.THIM_TEBD_energy(0.5,1,N,24,400,"TFIM_off_energy_D_24_1")
=#
THIM_TEBD.THIM_TEBD_energy(0.5,1,N,24,400,"TFIM_off_energy_stochastic_D_24_1",stochastic=true)
