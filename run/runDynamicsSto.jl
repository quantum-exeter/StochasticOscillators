using SpiDy
using Statistics
using Distributions
using ProgressMeter
using NPZ

### Parameters ###
J = LorentzianSD(0.3^2, 0.5, 0.1) # prm3
ω0 = 1 # use this for all calculations - counter term is toggled in diffeqsolver
T = 0.1

### SpiDy paratmeters ###
noise = ClassicalNoise(T);
matrix = IsoCoupling(1.);
navg = 100000
σ0 = 0.5

Δt = 0.1
N = 2000
tspan = (0, N*Δt)
saveat = (0:1:N)*Δt

########################
########################

println("Starting...")

progress = Progress(navg);

solx = zeros(navg, 3, length(saveat))
solp = zeros(navg, 3, length(saveat))

Threads.@threads for i in 1:navg
    d = rand(Normal(0., sqrt(σ0)), 6)
    x0 = [d[1], d[2], d[3]]
    p0 = [d[4], d[5], d[6]]
    bfields = [bfield(N, Δt, J, noise),
               bfield(N, Δt, J, noise),
               bfield(N, Δt, J, noise)];
    sol = diffeqsolver(x0, p0, tspan, J, bfields, matrix; saveat=saveat, atol=1e-6, rtol=1e-6);
    solx[i, :, :] = sol[1:3, :]
    solp[i, :, :] = sol[4:6, :]
    next!(progress)
end

var_xx = dropdims(var(solx, dims=[1,2]), dims=(1,2))
var_xp = [cov(solx[:,1,k], solp[:,1,k]) for k in 1:length(saveat)]
var_px = [cov(solp[:,1,k], solx[:,1,k]) for k in 1:length(saveat)]
var_pp = dropdims(var(solp, dims=[1,2]), dims=(1,2))

########################
########################

npzwrite("./data/dynamics/new/var_t_sto_prm3_cl.npz",
    Dict("lambda^2" => J.α, 
         "omega_p" => J.ω0,
         "gamma" => J.Γ,
         "omega_0" => ω0,
         "T" => T,
         "dt" => Δt,
         "N" => N,
         "t" => saveat,
         "navg" => navg,
         "var_0" => σ0,
         "var_xx" => var_xx,
         "var_xp" => var_xp,
         "var_px" => var_px,
         "var_pp" => var_pp))