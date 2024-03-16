using ProgressMeter
using NPZ

include("../lib/OpenSystem.jl")
using .OpenSystem

### Parameters ###
J = LorentzianSD(0.3^2, 0.5, 0.1) # prm3

ω0 = 1
# ω0eff = ω0 # no counter term
ω0eff = sqrt(1 + 2*reorganisation_energy(J)) # counter term

T = 1

Δt = 0.1;
N = 200;
tspan = (0., N*Δt);
saveat = (0:1:N)*Δt;

xx0 = 0.5
xp0 = 0.
pp0 = 0.5

########################
########################

println("Starting...")

progress = Progress(length(saveat));

var_xx = zeros(length(saveat));
var_xp = zeros(length(saveat));
var_px = zeros(length(saveat));
var_pp = zeros(length(saveat));

Threads.@threads for n in eachindex(saveat)
    var_xx[n] = σxx(J, ω0eff, T, saveat[n], xx0, xp0, pp0)
    var_xp[n] = σxp(J, ω0eff, T, saveat[n], xx0, xp0, pp0)
    var_px[n] = var_xp[n]
    var_pp[n] = σpp(J, ω0eff, T, saveat[n], xx0, xp0, pp0)
    next!(progress)
end

########################
########################

npzwrite("./data/dynamics/var_t_osys_prm3.npz",
    Dict("lambda^2" => J.α, 
         "omega_p" => J.ω0,
         "gamma" => J.Γ,
         "omega_0" => ω0,
         "omega_0_eff" => ω0eff,
         "T" => T,
         "dt" => Δt,
         "N" => N,
         "t" => saveat,
         "var_xx" => var_xx,
         "var_xp" => var_xp,
         "var_px" => var_px,
         "var_pp" => var_pp))