using ProgressMeter
using NPZ

include("../lib/OpenSystem.jl")
using .OpenSystem

### Parameters ###
# J = LorentzianSD(0.3^2, 0.5, 0.8) # prm6
J = LorentzianSD(2^2, 0.5, 0.8) # prm7

ω0 = 1
# ω0eff = ω0 # no counter term
ω0eff = sqrt(1 + 2*reorganisation_energy(J)) # counter term

T = LinRange(0, 1, 50)

########################
########################

println("Starting...")

progress = Progress(length(T));

var_xx_gibbs = zeros(length(T));
var_xx = zeros(length(T));
var_xp = zeros(length(T));
var_px = zeros(length(T));
var_pp = zeros(length(T));

Threads.@threads for n in eachindex(T)
    var_xx_gibbs[n] = σxxUw(ω0, T[n])
    var_xx[n] = σxx(J, ω0eff, T[n])
    var_xp[n] = σxp(J, ω0eff, T[n])
    var_px[n] = var_xp[n]
    var_pp[n] = σpp(J, ω0eff, T[n])
    next!(progress)
end

########################
########################

npzwrite("./data/steady_state/var_T_osys_prm7.npz",
    Dict("lambda^2" => J.α, 
         "omega_p" => J.ω0,
         "gamma" => J.Γ,
         "omega_0" => ω0,
         "omega_0_eff" => ω0eff,
         "T" => T,
         "var_xx_gibbs" => var_xx_gibbs,
         "var_xx" => var_xx,
         "var_xp" => var_xp,
         "var_px" => var_px,
         "var_pp" => var_pp))