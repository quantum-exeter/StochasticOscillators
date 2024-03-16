using SpiDy
using Statistics
using Distributions
using ProgressBars
using NPZ

### Parameters ###
J = LorentzianSD(0.3^2, 0.5, 0.8) # prm6
# J = LorentzianSD(2^2, 0.5, 0.8) # prm7
ω0 = 1 # use this for all calculations - counter term is toggled in diffeqsolver
T = LinRange(0, 1, 10)
# T = 1

### SpiDy paratmeters ###
matrix = IsoCoupling(1.);
# navg = 20000;
navg = 1
σ0 = 0.5

Δt = 0.1
# N = 50_000
N = 100
tspan = (0., N*Δt)
saveat = ((N*4÷5):1:(5*N÷6))*Δt

########################
########################

println("Starting...")

var_xx = zeros(length(T));
var_xp = zeros(length(T));
var_px = zeros(length(T));
var_pp = zeros(length(T));

for n in ProgressBar(eachindex(T))
    noise = QuantumNoise(T[n]);
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
    end
    avg1 = Float64[]
    var1 = Float64[]
    cov1 = Float64[]
    for m in 1:length(saveat)-1
        n1 = m*navg 
        n2 = navg
        n12 = n1 + n2
        if m==1
            avg1 = [mean(solx[:, 1, m]), mean(solp[:, 1, m])]
            var1 = [var(solx[:, 1, m]), var(solp[:, 1, m])]
            cov1 = cov(solx[:, 1, m], solp[:, 1, m])   
        else
            avg2 = [mean(solx[:, 1, m+1]), mean(solp[:, 1, m+1])]
            var2 = [var(solx[:, 1, m+1]), var(solp[:, 1, m+1])]
            cov2 = cov(solx[:, 1, m+1], solp[:, 1, m+1])
            ####
            avg1 = [(n1*avg1[k] + n2*avg2[k])/n12 for k in 1:2]
            var1 = [(n1*var1[k] + n2*var2[k] + (n1*n2/n12)*(avg1[k] - avg2[k])^2)/n12 for k in 1:2]
            cov1 = (n1*cov1 + n2*cov2 + (n1*n2/n12)*(avg1[1] - avg2[1])*(avg1[2] - avg2[2]))/n12

        end
    end
    var_xx[n] = var1[1]
    var_xp[n] = cov1
    var_pp[n] = var1[2]
end

########################
########################

npzwrite("./data/var_T_sto_prm6_$(navg).npz",
    Dict("lambda^2" => J.α, 
         "omega_p" => J.ω0,
         "gamma" => J.Γ,
         "omega_0" => ω0,
         "T" => T,
         "dt" => Δt,
         "N" => N,
         "navg" => navg,
         "var_0" => σ0,
         "var_xx" => var_xx,
         "var_xp" => var_xp,
         "var_pp" => var_pp))
