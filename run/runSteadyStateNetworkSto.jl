using SpiDy
using Statistics
using Distributions
using ProgressBars
using NPZ

########################
########################

### Parameters ###
J = LorentzianSD(0.3^2, 0.5, 0.8)
ω0 = 1

# T = LinRange(0.0, 1.0, 10)
T = 0.2
T_c = T
T_h = 10*T

nosc = 2
κ = 0.1
JH = Nchain(nosc, κ)

### SpiDy paratmeters ###
matrix = IsoCoupling(1.);
ntraj = 10000
σ0 = 0.5

Δt = 0.1
N = 150_000

tspan= (0., N*Δt)
saveat = (4*div(N,5):1:5*div(N,6))*Δt

########################
########################

println("Starting...")

x_traj = zeros(nosc, length(saveat), ntraj, length(T));
p_traj = zeros(nosc, length(saveat), ntraj, length(T));
v_traj = zeros(nosc, length(saveat), ntraj, length(T));
w_traj = zeros(nosc, length(saveat), ntraj, length(T));
b_traj = zeros(nosc, length(saveat), ntraj, length(T));
g_traj = zeros(nosc, length(saveat), ntraj, length(T));

current1ss = zeros(length(T))
current2ss = zeros(length(T))

lck = ReentrantLock()
for n in eachindex(T)
    noise_c = QuantumNoise(T_c[n]);
    noise_h = QuantumNoise(T_h[n]);
    Threads.@threads for i in ProgressBar(1:ntraj)
        x0 = rand(Normal(0., sqrt(σ0)), 3*nosc)
        p0 = rand(Normal(0., sqrt(σ0)), 3*nosc)
        bfields_c = [bfield(N, Δt, J, noise_c), bfield(N, Δt, J, noise_c), bfield(N, Δt, J, noise_c)];
        bfields_h = [bfield(N, Δt, J, noise_h), bfield(N, Δt, J, noise_h), bfield(N, Δt, J, noise_h)];
        sol = diffeqsolver(x0, p0, tspan, [J, J], [bfields_c, bfields_h], [[1,0],[0,1]], [matrix, matrix]; JH=JH, saveat=saveat, save_fields=true, alg=Vern7(), atol=1e-6, rtol=1e-6);
        solx = @view sol[1:3*nosc, :]
        solp = @view sol[1+3*nosc:6*nosc, :]
        solv = @view sol[1+6*nosc:(6*nosc+3*2), :]
        solw = @view sol[1+(6*nosc+3*2):(6*nosc+6*2), :]
        Threads.lock(lck) do
            for nn in 1:2
                x_traj[nn, :, i, n] = solx[1+(nn-1)*3, :]
                p_traj[nn, :, i, n] = solp[1+(nn-1)*3, :]
                v_traj[nn, :, i, n] = solv[1+(nn-1)*3, :]
                w_traj[nn, :, i, n] = solw[1+(nn-1)*3, :]
            end
            b_traj[1, :, i, n] = bfields_c[1].(saveat)
            b_traj[2, :, i, n] = bfields_h[1].(saveat)
            g_traj[1, :, i, n] = b_traj[1, :, i, n] + v_traj[1, :, i, n]
            g_traj[2, :, i, n] = b_traj[2, :, i, n] + v_traj[2, :, i, n]
        end
    end
    avg1 = Float64[]
    cov1 = Float64[]
    avg2 = Float64[]
    cov2 = Float64[]
    for m in 1:length(saveat)-1
        n1 = m*ntraj
        n2 = ntraj
        n12 = n1 + n2
        if m==1
            avg1 = [mean(g_traj[1, m, :, n]), mean(p_traj[1, m, :, n])]
            cov1 = cov(g_traj[1, m, :, n], p_traj[1, m, :, n])
            ####
            avg2 = [mean(g_traj[2, m, :, n]), mean(p_traj[2, m, :, n])]
            cov2 = cov(g_traj[2, m, :, n], p_traj[2, m, :, n])   
        else
            avg1′ = [mean(g_traj[1, m+1, :, n]), mean(p_traj[1, m+1, :, n])]
            cov1′ = cov(g_traj[1, m+1, :, n], p_traj[1, m+1, :, n])   
            avg2′ = [mean(g_traj[2, m+1, :, n]), mean(p_traj[2, m+1, :, n])]
            cov2′ = cov(g_traj[2, m+1, :, n], p_traj[2, m+1, :, n])   
            ####
            avg1 = [(n1*avg1[k] + n2*avg1′[k])/n12 for k in 1:2]
            cov1 = (n1*cov1 + n2*cov1′ + (n1*n2/n12)*(avg1[1] - avg1′[1])*(avg1[2] - avg1′[2]))/n12
            avg2 = [(n1*avg2[k] + n2*avg2′[k])/n12 for k in 1:2]
            cov2 = (n1*cov2 + n2*cov2′ + (n1*n2/n12)*(avg2[1] - avg2′[1])*(avg2[2] - avg2′[2]))/n12
        end
    end
    current1ss[n] = cov1
    current2ss[n] = cov2
end

########################
########################

npzwrite("./data/q_T_sto_network_$(ntraj).npz",
    Dict("lambda^2" => J.α, 
         "omega_p" => J.ω0,
         "gamma" => J.Γ,
         "omega_0" => ω0,
         "T" => T,
         "T_c" => T_c,
         "T_h" => T_h,
         "nosc" => nosc,
         "kappa" => κ,
         "dt" => Δt,
         "N" => N,
         "ntraj" => ntraj,
         "var_0" => σ0,
         "t" => saveat,
         "current1ss" => current1ss,
         "current2ss" => current2ss));
