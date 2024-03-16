######################
#### variances.jl ####
######################

### Steady state ###
function σxx(J::LorentzianSD, ω0eff, T)
    integrand(ω) = power(J, ω, T)*g₀(J, ω, T, ω0eff)*g₀T(J, ω, T, ω0eff)
    out = quadgk(ω -> integrand(ω), 0, Inf)[1]
    return realifclose(out, tol=1e-8)
end

σxp(J::LorentzianSD, ω0eff, T) = 0.0

function σpp(J::LorentzianSD, ω0eff, T)
    integrand(ω) = ω^2*power(J, ω, T)*g₀(J, ω, T, ω0eff)*g₀T(J, ω, T, ω0eff)
    out = quadgk(ω -> integrand(ω), 0, Inf)[1]
    return realifclose(out, tol=1e-8)
end

### Steady state limits ###
σxxUw(ω0, T) = (1/tanh(ω0/(2*T)))/(2*ω0)
σxxUs(ω0, T) = T/(ω0^2)

### Steady state (classical) ###
function σxxCl(J::LorentzianSD, ω0eff, T)
    integrand(ω) = powerCl(J, ω, T)*g₀(J, ω, T, ω0eff)*g₀T(J, ω, T, ω0eff)
    out = quadgk(ω -> integrand(ω), 0, Inf)[1]
    return realifclose(out)
end

σxpCl(J::LorentzianSD, ω0eff, T) = 0.0

function σppCl(J::LorentzianSD, ω0eff, T)
    integrand(ω) = ω^2*powerCl(J, ω, T)*g₀(J, ω, T, ω0eff)*g₀T(J, ω, T, ω0eff)
    out = quadgk(ω -> integrand(ω), 0, Inf)[1]
    return realifclose(out)
end

### Dynamics ###
function σxx(J::LorentzianSD, ω0eff, T, t, xx0, xp0, pp0)
    function integrand(x, f)
        ω(ω′) = ω′/(1 - ω′)
        fn = real(power(J, ω(x[1]), T)*(cis(-ω(x[1])*x[2]*t)*g₂(J, ω0eff, x[2]*t)*cis(ω(x[1])*x[3]*t)*g₂(J, ω0eff, x[3]*t))*1/(1 - x[1])^2*t^2)
        f[1], f[2] = reim(fn)
    end
    init = g₁(J, ω0eff, t)^2*xx0 + 2*g₁(J, ω0eff, t)*g₂(J, ω0eff, t)*xp0 + g₂(J, ω0eff, t)^2*pp0 
    out = cuhre(integrand, 3, 2)[1][1] + init
    return realifclose(out)
end

function σxp(J::LorentzianSD, ω0eff, T, t, xx0, xp0, pp0)
    function integrand(x, f)
        ω(ω′) = ω′/(1 - ω′)
        fn = real(power(J, ω(x[1]), T)*(cis(-ω(x[1])*x[2]*t)*g₂(J, ω0eff, x[2]*t)*cis(ω(x[1])*x[3]*t)*g₁(J, ω0eff, x[3]*t))*1/(1 - x[1])^2*t^2)
        f[1], f[2] = reim(fn)
    end
    init = g₁(J, ω0eff, t)*g₃(J, ω0eff, t)*xx0 + (g₁(J, ω0eff, t)^2 + g₂(J, ω0eff, t)*g₃(J, ω0eff, t))*xp0 + g₂(J, ω0eff, t)*g₁(J, ω0eff, t)*pp0 
    out = cuhre(integrand, 3, 2)[1][1] + init
    return realifclose(out)
end

function σpp(J::LorentzianSD, ω0eff, T, t, xx0, xp0, pp0)
    function integrand(x, f)
        ω(ω′) = ω′/(1 - ω′)
        fn = real(power(J, ω(x[1]), T)*(cis(-ω(x[1])*x[2]*t)*g₁(J, ω0eff, x[2]*t)*cis(ω(x[1])*x[3]*t)*g₁(J, ω0eff, x[3]*t))*1/(1 - x[1])^2*t^2)
        f[1], f[2] = reim(fn)
    end
    init = g₃(J, ω0eff, t)^2*xx0 + 2*g₃(J, ω0eff, t)*g₁(J, ω0eff, t)*xp0 + g₁(J, ω0eff, t)^2*pp0 
    out = cuhre(integrand, 3, 2)[1][1] + init
    return realifclose(out)
end

### Dynamics (classical) ###
function σxxCl(J::LorentzianSD, ω0eff, T, t, xx0, xp0, pp0)
    function integrand(x, f)
        ω(ω′) = ω′/(1 - ω′)
        fn = real(powerCl(J, ω(x[1]), T)*(cis(-ω(x[1])*x[2]*t)*g₂(J, ω0eff, x[2]*t)*cis(ω(x[1])*x[3]*t)*g₂(J, ω0eff, x[3]*t))*1/(1 - x[1])^2*t^2)
        f[1], f[2] = reim(fn)
    end
    init = g₁(J, ω0eff, t)^2*xx0 + 2*g₁(J, ω0eff, t)*g₂(J, ω0eff, t)*xp0 + g₂(J, ω0eff, t)^2*pp0 
    out = cuhre(integrand, 3, 2)[1][1] + init
    return realifclose(out)
end

function σxpCl(J::LorentzianSD, ω0eff, T, t, xx0, xp0, pp0)
    function integrand(x, f)
        ω(ω′) = ω′/(1 - ω′)
        fn = real(powerCl(J, ω(x[1]), T)*(cis(-ω(x[1])*x[2]*t)*g₂(J, ω0eff, x[2]*t)*cis(ω(x[1])*x[3]*t)*g₁(J, ω0eff, x[3]*t))*1/(1 - x[1])^2*t^2)
        f[1], f[2] = reim(fn)
    end
    init = g₁(J, ω0eff, t)*g₃(J, ω0eff, t)*xx0 + (g₁(J, ω0eff, t)^2 + g₂(J, ω0eff, t)*g₃(J, ω0eff, t))*xp0 + g₂(J, ω0eff, t)*g₁(J, ω0eff, t)*pp0 
    out = cuhre(integrand, 3, 2)[1][1] + init
    return realifclose(out)
end

function σppCl(J::LorentzianSD, ω0eff, T, t, xx0, xp0, pp0)
    function integrand(x, f)
        ω(ω′) = ω′/(1 - ω′)
        fn = real(powerCl(J, ω(x[1]), T)*(cis(-ω(x[1])*x[2]*t)*g₁(J, ω0eff, x[2]*t)*cis(ω(x[1])*x[3]*t)*g₁(J, ω0eff, x[3]*t))*1/(1 - x[1])^2*t^2)
        f[1], f[2] = reim(fn)
    end
    init = g₃(J, ω0eff, t)^2*xx0 + 2*g₃(J, ω0eff, t)*g₁(J, ω0eff, t)*xp0 + g₁(J, ω0eff, t)^2*pp0 
    out = cuhre(integrand, 3, 2)[1][1] + init
    return realifclose(out)
end