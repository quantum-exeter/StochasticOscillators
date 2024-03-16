#####################
#### spectrum.jl ####
#####################

### Power spectrum ###
power(J::LorentzianSD, ω, T) = sd(J, ω)*coth(ω/(2*T))
powerCl(J::LorentzianSD, ω, T) = sd(J, ω)*2*T/ω

### Green's functions ###
g₀(J::LorentzianSD, ω, T, ω0) = 1/(-(J.α)/(J.ω0^2 - 1im*J.Γ*ω - ω^2) - ω^2 + ω0^2)
g₀T(J::LorentzianSD, ω, T, ω0) = conj(g₀(J, ω, T, ω0))

function g₁(J::LorentzianSD, ω, t)
    g₁s(s) =  s/(-(J.α/(J.ω0^2 + J.Γ*s + s^2)) + s^2 + ω^2)
    g₁t = Talbot(s -> g₁s(s), 80)
    return g₁t(t)
end

function g₂(J::LorentzianSD, ω, t)
    g₂s(s) =  1/(-(J.α/(J.ω0^2 + J.Γ*s + s^2)) + s^2 + ω^2)
    g₂t = Talbot(s -> g₂s(s), 80)
    return g₂t(t)
end

function g₃(J::LorentzianSD, ω, t)
    f(x) = g₁(J, ω, x)
    return ForwardDiff.derivative(f, t)
end