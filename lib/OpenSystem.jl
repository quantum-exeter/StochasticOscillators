module OpenSystem

    ### Import packages ###
    using SpiDy
    using SpectralDensities
    using QuantumUtilities
    using InverseLaplace
    using QuadGK
    using Cuba
    using ForwardDiff
    using LinearAlgebra

    ### Inclusions ###
    include("spectrum.jl")
    include("variances.jl")
    include("coupling_regimes.jl")
    include("network.jl")

    ####################################
    ####################################
    ####################################

    ### Exports ###
    export LorentzianSD, reorganisation_energy, σxx, σxp, σpp, σxxUw, σxxUs, σxxCl, σxpCl, σppCl, coupling_regimes, Q
    
end