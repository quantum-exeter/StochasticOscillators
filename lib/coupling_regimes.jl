#####################
#### coupling_regimes.jl ##
#####################

function coupling_regimes(σmf, σuw, σwk, σus, λ, tol)
    λuw = []
    λwk = []
    λus = []
    for i in 1:length(λ)
        mf = σmf[i]
        err_uw = abs((mf - σuw[i])/(mf))
        err_wk = abs((mf - σwk[i])/(mf))
        err_us = abs((mf - σus[i])/(mf))
        if err_uw > tol && isempty(λuw)
            λuw = λ[i]
        end
        if err_wk > tol && isempty(λwk)
            λwk = λ[i]
        end
        if err_us ≤ tol && isempty(λus)
            λus = λ[i]
        end
    end
    return(λuw, λwk, λus)
end