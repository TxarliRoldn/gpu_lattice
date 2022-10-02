using ADerrors
using CSV
using DataFrames
using DataStructures
using JLD2
using SpecialFunctions
using Statistics
using TOML

wpm    = Dict{String,Vector{Float64}}()
chosen = [-1, 1.5, -1, -1]

function alphaMS(r0, powa=3, Nf=0)
    lambdaMS = uwreal([0.660, 0.011], "reference_lambda_MS")
    logmuLam = log(r0^2/lambdaMS^2)

    beta0 = ( 33.0 - 2.0 * Nf )/( 12.0 * pi )
    beta1 = ( 153.0 - 19.0 * Nf )/( 24.0 * pi^2 )
    beta2 = ( 77139.0 - 15099.0 * Nf + 325.0 * Nf^2 )/( 3456.0 * pi^3 )
    beta3 = 1/( 4.0 * pi )^4*( 149753.0/6.0 + 3564.0 * zeta(3) - ( 1078361.0 / 162.0 + 6508.0 / 27.0 * zeta(3) ) * Nf  + ( 50065.0 / 162.0 + 6472.0 / 81.0 * zeta(3) ) * Nf^2 + 1093.0 / 729.0 * Nf^3 )   

    a_N0LO =  1.0 / (beta0 * logmuLam)

    a_N1LO = -beta1 * log(logmuLam) / ( beta0^3 * (logmuLam)^2 )

    a_N2LO = 1.0 / ( beta0^3 * (logmuLam)^3 ) * ( beta1^2 / beta0^2 * ( ( log(logmuLam) )^2 - log(logmuLam) - 1.0 ) + beta2 / beta0 )
    
    a_N3LO = 1.0 / ( beta0^4 * (logmuLam)^4 ) * ( beta1^3/beta0^3 * ( -( log(logmuLam) )^3 + 5.0/2.0 * ( log(logmuLam) )^2 + 2.0 * log(logmuLam) -1.0/2.0 ) -3.0 * beta1*beta2/beta0^2* log(logmuLam) + beta3 / ( 2.0 * beta0 ) )

    alpha_ms = a_N0LO*(powa >= 0) + a_N1LO*(powa >= 1) + a_N2LO*(powa >= 2) + a_N3LO*(powa >= 3)
    return alpha_ms
end

dchi(r0, t) = 27/56 * ((alphaMS(r0)/alphaMS(r0/sqrt(8*t)))^(7/11) - 1.0)

function getObs(y, x, refVal, mapfun=nothing)
    r   = [findall(y .< refVal)[end], findall(y .> refVal)[1]]
    obs = (x[r[1]]*(y[r[2]]-refVal) - x[r[2]]*(y[r[1]]-refVal))/(y[r[2]] - y[r[1]])
    if mapfun !== nothing
        obs = mapfun(obs)
    end
    uwerr(obs, wpm)
    return value(obs), err(obs)
end

function main()
    df = DataFrame()
    params = TOML.parsefile(ARGS[1])

    volumes  = params["General"]["volumes"]
    prefixes = params["General"]["flows"]
    chis     = params["General"]["coefs"]
    r0s      = Array{uwreal}(undef, length(volumes))
    if params["General"]["RG"]
        for (i, volume) in enumerate(volumes)
            global wpm["r0a"*volume] = chosen
            r0s[i] = uwreal([params["General"]["sommer_scales"][i], params["General"]["err_sommer_scales"][i]], "r0a"*volume)
            uwerr(r0s[i], wpm)
        end
    end

    ttENames  = params["Action_density"]["names"]
    ttEValues = params["Action_density"]["values"]
    ttEMaps   = params["Action_density"]["functions"]

    tdttENames  = params["Derivative_action_density"]["names"]
    tdttEValues = params["Derivative_action_density"]["values"]
    tdttEMaps   = params["Derivative_action_density"]["functions"]

    for (prefix, chi) in zip(prefixes, chis)
        for (j, volume) in enumerate(volumes)
            tchi = 0.0

            observables = OrderedDict()
            observables["id"]     = params["General"]["id"]
            observables["flow"]   = prefix
            observables["volume"] = volume
            
            global wpm["plaquette"*prefix*volume] = chosen
            global wpm["clover"*prefix*volume]    = chosen

            fileY = jldopen("JLD2_DIR/"*prefix*"Y"*volume*".jld2", "r")
            fileW = jldopen("JLD2_DIR/"*prefix*"W"*volume*".jld2", "r")
            nmeas = size(fileY["ttE"])[1]
            nflow = size(fileY["ttE"])[2]
            tfw   = Array{Float64}(undef, 2*length(ttEValues))
            dtfw  = Array{Float64}(undef, 2*length(tdttEValues))
            tte   = Array{uwreal}(undef, 2*length(ttEValues))
            tdtte = Array{uwreal}(undef, 2*length(tdttEValues))

            ttEtemp0   = 0
            tdttEtemp0 = 0
            ttEcont    = 1
            tdttEcont  = 1
            for i in 1:nflow
                if params["General"]["RG"]
                    tchi = chi + dchi(r0s[j], fileY["tfw"][i])
                else
                    tchi = chi
                end

                ttEtemp1   = tchi*mean(fileW["ttE"][:,i]) + (1-tchi)*mean(fileY["ttE"][:,i])
                tdttEtemp1 = tchi*mean(fileW["tdttE"][:,i]) + (1-tchi)*mean(fileY["tdttE"][:,i])

                if ttEcont != length(ttEValues)+1
                    if (ttEtemp0 < ttEValues[ttEcont]) && (ttEtemp1 > ttEValues[ttEcont])
                        tte[2*ttEcont]   = tchi*uwreal(fileW["ttE"][:,i], "plaquette"*prefix*volume) + (1-tchi)*uwreal(fileY["ttE"][:,i], "clover"*prefix*volume)
                        tte[2*ttEcont-1] = tchi*uwreal(fileW["ttE"][:,i-1], "plaquette"*prefix*volume) + (1-tchi)*uwreal(fileY["ttE"][:,i-1], "clover"*prefix*volume)
                        tfw[2*ttEcont]   = fileY["tfw"][i]
                        tfw[2*ttEcont-1] = fileY["tfw"][i-1]
                        ttEcont += 1
                    end
                end
                if tdttEcont != length(tdttEValues)+1
                    if (tdttEtemp0 < tdttEValues[tdttEcont]) && (tdttEtemp1 > tdttEValues[tdttEcont])
                        tdtte[2*tdttEcont]   = tchi*uwreal(fileW["tdttE"][:,i], "plaquette"*prefix*volume) + (1-tchi)*uwreal(fileY["tdttE"][:,i], "clover"*prefix*volume)
                        tdtte[2*tdttEcont-1] = tchi*uwreal(fileW["tdttE"][:,i-1], "plaquette"*prefix*volume) + (1-tchi)*uwreal(fileY["tdttE"][:,i-1], "clover"*prefix*volume)
                        dtfw[2*tdttEcont]    = fileY["tfw"][i]
                        dtfw[2*tdttEcont-1]  = fileY["tfw"][i-1]
                        tdttEcont += 1
                    end
                end

                ttEtemp0   = ttEtemp1
                tdttEtemp0 = tdttEtemp1
            end
            close(fileY)
            close(fileW)

            for (name, refVal, mapfun) in zip(ttENames, ttEValues, ttEMaps)
                if mapfun != 0
                    observables[name], observables["err_"*name] = getObs(tte, tfw, refVal, getfield(Main, Symbol(mapfun)))
                else
                    observables[name], observables["err_"*name] = getObs(tte, tfw, refVal)
                end
            end
            for (name, refVal, mapfun) in zip(tdttENames, tdttEValues, tdttEMaps)
                if mapfun != 0
                    observables[name], observables["err_"*name] = getObs(tdtte, tfw, refVal, getfield(Main, Symbol(mapfun)))
                else
                    observables[name], observables["err_"*name] = getObs(tdtte, tfw, refVal)
                end
            end

            if !params["General"]["append"] && (prefix == prefixes[1]) && (volume == volumes[1])
                df = DataFrame(observables)
            elseif params["General"]["append"] && (prefix == prefixes[1]) && (volume == volumes[1])
                df = CSV.read(ARGS[2], DataFrame)
                push!(df, observables)
            else
                push!(df, observables)
            end
        end
    end
    CSV.write(ARGS[2], df)
end

main()