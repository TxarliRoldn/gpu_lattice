using ADerrors
using JLD2
using JSON
using LinearAlgebra
using SpecialFunctions
using Statistics
using TOML

wpm    = Dict{String,Vector{Float64}}()
chosen = [-1, 1.5, -1, -1]

###########################
### PARSE THE TOML FILE ###
###########################

function inParse(filename)
    params = TOML.parsefile(filename)

    # Global field
    id  = params["id"]
    fls = params["files"]
    uis = params["uids"]

    # Simulation info
    bts = params["Simulation"]["betas"]
    flw = params["Simulation"]["flow"]

    # Analysis peculiarities
    xi  = params["Analysis"]["xi"]
    rgB = params["Analysis"]["rg"]

    # Action density
    En = params["Action_density"]["names"]
    Ev = params["Action_density"]["values"]

    # Derivative of action density
    dEn = params["Derivative_action_density"]["names"]
    dEv = params["Derivative_action_density"]["values"]

    return id, fls, uis, bts, flw, xi, rgB, En, Ev, dEn, dEv
end

#############################################
### FUNCTIONS TO INTERACT WITH JLD2 FILES ###
#############################################

function derivate(y, x)
    dy = Array{Float64}(undef, size(y)[1], size(y)[2]-2)
    for i in 2:(length(x)-1)
        dy[:,i-1] = (y[:,i+1] .- y[:,i-1]) ./ (x[i+1] - x[i-1])
    end
    return dy
end

function loadXsl(file, discretization) # discretization = Ysl or Wsl
	data  = jldopen(file, "r")
	nmeas = parse(Int64, keys(data[discretization])[end])
	sfq   = parse(Int64, keys(data[discretization])[1])
	nfw   = size(data["tfw"])[1]
	tfw   = data["tfw"][:]

	Xsl = Array{Float64}(undef, nmeas, nfw)
	for (i,key) in enumerate(keys(data[discretization]))
		Xsl[(i-1)*sfq+1:i*sfq, :] = mean(data[discretization][key][:,:,:,:], dims=3)[:,:,1,1]
	end
	close(data)

    ttX    = tfw'.^2 .* Xsl
    tdttX  = tfw[2:end-1]' .* derivate(ttX, tfw)
    ttX    = ttX[:,2:end-1]
	tfw    = tfw[2:end-1]

	return tfw, ttX, tdttX
end

###########################################
### FUNCTIONS FOR RENORMALIZATION GROUP ###
###########################################

function r0aPoly(beta)
    global wpm["r0a.$beta"] = chosen
    if (beta < 5.7) || (beta > 6.9)
        exit()
    elseif beta < 6
        r0a  = exp(1.6805 + 1.7139*(beta - 6) - 0.8155*(beta - 6)^2 + 0.6667*(beta - 6)^3)
        dr0a = (0.003/0.87 * (beta - 5.7) + 0.003) * r0a
        ur0a = uwreal([r0a, dr0a], "r0a.$beta")
        uwerr(ur0a)
        return ur0a
    else
        r0a  = 9.5130 + 12.965*(beta - 6.38) + 6.982*(beta - 6.38)^2 + 4.16*(beta - 6.38)^3
        dr0a = (0.01 * (beta - 6) + 0.001) * r0a
        ur0a = uwreal([r0a, dr0a], "r0a.$beta")
        uwerr(ur0a)
        return ur0a
    end
end

function alphaMS(beta, mulr0=1, powa=3, Nf=0)
    r0 = r0aPoly(beta) * mulr0
    lambdaMS  = uwreal([0.660, 0.011], "refLamMS")
    logmuLam  = log(r0^2/lambdaMS^2)
    llogmuLam = log(logmuLam)

    zeta3 = zeta(3)
    beta0 = (33.0 - 2.0*Nf)/(12.0*pi)
    beta1 = (153.0 - 19.0*Nf)/(24.0*pi^2)
    beta2 = (77139.0 - 15099.0*Nf + 325.0*Nf^2)/(3456.0*pi^3)
    beta3 = (149753.0/6.0 + 3564.0*zeta3 - (1078361.0/162.0 + 6508.0/27.0*zeta3)*Nf + (50065.0/162.0 + 6472.0/81.0*zeta3)*Nf^2 + 1093.0/729.0*Nf^3)/(4.0*pi)^4

    aN0LO = 1.0/(beta0*logmuLam)
    aN1LO = -beta1*llogmuLam/(beta0^3*logmuLam^2)
    aN2LO = 1.0/(beta0^3*logmuLam^3)*(beta1^2/beta0^2*(llogmuLam^2 - llogmuLam - 1.0) + beta2/beta0)
    aN3LO = 1.0/(beta0^4*logmuLam^4)*(beta1^3/beta0^3*(-llogmuLam^3 + 5.0/2.0*llogmuLam^2 + 2.0*llogmuLam -1.0/2.0) - 3.0*beta1*beta2/beta0^2*llogmuLam + beta3/(2.0*beta0))

    alphaMS = aN0LO*(powa >= 0) + aN1LO*(powa >= 1) + aN2LO*(powa >= 2) + aN3LO*(powa >= 3)
    return alphaMS
end

dxi(beta, t) = 27/56 * ((alphaMS(beta)/alphaMS(beta, 1/sqrt(8*t)))^(7/11) - 1.0)

###############################################
### GET THE VALUES NEEDED FOR EXTRAPOLATION ###
###############################################

function fileIteration(fl, bt, xi, rgB, En, Ev, dEn, dEv)
    nE, ndE  = length(Ev), length(dEv)

    # Load the JLD2 files
    t, Y, dY = loadXsl(fl, "Ysl")
    t, W, dW = loadXsl(fl, "Wsl")
    nMs, nFw = size(Y)

    # Preallocate arrays
    rt  = Array{Float64}(undef, 2*nE)
    drt = Array{Float64}(undef, 2*ndE)
    rE  = Array{uwreal}(undef,  2*nE)
    drE = Array{uwreal}(undef,  2*ndE)

    # Fast iteration for the flow times
    Et0, dEt0, Ec, dEc = 0.0, 0.0, 1, 1
    for i in 1:nFw
        tX  = xi + rgB * dxi(bt, t[i]) # Adjusted xi for RG
        otX = 1.0 - tX

        Et1  = tX*mean(W[:,i])  + otX*mean(Y[:,i])
        dEt1 = tX*mean(dW[:,i]) + otX*mean(dY[:,i])

        if Ec != nE + 1
            if (Et0 < Ev[Ec]) && (Et1 > Ev[Ec])
                rE[2*Ec]   = tX*uwreal(W[:,i], "pl.$fl")   + otX*uwreal(Y[:,i], "cl.$fl")
                rE[2*Ec-1] = tX*uwreal(W[:,i-1], "pl.$fl") + otX*uwreal(Y[:,i-1], "cl.$fl")
                rt[2*Ec]   = t[i]
                rt[2*Ec-1] = t[i-1]
                Ec += 1
            end
        end
        if dEc != ndE + 1
            if (dEt0 < dEv[dEc]) && (dEt1 > dEv[dEc])
                drE[2*dEc]   = tX*uwreal(dW[:,i], "pl.$fl")   + otX*uwreal(dY[:,i], "cl.$fl")
                drE[2*dEc-1] = tX*uwreal(dW[:,i-1], "pl.$fl") + otX*uwreal(dY[:,i-1], "cl.$fl")
                drt[2*dEc]   = t[i]
                drt[2*dEc-1] = t[i-1]
                dEc += 1
            end
        end
        Et0, dEt0 = Et1, dEt1
    end

    return rt, drt, rE, drE
end

#########################################
### FUNCTIONS TO BUILD THE DICTIONARY ###
#########################################

function getObs!(dict, y, x, refVal, name)
    r1, r2 = findall(y .< refVal)[end], findall(y .> refVal)[1]
    obs    = x[r1] + (refVal - y[r1]) * (y[r2] - y[r1])/(x[r2] - x[r1])
    uwerr(obs, wpm)
    dict[name] = Dict("ADerr"        => obs,
                      "Value"        => value(obs),
                      "Uncertainty"  => err(obs),
                      "Correlations" => Dict())
end

function corrDict!(dict)
    for n1 in keys(dict)
        for n2 in keys(dict)
            if n1 == n2
                continue
            end
            tempMat = ADerrors.cov([dict[n1]["ADerr"], dict[n2]["ADerr"]], wpm)
            dict[n1]["Correlations"][n2] = tempMat[1,2]/sqrt(tempMat[1,1]*tempMat[2,2])
        end
    end
end

function cleanDict!(dict)
    for name in keys(dict)
        delete!(dict[name], "ADerr")
    end
end

function main()
    inFile  = ARGS[1] # to be changed
    outFile = ARGS[2]
    id, fls, uis, bts, flw, xi, rgB, En, Ev, dEn, dEv = inParse(inFile)

    dictObs = Dict(id => [])

    for (i, (fl, bt, ui)) in enumerate(zip(fls, bts, uis))
        append!(dictObs[id], [Dict("UID" => ui, "Beta" => bt, "Flow" => flw, "Observables" => Dict())])
            
        rt, drt, rE, drE = fileIteration(fl, bt, xi, rgB, En, Ev, dEn, dEv)
        
        for (name, refVal) in zip(En, Ev)
            getObs!(dictObs[id][end]["Observables"], rE,  rt,  refVal, name)
        end
        for (name, refVal) in zip(dEn, dEv)
            getObs!(dictObs[id][end]["Observables"], drE, drt, refVal, name)
        end
        corrDict!(dictObs[id][end]["Observables"])
        cleanDict!(dictObs[id][end]["Observables"])
    end

    dictSave = isfile(outFile) ? merge(JSON.parsefile(outFile), dictObs) : dictObs
    open(outFile, "w") do io
        JSON.print(io, dictSave)
    end
end

main()
