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

### FUNCTION inParse ###
# Parse TOML file and return its data as variables
#
# Inputs:
# - filename: path to the TOML file
#
# Outputs:
# - id: id of this run
# - fls: array of data files
# - uis: unique id for each file
# - bts: array of beta values
# - flw: used flow
# - xi: weight of the plaquette action density
# - rgB: boolean whether to perform or nor renormalization 
#        group improvement (set xi correctly when doing so)
# - En: names of action density observables
# - Ev: flow time values of action density observables
# - dEn: names of derivative action density observables
# - dEv: flow time values of derivative action density observables
### --- ###
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

### FUNCTION derivate ###
# Central differences of y w.r.t. x
#
# Inputs:
# - y: y vector
# - x: x vector, same size as y
#
# Outputs:
# - dy: derivative vector
### --- ###
function derivate(y, x)
    dy = Array{Float64}(undef, size(y)[1], size(y)[2]-2)
    for i in 2:(length(x)-1)
        dy[:,i-1] = (y[:,i+1] .- y[:,i-1]) ./ (x[i+1] - x[i-1])
    end
    return dy
end

### FUNCTION loadXsl ###
# Reads JLD2 file and extract action density related quantities
# (TODO: topological charge)
#
# Inputs:
# - file: path to JLD2 file
# - discretization: either Ysl (clover) or Wsl (plaquette)
#
# Outputs:
# - tfw: flow time vector trimmed in the extremes
# - ttX: adimensional action density trimmed in the extremes
# - tdttX: log derivative of the adimensional action density
### --- ###
function loadXsl(file, discretization)
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

### FUNCTION r0aPoly ###
# Calculate r_0/a from \beta using functional
# polynomial form.
# (TODO: right now based on my Master's Thesis,
# the method is improvable and polynomials should coincide in \beta = 6)
#
# Inputs:
# - beta: value of beta (6/g_0^2)
#
# Outputs:
# - ur0a: uwreal object of r_0/a
### --- ###
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

### FUNCTION alphaMS ###
# Calculate \alpha_{MS} from \beta using
# r_0\Lambda_{MS} from Ramos, Dalla Brida.
# Function adapted from R function of Gregorio Herdoiza
#
# Inputs:
# - beta: value of \beta
# - mulr0: scalar to multiply r_0/a (default = 1)
# - powa: order of perturbation theory: 0, 1, 2, 3 (default = 3)
# - Nf: number of flavours (default = 0)
### --- ###
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

### FUNCTION dxi ###
# Modification to \xi for RG (assuming well selected xi)
# 
# Inputs:
# - beta: value of \beta
# - t: flow time
#
# Outputs:
# - \Delta \xi
### --- ###
function dxi(beta, t)
    return 27/56 * ((alphaMS(beta)/alphaMS(beta, 1/sqrt(8*t)))^(7/11) - 1.0)
end

###############################################
### GET THE VALUES NEEDED FOR EXTRAPOLATION ###
###############################################

### FUNCTION fileIteration ###
# Get the needed values for extrapolation for one data file
#
# Inputs:
# - fl: file path
# - bt: \beta value
# - xi: weight of plaquette
# - rgB: boolen whether to perform RG or not
# - En: names of action density observables
# - Ev: flow time values of action density observables
# - dEn: names of derivative action density observables
# - dEv: flow time values of derivative action density observables
#
# Outputs:
# - rt: flow time values before and after extrapolation point of AD
# - drt: flow time values before and after extrapolation point of dAD
# - rE: AD values before and after extrapolation point of AD
# - drE: dAD values before and after extrapolation point of dAD
### --- ###
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

### FUNCTION getObs! ###
# Extrapolate and save into dict
#
# Inputs:
# - dict: "Observable" dict to modify.
# - x: float array with the just before
#   and after values of the x axis.
# - y: uwreal array with the just before
#   and after values of the y axis.
# - refVal: reference value.
# - name: name of the observable
### --- ###
function getObs!(dict, x, y, refVal, name)
    r1, r2 = findall(y .< refVal)[end], findall(y .> refVal)[1]
    obs    = x[r1] + (refVal - y[r1]) * (y[r2] - y[r1])/(x[r2] - x[r1])
    uwerr(obs, wpm)
    dict[name] = Dict("ADerrors"     => obs,
                      "Value"        => value(obs),
                      "Uncertainty"  => err(obs),
                      "Correlations" => Dict())
end

### FUNCTION corrDicr! ###
# Add the correlations with the rest
# of the observables.
#
# Inputs:
# - dict: "Observable" dict to modify
### --- ###
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

############
### MAIN ###
############

function main()
    # Parse command line and input file
    inFile  = ARGS[1] # to be changed
    outFile = ARGS[2]
    id, fls, uis, bts, flw, xi, rgB, En, Ev, dEn, dEv = inParse(inFile)

    # Create Dict
    dictObs = Dict(id => [])

    # Iterate over data files
    for (i, (fl, bt, ui)) in enumerate(zip(fls, bts, uis))
        append!(dictObs[id], [Dict("UID" => ui, "Beta" => bt, "Flow" => flw, "Observables" => Dict())])
            
        rt, drt, rE, drE = fileIteration(fl, bt, xi, rgB, En, Ev, dEn, dEv)
        
        for (name, refVal) in zip(En, Ev)
            getObs!(dictObs[id][end]["Observables"], rt, rE,  refVal, name)
        end
        for (name, refVal) in zip(dEn, dEv)
            getObs!(dictObs[id][end]["Observables"], drt, drE, refVal, name)
        end
        corrDict!(dictObs[id][end]["Observables"])
    end

    # Create a dict copy without uwreals
    cdict = copy(dictObs)
    for i in eachindex(fls)
        for name in keys(cdict[id][i]["Observables"])
            delete!(cdict[id][i]["Observables"][name], "ADerrors")
        end
    end
	
    # JSON save
    dictJSON = isfile(outFile*".json") ? merge(JSON.parsefile(outFile*".json"), cdict) : cdict
    open(outFile*".json", "w") do io
        JSON.print(io, dictJSON)
    end
    
    # JLD2 save
    jldopen(outFile*".jld2", "a+") do file
        file[id] = dictObs[id]
    end
end
    
end

main()
