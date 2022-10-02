using JLD2
using Statistics

if !isdir("JLD2_DIR")
	mkpath("JLD2_DIR")
end

function derivate(y, x)
    dy = Array{Float64}(undef, size(y)[1], size(y)[2]-2)
    for i in 2:(length(x)-1)
        dy[:,i-1] = (y[:,i+1] .- y[:,i-1]) ./ (x[i+1] - x[i-1])
    end
    return dy
end

function saveYsl(file)
	data  = jldopen(file, "r")
	nmeas = parse(Int64, keys(data["Ysl"])[end])
	sfq   = parse(Int64, keys(data["Ysl"])[1])
	nfw   = size(data["tfw"])[1]
	lsize = [data["attrs"]["L_$i"] for i in 1:4]
	tfw   = data["tfw"][:]

	Ysl = Array{Float64}(undef, nmeas, nfw)
	for (i,key) in enumerate(keys(data["Ysl"]))
		Ysl[(i-1)*sfq+1:i*sfq, :] = mean(data["Ysl"][key][:,:,:,:], dims=3)[:,:,1,1]
	end
	close(data)

    sysize = prod(string.(lsize) .* ["x", "x", "x", ""])
    ttY    = tfw'.^2 .* Ysl
    tdttY  = tfw[2:end-1]' .* derivate(ttY, tfw)
    ttY    = ttY[:,2:end-1]
	tfw    = tfw[2:end-1]

	return tfw, ttY, tdttY, sysize
end

function saveWsl(file)
	data  = jldopen(file, "r")
	nmeas = parse(Int64, keys(data["Wsl"])[end])
	sfq   = parse(Int64, keys(data["Wsl"])[1])
	nfw   = size(data["tfw"])[1]
	lsize = [data["attrs"]["L_$i"] for i in 1:4]
	tfw   = data["tfw"][:]

	Wsl = Array{Float64}(undef, nmeas, nfw)
	for (i,key) in enumerate(keys(data["Wsl"]))
		Wsl[(i-1)*sfq+1:i*sfq, :] = mean(data["Wsl"][key][:,:,:,:], dims=3)[:,:,1,1]
	end
	close(data)

	sysize = prod(string.(lsize) .* ["x", "x", "x", ""])
    ttW    = tfw'.^2 .* Wsl
    tdttW  = tfw[2:end-1]' .* derivate(ttW, tfw)
    ttW    = ttW[:,2:end-1]
	tfw    = tfw[2:end-1]

	return tfw, ttW, tdttW, sysize
end

function main()
	prefix = ARGS[1]
	files  = ARGS[2:end]

	tfw    = []
	ttE    = []
	tdttE  = []
	sysize = ""
	for file in files
		tfw, temp_ttE, temp_tdttE, sysize = saveYsl(file)
		if file == files[1]
			ttE   = copy(temp_ttE)
			tdttE = copy(temp_tdttE)
		else
			ttE   = cat(ttE, temp_ttE, dims=1)
			tdttE = cat(tdttE, temp_tdttE, dims=1)
		end
	end
	jldsave("JLD2_DIR/"*prefix*"Y"*sysize*".jld2"; tfw, ttE, tdttE)
	tfw    = []
	ttE    = []
	tdttE  = []
	sysize = ""
	for file in files
		tfw, temp_ttE, temp_tdttE, sysize = saveWsl(file)
		if file == files[1]
			ttE   = copy(temp_ttE)
			tdttE = copy(temp_tdttE)
		else
			ttE   = cat(ttE, temp_ttE, dims=1)
			tdttE = cat(tdttE, temp_tdttE, dims=1)
		end
	end
	jldsave("JLD2_DIR/"*prefix*"W"*sysize*".jld2"; tfw, ttE, tdttE)
end

main()