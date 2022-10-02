using InteractiveUtils
using Pkg
using Printf
using Random

using ArgParse
using CUDA
using JLD2
using TOML

using LatticeGPU
CUDA.allowscalar(false)

aps = ArgParseSettings()
@add_arg_table! aps begin
    "--input", "-i"
        help = "specify input file"
		required = true
	"--config", "-c"
        help = "configuration file for continuation"
end

function checks(params)
	if (params["Simulation"]["group"] != "SU3") && (params["Simulation"]["group"] != "SU2")
		error("[ERROR]: group must be either \"SU2\" or \"SU3\".")
	end
	if (params["Simulation"]["action"] != "WILSON") && (params["Simulation"]["action"] != "IMPROVED")
		error("[ERROR]: action must be either \"WILSON\" or \"IMPROVED\".")
	end
	if (params["Simulation"]["flow"] != "WILSON") && (params["Simulation"]["flow"] != "ZEUTHEN") && (params["Simulation"]["flow"] != "BOTH")
		error("[ERROR]: flow must be either \"WILSON\", \"ZEUTHEN\" or \"BOTH\".")
	end
	if (params["Boundary"]["type"] != "PERIODIC") && (params["Boundary"]["type"] != "SF")
		error("[ERROR]: type must be either \"PERIODIC\" or \"SF\".")
	end
	if (params["Boundary"]["type"] == "SF") && ((params["Simulation"]["flow"] == "ZEUTHEN") || (params["Simulation"]["flow"] == "BOTH"))
		error("[ERROR]: Zeuthen integration not supported on SF.")
	end
	if (params["Flow"]["integrator"] != "RK3") && (params["Flow"]["integrator"] != "RK2") && (params["Flow"]["integrator"] != "EULER")
		error("[ERROR]: Flow integrator must be \"RK3\", \"RK2\" or \"EULER\".")
	end
	if (params["HMC"]["integrator"] != "OMF4") && (params["HMC"]["integrator"] != "OMF2") && (params["HMC"]["integrator"] != "LEAPFROG")
		error("[ERROR]: HMC integrator must be \"OMF4\", \"OMF2\" or \"LEAPFROG\".")
	end
	if (params["HMC"]["nth"] % params["HMC"]["dtr_cfg"] != 0) || (params["HMC"]["ntr"] % params["HMC"]["dtr_cfg"] != 0)
		error("[ERROR]: nth and ntr must be multiples of dtr_cfg.")
	end
	if params["Run"]["loadcfg"] &&  (params["HMC"]["nth"] != 0)
		error("[ERROR]: nth must be zero in a continuation run.")
	end
	if (params["HMC"]["dtr_cfg"] % params["HMC"]["dtr_ms"] != 0) || (params["HMC"]["dtr_ms"] % params["HMC"]["dtr_log"] != 0)
		error("[ERROR]: dtr_cfg must be multiple of dtr_ms, a multiple of dtr_log.")
	end
	if params["Flow"]["ntot"] % params["Flow"]["dnms"] != 0
		error("[ERROR]: ntot must be multiple of dnms.")
	end
	if isfile(params["Run"]["dat"]*"/"*params["Run"]["name"]*".nc") || isfile(params["Run"]["log"]*"/"*params["Run"]["name"]*".log") || isfile(params["Run"]["cfg"]*"/"*params["Run"]["name"]*".tar.gz")
		error("[ERROR]: Files already exists.")
	end
	if params["Run"]["loadcfg"] && !(isfile(ARGS[2]))
		error("[ERROR]: Config file does not exist.")
	end
end

function preparation(params)
	DIM = params["Simulation"]["dimensions"]
	GRP = params["Simulation"]["group"] == "SU3" ? SU3 : SU2

	# Flow
	c0 = params["Simulation"]["action"] == "WILSON" ? 1.0 : 5.0/3.0
	if params["Simulation"]["flow"] == "WILSON"
		fwint = params["Flow"]["integrator"] == "RK3" ? wfl_rk3(Float64, params["Flow"]["eps"], 1.0E-6) : (params["Flow"]["integrator"] == "RK2" ? wfl_rk2(Float64, params["Flow"]["eps"], 1.0E-6) : wfl_euler(Float64, params["Flow"]["eps"], 1.0E-6))
	elseif params["Simulation"]["flow"] == "ZEUTHEN"
		fwint = params["Flow"]["integrator"] == "RK3" ? zfl_rk3(Float64, params["Flow"]["eps"], 1.0E-6) : (params["Flow"]["integrator"] == "RK2" ? zfl_rk2(Float64, params["Flow"]["eps"], 1.0E-6) : zfl_euler(Float64, params["Flow"]["eps"], 1.0E-6))
	else
		fwint = (params["Flow"]["integrator"] == "RK3" ? wfl_rk3(Float64, params["Flow"]["eps"], 1.0E-6) : (params["Flow"]["integrator"] == "RK2" ? wfl_rk2(Float64, params["Flow"]["eps"], 1.0E-6) : wfl_euler(Float64, params["Flow"]["eps"], 1.0E-6)), params["Flow"]["integrator"] == "RK3" ? zfl_rk3(Float64, params["Flow"]["eps"], 1.0E-6) : (params["Flow"]["integrator"] == "RK2" ? zfl_rk2(Float64, params["Flow"]["eps"], 1.0E-6) : zfl_euler(Float64, params["Flow"]["eps"], 1.0E-6)))
	end

	# Boundary conditions
	BCS  = params["Boundary"]["type"] == "PERIODIC" ? BC_PERIODIC : BC_SF_AFWB
	lp   = params["Boundary"]["type"] == "PERIODIC" ? SpaceParm{DIM}(tuple(params["Simulation"]["size"]...), tuple(params["Simulation"]["blocks"]...), BCS, tuple(params["Boundary"]["ntwist"]...)) : SpaceParm{DIM}(tuple(params["Simulation"]["size"]...), tuple(params["Simulation"]["blocks"]...), BCS, tuple(params["Boundary"]["ntwist"]...))
	gp   = params["Boundary"]["type"] == "PERIODIC" ? GaugeParm{Float64}(GRP{Float64}, params["Simulation"]["beta"], c0) : GaugeParm{Float64}(GRP{Float64}, params["Simulation"]["beta"], c0, tuple(params["Boundary"]["cG"], params["Boundary"]["cG"]), tuple(params["Boundary"]["phiT"]...), lp.iL[1:(end-1)])

	# HMC
	mcint = params["HMC"]["integrator"] == "OMF4" ? omf4(Float64, params["HMC"]["eps"], params["HMC"]["nstep"]) : (params["HMC"]["integrator"] == "OMF2" ? omf2(Float64, params["HMC"]["eps"], params["HMC"]["nstep"]) : leapfrog(Float64, params["HMC"]["eps"], params["HMC"]["nstep"]))

	# Seed
	Random.seed!(CURAND.default_rng(), params["Run"]["seed"])
	Random.seed!(params["Run"]["seed"])

	return GRP, lp, gp, mcint, fwint
end

function create_log(params, GRP, lp, gp, mcint, fwint)
	flog = open(params["Run"]["log"]*"/"*params["Run"]["name"]*".log", "w+");
	
	println(flog, "# HMC SU("*((GRP == SU3) ? "3" : "2")*") LatticeGPU simulation")
	println(flog, "\n## Environmental Info")
	println(flog, "User: ", params["Run"]["user"])
	println(flog, "Host: ", params["Run"]["host"])
	println(flog, "Seed: ", params["Run"]["seed"])
	versioninfo(flog)
	Pkg.status(io=flog)
	println(flog, "\n## Simulation Info")
	println(flog, lp)
	println(flog, gp)
	println(flog, mcint)
	println(flog, fwint)

	# write(flog, TranscodingStreams.TOKEN_END)
	flush(flog)

	return flog
end

function create_datafile(params)
	traj = jldopen(params["Run"]["dat"]*"/"*params["Run"]["name"]*".jld2", "w")
	if params["Simulation"]["flow"] != "BOTH"
		meas = jldopen(params["Run"]["dat"]*"/"*params["Run"]["name"]*"_flow.jld2", "w")
	else
		meas_w = jldopen(params["Run"]["dat"]*"/"*params["Run"]["name"]*"wilson_flow.jld2", "w")
		meas_z = jldopen(params["Run"]["dat"]*"/"*params["Run"]["name"]*"zeuthen_flow.jld2", "w")
	end

	attrs = Dict()
	for (i,l) in enumerate(params["Simulation"]["size"])
		attrs["L_$i"] = l
	end
	attrs["seed"] = params["Run"]["seed"]
	attrs["BCS"]  = params["Boundary"]["type"]
	attrs["beta"] = params["Simulation"]["beta"]
	traj["attrs"] = attrs
	if params["Simulation"]["flow"] != "BOTH"
		meas["attrs"] = attrs
	else
		meas_w["attrs"] = attrs
		meas_z["attrs"] = attrs
	end

	dimst = Dict()
	dimsf = Dict()
	dimst["nth"] = params["HMC"]["nth"]
	dimst["ntr"] = params["HMC"]["ntr"]
	dimsf["nms"] = div(params["HMC"]["ntr"]-params["HMC"]["nth"], params["HMC"]["dtr_ms"])
	dimsf["nfw"] = div(params["Flow"]["ntot"], params["Flow"]["dnms"]) + 1
	dimsf["nts"] = params["Simulation"]["size"][end]
	dimsf["asy"] = 1 + (params["Boundary"]["type"] == "SF")
	traj["dims"] = dimst
	if params["Simulation"]["flow"] != "BOTH"
		meas["dims"] = dimsf
	else
		meas_w["dims"] = dimsf
		meas_z["dims"] = dimsf
	end

	trajm = div(dimst["ntr"], params["HMC"]["dtr_cfg"])
	measm = div(dimsf["nms"], params["HMC"]["dtr_cfg"]) + !params["Run"]["loadcfg"]

	traj["tmc"] = collect(1:dimst["ntr"])
	JLD2.Group(traj, "avpl", est_num_entries=trajm, est_link_name_len=JLD2.link_size(string(dimst["ntr"])))
	JLD2.Group(traj, "dH", est_num_entries=trajm, est_link_name_len=JLD2.link_size(string(dimst["ntr"])))
	JLD2.Group(traj, "iac", est_num_entries=trajm, est_link_name_len=JLD2.link_size(string(dimst["ntr"])))
	if params["Boundary"]["type"] == "SF"
		JLD2.Group(traj, "dsdeta", est_num_entries=trajm, est_link_name_len=JLD2.link_size(string(dimst["ntr"])))
		JLD2.Group(traj, "ddnu", est_num_entries=trajm, est_link_name_len=JLD2.link_size(string(dimst["ntr"])))
	end
	
	if params["Simulation"]["flow"] != "BOTH"
		meas["tmc"] = collect(1:params["HMC"]["dtr_ms"]:(params["HMC"]["ntr"]-params["HMC"]["nth"]))
		meas["tfw"] = params["Flow"]["eps"] .* collect(0:params["Flow"]["dnms"]:params["Flow"]["ntot"])
		JLD2.Group(meas, "Wsl", est_num_entries=measm, est_link_name_len=JLD2.link_size(string(dimsf["nms"])))
		JLD2.Group(meas, "Ysl", est_num_entries=measm, est_link_name_len=JLD2.link_size(string(dimsf["nms"])))
		JLD2.Group(meas, "Qsl", est_num_entries=measm, est_link_name_len=JLD2.link_size(string(dimsf["nms"])))
	else
		meas_w["tmc"] = collect(1:params["HMC"]["dtr_ms"]:(params["HMC"]["ntr"]-params["HMC"]["nth"]))
		meas_w["tfw"] = params["Flow"]["eps"] .* collect(0:params["Flow"]["dnms"]:params["Flow"]["ntot"])
		JLD2.Group(meas_w, "Wsl", est_num_entries=measm, est_link_name_len=JLD2.link_size(string(dimsf["nms"])))
		JLD2.Group(meas_w, "Ysl", est_num_entries=measm, est_link_name_len=JLD2.link_size(string(dimsf["nms"])))
		JLD2.Group(meas_w, "Qsl", est_num_entries=measm, est_link_name_len=JLD2.link_size(string(dimsf["nms"])))
		meas_z["tmc"] = collect(1:params["HMC"]["dtr_ms"]:(params["HMC"]["ntr"]-params["HMC"]["nth"]))
		meas_z["tfw"] = params["Flow"]["eps"] .* collect(0:params["Flow"]["dnms"]:params["Flow"]["ntot"])
		JLD2.Group(meas_z, "Wsl", est_num_entries=measm, est_link_name_len=JLD2.link_size(string(dimsf["nms"])))
		JLD2.Group(meas_z, "Ysl", est_num_entries=measm, est_link_name_len=JLD2.link_size(string(dimsf["nms"])))
		JLD2.Group(meas_z, "Qsl", est_num_entries=measm, est_link_name_len=JLD2.link_size(string(dimsf["nms"])))
	end
	
	close(traj)
	if params["Simulation"]["flow"] != "BOTH"
		close(meas)
	else
		close(meas_w)
		close(meas_z)
	end
end

function log_verbose(flog, i, dH, iac, avpl, avac_tot, avt, avt_tot)
	@printf(flog, "\nTrajectory no %d\n", i)
	@printf(flog, "dH = %+7.1e, iac = %d\n", dH[i], iac[i])
	@printf(flog, "Average plaquette = %8.6f\n", avpl[i])
	@printf(flog, "Acceptance rate = %8.6f\n", avac_tot/i)
	@printf(flog, "Time per trajectory = %8.2e sec (average = %8.2e sec)\n", avt, avt_tot/i)
	flush(flog)
	return
end

function traj_meas(params, i, avpl, dH, iac, dsdeta=nothing, ddnu=nothing)
	si   = lpad(i,ndigits(params["HMC"]["ntr"]),"0")
	oi   = i - params["HMC"]["dtr_cfg"] + 1
	traj = jldopen(params["Run"]["dat"]*"/"*params["Run"]["name"]*".jld2", "r+")
	traj["avpl/"*si] = avpl[oi:i]
	traj["dH/"*si]   = dH[oi:i]
	traj["iac/"*si]  = Int8.(iac[oi:i])
	if (dsdeta !== nothing) && (ddnu !== nothing)
		traj["dsdeta/"*si] = dsdeta[oi:i]
		traj["ddnu/"*si]   = ddnu[oi:i]
	end
	close(traj)
end

function flow_compute(flog, i, params, Wmat, Ymat, Qmat, tmat, U, ymws, gp, lp, fwint, type=nothing)
	ci = (i == params["HMC"]["nth"]) ? 1 : (div(i - params["HMC"]["nth"], params["HMC"]["dtr_ms"]) - 1) % div(params["HMC"]["dtr_cfg"], params["HMC"]["dtr_ms"]) + 1
	ymws.U1 .= U

	Eoft_plaq(tmat, U, gp, lp, ymws)
	Wmat[ci,1,:,:] = sum(tmat, dims=2)

	Eoft_clover(tmat, U, gp, lp, ymws)
	Ymat[ci,1,:,:] = sum(tmat, dims=2)

	Qtop(tmat, U, gp, lp, ymws)
	Qmat[ci,1,:,:] = sum(tmat, dims=2)

	if type === nothing
		@printf(flog, "\nMeasurement run:\n")
	else
		@printf(flog, "\nMeasurement run (%s):\n", type)
	end
	@printf(flog, "\nn =    0, t = %8.2e, Wact = %12.6e, Yact = %12.6e, Q = %9.2e\n", 0.0, sum(Wmat[ci,1,:,:])/lp.iL[end], sum(Ymat[ci,1,:,:])/lp.iL[end], sum(Qmat[ci,1,:,:]))
	
	for j in 2:(div(params["Flow"]["ntot"], params["Flow"]["dnms"])+1)
		flw(U, fwint, params["Flow"]["dnms"], gp, lp, ymws)

		Eoft_plaq(tmat, U, gp, lp, ymws)
		Wmat[ci,j,:,:] = sum(tmat, dims=2)

		Eoft_clover(tmat, U, gp, lp, ymws)
		Ymat[ci,j,:,:] = sum(tmat, dims=2)

		Qtop(tmat, U, gp, lp, ymws)
		Qmat[ci,j,:,:] = sum(tmat, dims=2)

		@printf(flog, "n = %4d, t = %8.2e, Wact = %12.6e, Yact = %12.6e, Q = %9.2e\n", (j-1)*params["Flow"]["dnms"], (j-1)*params["Flow"]["eps"]*params["Flow"]["dnms"], sum(Wmat[ci,j,:,:])/lp.iL[end], sum(Ymat[ci,j,:,:])/lp.iL[end], sum(Qmat[ci,j,:,:]))
	end

	U .= ymws.U1
end

function flow_compute_SF(flog, i, params, Wmat, Ymat, Qmat, tmat, U, ymws, gp, lp, fwint)
	ci = (i == params["HMC"]["nth"]) ? 1 : (div(i - params["HMC"]["nth"], params["HMC"]["dtr_ms"]) - 1) % div(params["HMC"]["dtr_cfg"], params["HMC"]["dtr_ms"]) + 1
	nplsm = div(lp.npls, 2)
	ymws.U1 .= U

	Eoft_plaq(tmat, U, gp, lp, ymws)
	Wmat[ci,1,:,1] = sum(tmat[:,1:nplsm], dims=2)
	Wmat[ci,1,:,2] = sum(tmat[:,(nplsm+1):lp.npls], dims=2)

	Eoft_clover(tmat, U, gp, lp, ymws)
	Ymat[ci,1,:,1] = sum(tmat[:,1:nplsm], dims=2)
	Ymat[ci,1,:,2] = sum(tmat[:,(nplsm+1):lp.npls], dims=2)

	Qtop(tmat, U, gp, lp, ymws)
	Qmat[ci,1,:,:] = sum(tmat, dims=2)

	@printf(flog, "\nMeasurement run:\n")
	@printf(flog, "\nn =    0, t = %8.2e, Wact = %12.6e, Yact = %12.6e, Q = %9.2e\n", 0.0, sum(Wmat[ci,1,:,:])/lp.iL[end], sum(Ymat[ci,1,:,:])/lp.iL[end], sum(Qmat[ci,1,:,:]))
	
	for j in 2:(div(params["Flow"]["ntot"], params["Flow"]["dnms"])+1)
		flw(U, fwint, params["Flow"]["dnms"], gp, lp, ymws)

		Eoft_plaq(tmat, U, gp, lp, ymws)
		Wmat[ci,j,:,1] = sum(tmat[:,1:nplsm], dims=2)
		Wmat[ci,j,:,2] = sum(tmat[:,(nplsm+1):lp.npls], dims=2)

		Eoft_clover(tmat, U, gp, lp, ymws)
		Ymat[ci,j,:,1] = sum(tmat[:,1:nplsm], dims=2)
		Ymat[ci,j,:,2] = sum(tmat[:,(nplsm+1):lp.npls], dims=2)

		Qtop(tmat, U, gp, lp, ymws)
		Qmat[ci,j,:,:] = sum(tmat, dims=2)

		@printf(flog, "n = %4d, t = %8.2e, Wact = %12.6e, Yact = %12.6e, Q = %9.2e\n", (j-1)*params["Flow"]["dnms"], (j-1)*params["Flow"]["eps"]*params["Flow"]["dnms"], sum(Wmat[ci,j,:,:])/lp.iL[end], sum(Ymat[ci,j,:,:])/lp.iL[end], sum(Qmat[ci,j,:,:]))
	end

	U .= ymws.U1
end

function flow_meas(params, i, Wmat, Ymat, Qmat, type=nothing)
	if type === nothing
		meas = jldopen(params["Run"]["dat"]*"/"*params["Run"]["name"]*"_flow.jld2", "r+")
	else
		meas = jldopen(params["Run"]["dat"]*"/"*params["Run"]["name"]*type*"_flow.jld2", "r+")
	end
	si   = lpad(div(i-params["HMC"]["nth"], params["HMC"]["dtr_ms"]),ndigits(meas["dims"]["nms"]),"0")
	meas["Wsl/"*si] = Wmat
	meas["Ysl/"*si] = Ymat
	meas["Qsl/"*si] = Qmat
	close(meas)
end

function main(args)
	params = TOML.parsefile(args["input"])
	checks(params)
	GRP, lp, gp, mcint, fwint = preparation(params)
	flog = create_log(params, GRP, lp, gp, mcint, fwint)
	create_datafile(params)
	ymws = YMworkspace(GRP, Float64, lp)
	
	if params["Run"]["loadcfg"]
		println(flog, "\nLoading configuration from "*args["config"])
		cfile = load_object(args["config"])
		U     = CuArray(cfile)
	else
		println(flog, "\nCreating cold start configuration")
		U = vector_field(GRP{Float64}, lp)
		fill!(U, one(GRP{Float64}))
		params["Boundary"]["type"] == "SF" ? setbndfield(U, params["Boundary"]["phi0"], lp) : nothing
	end
	
	# Allocate arrays
	avpl = NaN .* ones(params["HMC"]["ntr"])
	dH   = NaN .* ones(params["HMC"]["ntr"])
	iac  =  -1 .* ones(Int8, params["HMC"]["ntr"])
	tmat = zeros(Float64, lp.iL[end], lp.npls)
	Wmat = zeros(Float64, div(params["HMC"]["dtr_cfg"], params["HMC"]["dtr_ms"]), div(params["Flow"]["ntot"], params["Flow"]["dnms"]) + 1, lp.iL[end], 1 + (params["Boundary"]["type"] == "SF"))
	Ymat = zeros(Float64, div(params["HMC"]["dtr_cfg"], params["HMC"]["dtr_ms"]), div(params["Flow"]["ntot"], params["Flow"]["dnms"]) + 1, lp.iL[end], 1 + (params["Boundary"]["type"] == "SF"))
	Qmat = zeros(Float64, div(params["HMC"]["dtr_cfg"], params["HMC"]["dtr_ms"]), div(params["Flow"]["ntot"], params["Flow"]["dnms"]) + 1, lp.iL[end], 1)
	if params["Simulation"]["flow"] == "BOTH"
		Wmat2 = zeros(Float64, div(params["HMC"]["dtr_cfg"], params["HMC"]["dtr_ms"]), div(params["Flow"]["ntot"], params["Flow"]["dnms"]) + 1, lp.iL[end], 1 + (params["Boundary"]["type"] == "SF"))
		Ymat2 = zeros(Float64, div(params["HMC"]["dtr_cfg"], params["HMC"]["dtr_ms"]), div(params["Flow"]["ntot"], params["Flow"]["dnms"]) + 1, lp.iL[end], 1 + (params["Boundary"]["type"] == "SF"))
		Qmat2 = zeros(Float64, div(params["HMC"]["dtr_cfg"], params["HMC"]["dtr_ms"]), div(params["Flow"]["ntot"], params["Flow"]["dnms"]) + 1, lp.iL[end], 1)
	end
	if params["Boundary"]["type"] == "SF"
		dsdeta = NaN .* ones(params["HMC"]["ntr"])
		ddnu   = NaN .* ones(params["HMC"]["ntr"])
	end

	# Thermalization
	println(flog, "\n## Thermalization")
	flush(flog)

	avt, avt_tot, avac, avac_tot = 0.0, 0.0, 0.0, 0.0
	for i in 1:params["HMC"]["nth"]
		t_start = time_ns()

		dH[i], iac[i] = HMC!(U, mcint, lp, gp, ymws)
		avpl[i]       = plaquette(U, lp, gp, ymws)
		if params["Boundary"]["type"] == "SF"
			dsdeta[i], ddnu[i] = sfcoupling(U, lp, gp, ymws)
		end

		avac += iac[i]
		avt  += time_ns() - t_start

		if i % params["HMC"]["dtr_log"] == 0
			avt       = 1e-9*avt/params["HMC"]["dtr_log"]
			avac_tot += avac
			avt_tot  += avt*params["HMC"]["dtr_log"]
			log_verbose(flog, i, dH, iac, avpl, avac_tot, avt, avt_tot)
			avt, avac = 0.0, 0.0

			if i % params["HMC"]["dtr_cfg"] == 0
				params["Boundary"]["type"] == "SF" ? traj_meas(params, i, avpl, dH, iac, dsdeta, ddnu) : traj_meas(params, i, avpl, dH, iac)
				
				ct_start = time_ns()
				save_object(params["Run"]["cfg"]*"/"*params["Run"]["name"]*"n"*string(i), Array(U))
				@printf(flog, "\nConfiguration no %d exported in %8.2e sec\n", div(i, params["HMC"]["dtr_cfg"]), 1e-9*(time_ns() - ct_start))
			end
		end	
	end

	# Measurements
	println(flog, "\n## Measurements")
	flush(flog)

	mavt, mavt_tot = 0.0, 0.0
	for i in (params["HMC"]["nth"]+1):params["HMC"]["ntr"]
		t_start = time_ns()

		dH[i], iac[i] = HMC!(U, mcint, lp, gp, ymws)
		avpl[i]       = plaquette(U, lp, gp, ymws)
		if params["Boundary"]["type"] == "SF"
			dsdeta[i], ddnu[i] = sfcoupling(U, lp, gp, ymws)
		end

		avac += iac[i]
		avt  += time_ns() - t_start

		if i % params["HMC"]["dtr_log"] == 0
			avt       = 1e-9*avt/params["HMC"]["dtr_log"]
			avac_tot += avac
			avt_tot  += avt*params["HMC"]["dtr_log"]
			log_verbose(flog, i, dH, iac, avpl, avac_tot, avt, avt_tot)
			avt, avac = 0.0, 0.0

			if i % params["HMC"]["dtr_ms"] == 0
				mt_start = time_ns()

				if params["Simulation"]["flow"] != "BOTH"
					params["Boundary"]["type"] == "SF" ? flow_compute_SF(flog, i, params, Wmat, Ymat, Qmat, tmat, U, ymws, gp, lp, fwint) : flow_compute(flog, i, params, Wmat, Ymat, Qmat, tmat, U, ymws, gp, lp, fwint)
				else
					flow_compute(flog, i, params, Wmat, Ymat, Qmat, tmat, U, ymws, gp, lp, fwint[1], "wilson")
					flow_compute(flog, i, params, Wmat2, Ymat2, Qmat2, tmat, U, ymws, gp, lp, fwint[2], "zeuthen")
				end
				
				mavt = 1e-9*(time_ns() - mt_start)
				mavt_tot += mavt
				@printf(flog, "\nConfiguration fully processed in %8.2e sec (average = %8.2e sec)\n", mavt, mavt_tot/div(i-params["HMC"]["nth"], params["HMC"]["dtr_ms"]))
				flush(flog)

				if i % params["HMC"]["dtr_cfg"] == 0
					params["Boundary"]["type"] == "SF" ? traj_meas(params, i, avpl, dH, iac, dsdeta, ddnu) : traj_meas(params, i, avpl, dH, iac)
					if params["Simulation"]["flow"] != "BOTH"
						flow_meas(params, i, Wmat, Ymat, Qmat)
					else
						flow_meas(params, i, Wmat, Ymat, Qmat, "wilson")
						flow_meas(params, i, Wmat2, Ymat2, Qmat2, "zeuthen")
					end
					@printf(flog, "Measured data saved\n")

					ct_start = time_ns()
					save_object(params["Run"]["cfg"]*"/"*params["Run"]["name"]*"n"*string(i), Array(U))
					@printf(flog, "\nConfiguration no %d exported in %8.2e sec\n", div(i, params["HMC"]["dtr_cfg"]), 1e-9*(time_ns() - ct_start))
				
					flush(flog)
				end
			end
		end	
	end

	print(flog, "\nProcess completed.")
	close(flog)

	com1 = "tar -cf "*params["Run"]["cfg"]*"/"*params["Run"]["name"]*".tar "*params["Run"]["cfg"]*"/"*params["Run"]["name"]*"n*";
	com2 = "gzip -9 "*params["Run"]["cfg"]*"/"*params["Run"]["name"]*".tar";
	com3 = "rm "*params["Run"]["cfg"]*"/"*params["Run"]["name"]*"n*";
	run(`bash -c $com1`)
	run(`bash -c $com2`)
	run(`bash -c $com3`)
end

parsed_args = parse_args(ARGS, aps)
main(parsed_args)