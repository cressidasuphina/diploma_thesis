-- LogSystem: streams per-tick training metrics to a local logging server
-- (see log_server.py) as CSV rows, plus a final run summary.

local HttpService = game:GetService("HttpService")
local LOG_SERVER = "http://localhost:8081"

local HEADER = {
	"timestamp", "learning_steps", "macro_classifiers",
	"micro_classifiers", "avg_fitness", "max_fitness",
	"avg_experience", "avg_prediction", "avg_error",
	"elite_count", "dead_count", "goals_total",
	"goals_last_1000", "avg_steps_between_goals",
	"best_path_length", "avg_path_length",
	"current_epsilon", "pheromone_active_cells",
	"pheromone_max_value", "action_dist_forward",
	"action_dist_backward", "action_dist_left",
	"action_dist_right",
}

local LogSystem = {}
LogSystem.__index = LogSystem

function LogSystem.new()
	local self = setmetatable({}, LogSystem)

	local ok, err = pcall(function()
		HttpService:PostAsync(LOG_SERVER, HttpService:JSONEncode({ header = HEADER }))
	end)

	if ok then
		print("Log server connected, writing to file on server")
	else
		warn("Log server not reachable: " .. tostring(err))
		warn("Make sure python log_server.py is running")
	end

	return self
end

function LogSystem:record(xcs, agents)
	local totalN, totalF, totalExp = 0, 0, 0
	local totalPred, totalErr, maxF = 0, 0, 0
	local eliteCount, deadCount = 0, 0
	local actionCounts = {
		["forward|0"] = 0, ["backward|0"] = 0,
		["left|0"] = 0,    ["right|0"] = 0,
	}

	for _, cl in ipairs(xcs.classifiers) do
		totalN += cl.n
		totalF += cl.fitness
		totalExp += cl.exp
		totalPred += cl.prediction
		totalErr += (cl.error or 0)
		if cl.fitness > maxF then maxF = cl.fitness end
		if cl.fitness >= 0.9 then eliteCount += 1 end
		if cl.fitness < 0.1 then deadCount += 1 end
		if actionCounts[cl.actionSig] then
			actionCounts[cl.actionSig] += 1
		end
	end

	local numCl = math.max(#xcs.classifiers, 1)
	local avgF = totalF / numCl
	local avgExp = totalExp / numCl
	local avgP = totalPred / numCl
	local avgErr = totalErr / numCl

	local recentGoals = 0
	for _, ts in ipairs(xcs.goalTimestamps or {}) do
		if ts > xcs.timeStep - 1000 then recentGoals += 1 end
	end

	local avgPath = (xcs.goalCount or 0) > 0
		and (xcs.totalPathLength or 0) / xcs.goalCount
		or 0

	local activeCells, maxPhero = 0, 0
	for _, dirs in pairs(xcs.pheromone) do
		local cellMax = 0
		for _, v in pairs(dirs) do
			if v > cellMax then cellMax = v end
		end
		if cellMax > 2.0 then activeCells += 1 end
		if cellMax > maxPhero then maxPhero = cellMax end
	end

	local row = {
		math.floor(tick()),
		xcs.timeStep,
		#xcs.classifiers,
		totalN,
		string.format("%.4f", avgF),
		string.format("%.4f", maxF),
		string.format("%.1f", avgExp),
		string.format("%.3f", avgP),
		string.format("%.4f", avgErr),
		eliteCount,
		deadCount,
		xcs.goalCount or 0,
		recentGoals,
		string.format("%.1f", xcs.timeStep / math.max(xcs.goalCount or 1, 1)),
		xcs.bestPathLength or 0,
		string.format("%.1f", avgPath),
		string.format("%.3f", xcs.currentExplore or 0),
		activeCells,
		string.format("%.2f", maxPhero),
		actionCounts["forward|0"],
		actionCounts["backward|0"],
		actionCounts["left|0"],
		actionCounts["right|0"],
	}

	local ok = pcall(function()
		HttpService:PostAsync(LOG_SERVER, HttpService:JSONEncode({ row = row }))
	end)

	if not ok then
		warn("HTTP log failed, printing instead")
		print("LOG_CSV_ROW:" .. table.concat(row, ","))
	end
end

function LogSystem:summary(xcs)
	local avgPath = (xcs.goalCount or 0) > 0
		and (xcs.totalPathLength or 0) / xcs.goalCount
		or 0

	local summaryData = {
		total_steps = xcs.timeStep,
		total_goals = xcs.goalCount or 0,
		best_path_length = xcs.bestPathLength or 0,
		avg_path_length = string.format("%.1f", avgPath),
		avg_steps_per_goal = string.format("%.1f", xcs.timeStep / math.max(xcs.goalCount or 1, 1)),
		final_epsilon = string.format("%.3f", xcs.currentExplore or 0),
		final_macro_classifiers = #xcs.classifiers,
	}

	pcall(function()
		HttpService:PostAsync(LOG_SERVER, HttpService:JSONEncode({ summary = summaryData }))
	end)

	print("Experiment complete. Summary sent to log server.")
end

return LogSystem
