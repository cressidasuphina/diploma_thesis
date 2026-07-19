-- Main: spawns the training dummies, runs the per-agent step loop with a
-- barrier-synchronized central XCS update, and periodically reports progress.

local ServerStorage = game:GetService("ServerStorage")

local XCS = require(script.Parent.XCS)
local LogSystem = require(script.Parent.LogSystem)
local Agent = require(script.Parent.Agent)

local NUM_DUMMIES = 10
local ACTIONS = XCS.ACTIONS
local XCS_CONFIG = XCS.CONFIG

local DUMMY = ServerStorage:WaitForChild("Dummy")
local START = workspace:WaitForChild("Start")
local GOAL = workspace:WaitForChild("Goal")

local rand = Random.new()

local function executeAction(humanoid, humanoidRoot, action)
	local lookDir = humanoidRoot.CFrame.LookVector
	local rightDir = humanoidRoot.CFrame.RightVector

	local moveVector
	if action.moveType == "forward" then
		moveVector = lookDir
	elseif action.moveType == "backward" then
		moveVector = -lookDir
	elseif action.moveType == "left" then
		moveVector = -rightDir
	elseif action.moveType == "right" then
		moveVector = rightDir
	else
		moveVector = Vector3.new(0, 0, 0)
	end

	humanoid:Move(moveVector, false)
end

local function encodeLastAction(moveType)
	if moveType == "forward" then return {0, 0}
	elseif moveType == "left" then return {0, 1}
	elseif moveType == "right" then return {1, 0}
	elseif moveType == "backward" then return {1, 1}
	end
	return {0, 0}
end

-- Number of open directions from vision data (excludes "behind", so we
-- don't count the direction the agent came from as a valid choice).
local function countOpenDirections(vd)
	local open = 0
	if vd[1] == 0 then open += 1 end -- left
	if vd[3] == 0 then open += 1 end -- forward
	if vd[5] == 0 then open += 1 end -- right
	return open
end

-- Nudges the agent toward the center of a corridor using wall distances
-- from the vision system, so it doesn't scrape along one side.
local function applyCenteringForce(humanoidRoot, humanoid, lengths)
	if not lengths then return end

	local leftDist = lengths[1] or 10
	local rightDist = lengths[5] or 10

	-- only center when both sides have walls (inside a corridor); if one
	-- side is far away we're near a junction, so don't center
	local MAX_WALL_DIST = 10 -- studs, tuned to the maze cell width
	if leftDist > MAX_WALL_DIST or rightDist > MAX_WALL_DIST then return end

	-- positive = too far right, needs to move left
	local lateralError = rightDist - leftDist
	if math.abs(lateralError) < 0.3 then return end -- avoid oscillation

	local Kp = 0.15
	local correction = humanoidRoot.CFrame.RightVector * (-lateralError * Kp)
	local forward = humanoidRoot.CFrame.LookVector

	local blended = (forward + correction).Unit
	humanoid:Move(blended, false)
end

local function centralUpdate(xcs, agents)
	if xcs._stopped then return end

	for _, agent in ipairs(agents) do
		local exp = agent.pendingUpdate
		if not exp then continue end

		-- XCS TD update from the previous decision
		if exp.prevActionSet and #exp.prevActionSet > 0 then
			xcs:updateQ(exp.prevActionSet, exp.prevReward, exp.maxPred)
			xcs:runGA(exp.prevActionSet, exp.prevState)
		end

		if exp.reachedGoal then
			xcs:updateQ(exp.actionSet, exp.totalReward, 0)
			xcs:runGA(exp.actionSet, exp.state)

			if not xcs.hasReachedGoal then
				xcs.hasReachedGoal = true
				xcs.currentExplore = XCS_CONFIG.p_explore_after_goal
			end

			xcs.goalCount = (xcs.goalCount or 0) + 1

			local targetExplore = math.max(XCS_CONFIG.p_explore_min, xcs.currentExplore * 0.98)
			xcs.currentExplore = targetExplore
			print(string.format("[Exploration] goal reached -> epsilon %.3f", xcs.currentExplore))

			xcs.totalPathLength = (xcs.totalPathLength or 0) + #exp.currentPath
			local avgPath = xcs.totalPathLength / xcs.goalCount
			print(string.format("Goal #%d: path=%d avg=%.1f best=%d",
				xcs.goalCount, #exp.currentPath, avgPath, xcs.bestPathLength or 0))

			table.insert(xcs.goalTimestamps, xcs.timeStep)
		end

		if exp.reachedGoal and #exp.currentPath > 1 then
			local pathLen = #exp.currentPath
			if not xcs.bestPathLength or pathLen < xcs.bestPathLength then
				xcs.bestPathLength = pathLen
				xcs:postSuccessReinforce(exp.currentPath, pathLen, true, 1.0)
				print("New best path! length:", pathLen)
			else
				local relativeQuality = xcs.bestPathLength / pathLen
				xcs:postSuccessReinforce(exp.currentPath, pathLen, false, relativeQuality)
				print("Suboptimal path quality:", relativeQuality, "best:", xcs.bestPathLength)
			end
		end

		agent.pendingUpdate = nil
	end

	if xcs.timeStep % 50 == 0 and xcs.hasReachedGoal then
		print("DECAY")
		xcs:decayPheromones(0.98)
	end

	xcs.timeStep += 1
	xcs:performMaintenance()
end

local function simulateEvolution()
	print("Starting XCS...")

	local xcs = XCS.new()
	local agents = {}

	for i = 1, NUM_DUMMIES do
		local dummy = DUMMY:Clone()
		dummy.Name = "Dummy_" .. i
		dummy.Parent = workspace
		dummy:PivotTo(START.CFrame * CFrame.new(i * 3, 0, 0))
		task.wait(3)

		local agent = Agent.new(dummy, xcs)
		table.insert(agents, agent)
	end

	for _, agent in ipairs(agents) do
		agent:updateOtherNpcs(agents)
	end

	local numAgents = #agents
	local agentsDone = 0
	local barrierGeneration = 0

	local function runAgent(agent)
		local dummy = agent.npc
		local humanoid = dummy:FindFirstChild("Humanoid")
		if not humanoid then
			warn("Missing Humanoid for " .. dummy.Name)
			return
		end

		task.spawn(function()
			local ok, err = pcall(function()
				while dummy.Parent do
					-- ── STEP: perceive & decide ──────────────────────
					local state, visionData = agent:buildState(agent)
					local pos = dummy:GetPivot().Position
					local currentCell = agent:getGridKey(pos)

					local matchSet = xcs:generateMatchSet(state)
					local PA = xcs:predictionArrayFromMatchSet(matchSet)
					local action, actionSet = xcs:selectActionFromMatch(matchSet, PA, currentCell)
					agent.lastActionBits = encodeLastAction(action.moveType)

					local totalReward = 0
					local reachedGoal = false
					local stepsExecuted = 0
					local maxSteps = 1000
					local humanoidRoot = dummy:FindFirstChild("HumanoidRootPart")

					if #agent.episodePath == 0 then
						table.insert(agent.episodePath, currentCell)
					end

					agent.vision:updateVision(agent.otherNpcs)
					local vd, lengths = agent.vision:getVisionData()
					agent.vision:visualizeRays()

					applyCenteringForce(humanoidRoot, humanoid, lengths)
					executeAction(humanoid, humanoidRoot, action)

					-- ── EXECUTE: run until next decision point ───────
					while stepsExecuted < maxSteps do
						task.wait(0.05)

						local newPos = dummy:GetPivot().Position
						local cell = agent:getGridKey(newPos)

						agent.vision:updateVision(agent.otherNpcs)
						vd, lengths = agent.vision:getVisionData()
						agent.vision:visualizeRays()

						applyCenteringForce(humanoidRoot, humanoid, lengths)

						local prevCellKey = agent.lastCell
						local enteredNew = cell ~= prevCellKey
						if enteredNew then agent.lastCell = cell end

						-- stuck detection
						if cell == prevCellKey then
							agent.sameCellSteps += 1
						else
							agent.sameCellSteps = 0
						end

						if agent.sameCellSteps >= agent.maxSameCellSteps then
							totalReward -= 1.0
							agent.sameCellSteps = 0
							break
						end

						-- check the goal first so it's never missed
						if XCS.calculateGoalReward(newPos, GOAL) > 0 then
							totalReward += 100
							reachedGoal = true
							break
						end

						if enteredNew then
							table.insert(agent.episodePath, cell)

							local bonus = agent:isNewCell(newPos)
							if bonus > 0 then
								totalReward += bonus
							end
							agent.prevCell1 = cell
						end

						-- decision point: corridor (1 open dir) keeps walking,
						-- a junction or dead end stops for a new XCS decision
						if stepsExecuted >= 1 then
							local openDirs = countOpenDirections(vd)
							if openDirs >= 2 or openDirs == 0 then
								break
							end
						end

						stepsExecuted += 1
					end

					humanoid:Move(Vector3.new(0, 0, 0), false)

					-- ── BUFFER: hand results to the central update ───
					local maxPred = 0
					for _, v in pairs(PA) do
						if v > maxPred then maxPred = v end
					end
					if reachedGoal then maxPred = 0 end

					agent.pendingUpdate = {
						prevActionSet = agent.prevActionSet,
						prevReward = agent.prevReward,
						prevState = agent.prevState,
						maxPred = maxPred,
						actionSet = actionSet,
						state = state,
						reachedGoal = reachedGoal,
						currentPath = reachedGoal and table.clone(agent.episodePath) or {},
						totalReward = totalReward,
					}

					agent.prevActionSet = actionSet
					agent.prevReward = totalReward
					agent.prevState = table.clone(state)

					if reachedGoal then
						print("Goal reached! reward:", totalReward, "path:", #agent.episodePath)
						dummy:PivotTo(START.CFrame)
						agent:resetEpisode()
					end

					-- ── BARRIER: wait for all agents before the central
					-- update runs exactly once per generation ───────────
					agentsDone += 1
					local myGeneration = barrierGeneration

					if agentsDone >= numAgents then
						agentsDone = 0
						centralUpdate(xcs, agents)
						barrierGeneration += 1
					else
						local waitStart = tick()
						while barrierGeneration == myGeneration do
							task.wait()
							if tick() - waitStart > 10 then
								waitStart = tick() -- reset to avoid log spam
							end
						end
					end
				end
			end)

			if not ok then
				warn("Agent " .. agent.npc.Name .. " crashed: " .. tostring(err))
			end
		end)
	end

	for _, agent in ipairs(agents) do
		runAgent(agent)
	end

	local log = LogSystem.new()

	-- monitoring runs in its own coroutine, never inside the agent loop
	task.spawn(function()
		while true do
			task.wait(30)

			log:record(xcs, agents)

			-- fitness-based stop condition
			local totalF = 0
			for _, cl in ipairs(xcs.classifiers) do
				totalF += cl.fitness
			end
			local avgF = totalF / math.max(#xcs.classifiers, 1)

			if avgF >= 0.95 and xcs.goalCount > 10 then
				print(string.format("\nSTOPPING: avg fitness %.4f >= 0.95 at step %d", avgF, xcs.timeStep))
				log:summary(xcs)
				xcs:printRules(#xcs.classifiers)
				xcs:printPheromoneMap()
				xcs._stopped = true -- Roblox has no os.exit, so agents just stop updating
				break
			end

			print("\n" .. string.rep("=", 60))
			print("  XCS LEARNING PROGRESS REPORT")
			print(string.rep("=", 60))

			xcs:printPheromoneMap()
			xcs:printFitnessDebug()

			local totalNumerosity, avgFitness, avgExperience, avgPrediction, maxFitness = 0, 0, 0, 0, 0
			for _, cl in ipairs(xcs.classifiers) do
				totalNumerosity += cl.n
				avgFitness += cl.fitness
				avgExperience += cl.exp
				avgPrediction += cl.prediction
				if cl.fitness > maxFitness then maxFitness = cl.fitness end
			end

			local numClassifiers = #xcs.classifiers
			if numClassifiers > 0 then
				avgFitness /= numClassifiers
				avgExperience /= numClassifiers
				avgPrediction /= numClassifiers
			end

			print(string.format("  Learning Steps:      %d", xcs.timeStep))
			print(string.format("  Macro-classifiers:   %d", numClassifiers))
			print(string.format("  Micro-classifiers:   %d / %d", totalNumerosity, XCS_CONFIG.max_population))
			print(string.format("  Avg Fitness:         %.3f (max: %.3f)", avgFitness, maxFitness))
			print(string.format("  Avg Experience:      %.1f", avgExperience))
			print(string.format("  Avg Prediction:      %.3f", avgPrediction))

			print("\n  Top Rules by Fitness:")
			xcs:printRules(30)

			print("\n  Most Experienced Rules:")
			local sortedByExp = {}
			for _, cl in ipairs(xcs.classifiers) do table.insert(sortedByExp, cl) end
			table.sort(sortedByExp, function(a, b) return (a.exp or 0) > (b.exp or 0) end)
			for i = 1, math.min(10, #sortedByExp) do
				local cl = sortedByExp[i]
				local cond = ""
				for _, bit in ipairs(cl.condition) do cond = cond .. tostring(bit) end
				local actionStr = cl.action.moveType or "unknown"
				if cl.action.jump then actionStr = actionStr .. "+jump" end
				print(string.format("    [%d] If %s -> %s | exp=%d n=%d F=%.3f",
					i, cond, actionStr, cl.exp, cl.n, cl.fitness))
			end

			print("\n  Lowest Fitness Rules:")
			local sortedByF = {}
			for _, cl in ipairs(xcs.classifiers) do table.insert(sortedByF, cl) end
			table.sort(sortedByF, function(a, b) return (a.fitness or 0) < (b.fitness or 0) end)
			for i = 1, math.min(40, #sortedByF) do
				local cl = sortedByF[i]
				local cond = ""
				for _, bit in ipairs(cl.condition) do cond = cond .. tostring(bit) end
				local actionStr = cl.action.moveType or "unknown"
				if cl.action.jump then actionStr = actionStr .. "+jump" end
				print(string.format("    [%d] If %s -> %s | F=%.3f exp=%d n=%d as=%d",
					i, cond, actionStr, cl.fitness, cl.exp, cl.n, cl.as))
			end

			local recentGoals = 0
			local windowSize = 1000
			for _, ts in ipairs(xcs.goalTimestamps) do
				if ts > xcs.timeStep - windowSize then recentGoals += 1 end
			end
			print(string.format("  Goals total: %d  (last %d steps: %d)", xcs.goalCount, windowSize, recentGoals))
			print(string.format("  Avg steps between goals: %.0f", xcs.timeStep / math.max(xcs.goalCount, 1)))

			print("\n" .. string.rep("=", 60) .. "\n")
		end
	end)
end

simulateEvolution()
