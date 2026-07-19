-- XCS: Accuracy-based Learning Classifier System core.
-- Handles the classifier population, GA, matching, prediction updates and
-- the pheromone-trail reinforcement layer used to guide agent movement.

local rand = Random.new()

local ACTIONS = {
	{name = "forward", moveType = "forward"},
	{name = "backward", moveType = "backward"},
	{name = "left", moveType = "left"},
	{name = "right", moveType = "right"},
}

local XCS_CONFIG = {
	max_population = 800,   -- population size limit
	mu = 0.05,              -- mutation probability per allele
	chi = 0.8,               -- crossover probability
	theta_del = 25,          -- deletion threshold for experience
	delta = 0.4,             -- deletion fitness fraction
	theta_sub = 40,          -- subsumption experience threshold
	P_hash = 0.3,            -- probability of '#' when covering
	pI = 0.0001,             -- initial prediction
	eI = 0.0001,             -- initial error
	F_I = 0.0001,            -- initial fitness
	theta_mna = 1,           -- minimal number of actions in match set
	doGASubsumption = true,
	doActionSetSubsumption = true,
	theta_GA = 50,
	epsilon0 = 1.0,
	beta = 0.1,

	p_explore = 0.70,                -- before first goal (random walk)
	p_explore_after_goal = 0.3,      -- immediately after first goal
	p_explore_min = 0.1,             -- lower bound if decaying
	explore_decay_after_goal = 0.9,  -- optional gradual decay

	theta_mna_explore = 2,
}

local WEAK_CULL_CFG = {
	fitLow = 0.10,
	fitHigh = 0.30,
	minExpForJudgement = 20,   -- don't punish newborns immediately
	maxKeepWeak = 8,           -- keep only a small weak reserve
	maxNumerosityForWeak = 1,  -- flatten weak rules before optional removal
	ageForHardCull = 2000,     -- hard-cull old weak rules
}

local ALPHA = 0.1
local GAMMA = 0.97 -- discount factor

local XCS = {}
XCS.__index = XCS

-- exposed so other modules can share the same action set / config
XCS.ACTIONS = ACTIONS
XCS.CONFIG = XCS_CONFIG

local function randomAction()
	local idx = rand:NextInteger(1, #ACTIONS)
	local actionDef = ACTIONS[idx]
	return {
		moveType = actionDef.moveType,
		jump = false
	}
end

local function actionSignature(action)
	return string.format("%s|%d", action.moveType, action.jump and 1 or 0)
end

local function tablesMatchWithWildcard(condition, state)
	for i = 1, #state do
		local c_val = condition[i]
		local s_val = state[i]
		if c_val ~= '#' and c_val ~= s_val then
			return false
		end
	end
	return true
end

local function cloneClassifier(cl)
	local copy = {
		condition = {},
		action = { moveType = cl.action.moveType, jump = cl.action.jump },
		actionSig = cl.actionSig,
		prediction = cl.prediction,
		error = cl.error,
		fitness = cl.fitness,
		exp = cl.exp or 0,
		ts = cl.ts or 0,
		n = cl.n or 1,
		as = cl.as or 1,
	}
	for i = 1, #cl.condition do copy.condition[i] = cl.condition[i] end
	return copy
end

local function doesSubsume(subsumer, subsumed)
	if subsumer.actionSig ~= subsumed.actionSig then return false end
	if subsumer.exp <= XCS_CONFIG.theta_sub then return false end
	if subsumer.error >= XCS_CONFIG.epsilon0 then return false end

	local subsumer_wildcards, subsumed_wildcards = 0, 0
	for i = 1, #subsumer.condition do
		if subsumer.condition[i] == "#" then subsumer_wildcards += 1 end
		if subsumed.condition[i] == "#" then subsumed_wildcards += 1 end
	end
	if subsumer_wildcards <= subsumed_wildcards then return false end

	for i = 1, #subsumer.condition do
		local sub_bit = subsumer.condition[i]
		local subd_bit = subsumed.condition[i]
		if sub_bit ~= "#" and sub_bit ~= subd_bit then return false end
	end

	return true
end

local function couldSubsume(cl)
	return cl.exp > XCS_CONFIG.theta_sub and cl.error < XCS_CONFIG.epsilon0
end

local function isMoreGeneral(clgen, clspec)
	local genWild, specWild = 0, 0
	for i = 1, #clgen.condition do
		if clgen.condition[i] == "#" then genWild += 1 end
		if clspec.condition[i] == "#" then specWild += 1 end
	end
	if genWild <= specWild then return false end

	for i = 1, #clgen.condition do
		if clgen.condition[i] ~= "#" and clgen.condition[i] ~= clspec.condition[i] then
			return false
		end
	end
	return true
end

local function tournamentSelect(actionSet, k)
	if #actionSet == 1 then return actionSet[1] end
	if #actionSet < 4 then k = 2 end

	local best = nil
	for _ = 1, k do
		local candidate = actionSet[rand:NextInteger(1, #actionSet)]
		if best == nil or candidate.fitness > best.fitness then
			best = candidate
		end
	end
	return best
end

local function hybridCrossover(child1, child2)
	local cp = rand:NextInteger(1, 4)
	for i = cp + 1, 5 do
		child1.condition[i], child2.condition[i] = child2.condition[i], child1.condition[i]
	end

	if rand:NextNumber() < 0.5 then
		child1.condition[6], child2.condition[6] = child2.condition[6], child1.condition[6]
		child1.condition[7], child2.condition[7] = child2.condition[7], child1.condition[7]
	end

	if rand:NextNumber() < 0.5 then
		child1.condition[8], child2.condition[8] = child2.condition[8], child1.condition[8]
	end
end

local function mutateClassifier(cl, situation)
	local VISION_BITS = 5
	local muRate = XCS_CONFIG.mu

	for i = 1, #cl.condition do
		if rand:NextNumber() < muRate then
			if cl.condition[i] == "#" then
				cl.condition[i] = situation[i]
			elseif i <= VISION_BITS then
				cl.condition[i] = "#"
			else
				cl.condition[i] = 1 - cl.condition[i]
			end
		end
	end

	if rand:NextNumber() < XCS_CONFIG.mu then
		cl.action = randomAction()
		cl.actionSig = actionSignature(cl.action)
	end
end

local function calculateDeletionVote(cl, avgFitnessInPopulation)
	-- niche size * numerosity
	local vote = cl.as * cl.n
	local fitnessPerMicro = cl.fitness / math.max(cl.n, 1)

	if cl.exp > XCS_CONFIG.theta_del then
		-- standard XCS deletion: penalize experienced low-fitness rules
		if fitnessPerMicro < XCS_CONFIG.delta * avgFitnessInPopulation then
			vote = vote * (avgFitnessInPopulation / math.max(fitnessPerMicro, 0.0001))
		end
	else
		if cl.fitness < 0.3 and cl.n > 6 then
			vote = vote * 10.0 * (avgFitnessInPopulation / math.max(fitnessPerMicro, 0.0001))
		end
	end

	local wildcards = 0
	for _, bit in ipairs(cl.condition) do
		if bit == "#" then wildcards += 1 end
	end
	local wildcardRatio = wildcards / #cl.condition
	-- penalize over-general rules even if fit, they crowd out specific ones
	if wildcardRatio > 0.55 and cl.exp > 500 then
		vote = vote * 1.5
	end

	return vote
end

local function getDirectionBetween(cellA, cellB)
	local ax, az = cellA:match("(-?%d+),(-?%d+)")
	local bx, bz = cellB:match("(-?%d+),(-?%d+)")
	ax, az = tonumber(ax), tonumber(az)
	bx, bz = tonumber(bx), tonumber(bz)

	if bx == ax and bz == az - 1 then return "forward"
	elseif bx == ax and bz == az + 1 then return "backward"
	elseif bx == ax + 1 and bz == az then return "right"
	elseif bx == ax - 1 and bz == az then return "left"
	end
	return nil
end

local function isValidStep(cellA, cellB)
	local ax, az = cellA:match("(-?%d+),(-?%d+)")
	local bx, bz = cellB:match("(-?%d+),(-?%d+)")
	ax, az = tonumber(ax), tonumber(az)
	bx, bz = tonumber(bx), tonumber(bz)

	local dx = math.abs(bx - ax)
	local dz = math.abs(bz - az)
	return (dx + dz) == 1
end

-- large, sparse reward for reaching the goal region
function XCS.calculateGoalReward(position, goalPart)
	local distance = (position - goalPart.Position).Magnitude
	local GOAL_DISTANCE_THRESHOLD = 6 -- studs
	if distance < GOAL_DISTANCE_THRESHOLD then
		return 100.0
	end
	return 0.0
end

function XCS.new()
	local self = setmetatable({}, XCS)
	self.classifiers = {}
	self.pheromone = {}
	self.timeStep = 0
	self.bestPathLength = nil

	self.goalCount = 0
	self.totalPathLength = nil

	self.hasReachedGoal = false
	self.currentExplore = XCS_CONFIG.p_explore

	self.goalTimestamps = {}

	return self
end

function XCS:getPheroCell(cell)
	local val = self.pheromone[cell]
	if type(val) == "table" then
		return val
	end

	-- convert legacy scalar entries to the per-direction table format
	local newVal = {
		forward = val or 0,
		left = val or 0,
		right = val or 0,
		backward = val or 0,
	}
	self.pheromone[cell] = newVal
	return newVal
end

function XCS:postSuccessReinforce(path, pathLength, isBest, quality)
	quality = quality or 1.0

	local minStrength = 3.0
	local maxStrength = isBest and 20.0 or 6.0

	print(string.format("postSuccessReinforce: pathLength=%d isBest=%s quality=%.2f min=%.1f max=%.1f",
		pathLength, tostring(isBest), quality, minStrength * quality, maxStrength * quality))

	local validSteps = 0
	local reinforceLimit = 246
	local startIndex = math.max(1, #path - reinforceLimit)

	for i = #path - 1, startIndex, -1 do
		local cell = path[i]
		local nextCell = path[i + 1]

		if not isValidStep(cell, nextCell) then continue end
		validSteps += 1

		local dir = getDirectionBetween(cell, nextCell)
		if not dir then continue end

		local t = (#path - i) / math.max(pathLength - 1, 1)
		local strength = maxStrength * math.exp(-t * 3) * quality

		local phero = self:getPheroCell(cell)
		phero[dir] = math.min(20, (phero[dir] or 0) + strength)

		if isBest then
			for _, actionDef in ipairs(ACTIONS) do
				if actionDef.moveType ~= dir then
					local competing = phero[actionDef.moveType] or 0
					if competing > 0 then
						phero[actionDef.moveType] = competing * 0.7
					end
				end
			end
		end
	end

	print(string.format("  Valid steps: %d / %d", validSteps, pathLength - 1))
end

function XCS:decayPheromones(rate)
	rate = rate or 0.995
	local active, total = 0, 0

	for _, dirs in pairs(self.pheromone) do
		for dir, val in pairs(dirs) do
			local newVal = val * rate
			if math.abs(newVal) < 0.05 then
				newVal = 0
			elseif math.abs(newVal) > 1 then
				active += 1
			end
			dirs[dir] = newVal
			total += 1
		end
	end

	if active / math.max(total, 1) < 0.01 then
		self.hasReachedGoal = false
		self.currentExplore = XCS_CONFIG.p_explore
		print("Resetting to exploration")
	end
end

function XCS:printRules(limit)
	limit = limit or #self.classifiers

	local sortedClassifiers = {}
	for _, cl in ipairs(self.classifiers) do
		table.insert(sortedClassifiers, cl)
	end
	table.sort(sortedClassifiers, function(a, b)
		return (a.fitness or 0) > (b.fitness or 0)
	end)

	print("\n=== CURRENT CLASSIFIER RULES (sorted by fitness) ===")

	for i, cl in ipairs(sortedClassifiers) do
		if i > limit then break end

		local cond = ""
		for _, bit in ipairs(cl.condition) do
			cond = cond .. tostring(bit)
		end

		local actionStr = cl.action.moveType or "unknown"
		if cl.action.jump then actionStr = actionStr .. "+jump" end

		local p = cl.prediction or 0
		local e = cl.error or 0
		local F = cl.fitness or 0
		local exp = cl.exp or 0
		local n = cl.n or 1
		local as = cl.as or 1

		print(string.format(
			"[%d] If %s -> %s | p=%.3f e=%.3f F=%.3f exp=%d n=%d as=%.1f",
			i, cond, actionStr, p, e, F, exp, n, as
		))
	end

	print(string.format("Total classifiers: %d (limit shown: %d)",
		#self.classifiers, math.min(limit, #self.classifiers)))
	print("=================================\n")
end

function XCS:actionSetSubsumption(actionSet)
	if not XCS_CONFIG.doActionSetSubsumption then return end

	local bestCl = nil
	local maxWildcards = -1

	for _, cl in ipairs(actionSet) do
		if couldSubsume(cl) then
			local wildcards = 0
			for _, bit in ipairs(cl.condition) do
				if bit == "#" then wildcards += 1 end
			end
			if (wildcards > maxWildcards) or (wildcards == maxWildcards and rand:NextNumber() < 0.5) then
				maxWildcards = wildcards
				bestCl = cl
			end
		end
	end

	if not bestCl then return end

	local function sameCondition(a, b)
		for i = 1, #a.condition do
			if a.condition[i] ~= b.condition[i] then return false end
		end
		return true
	end

	local toRemove = {}
	for _, cl in ipairs(actionSet) do
		if cl ~= bestCl and cl.actionSig == bestCl.actionSig then
			if sameCondition(bestCl, cl) or isMoreGeneral(bestCl, cl) then
				bestCl.n += cl.n
				table.insert(toRemove, cl)
			end
		end
	end

	for _, cl in ipairs(toRemove) do
		for i = #self.classifiers, 1, -1 do
			if self.classifiers[i] == cl then
				table.remove(self.classifiers, i)
				break
			end
		end
	end
end

function XCS:cullJunkByAge()
	for i = #self.classifiers, 1, -1 do
		local cl = self.classifiers[i]
		local age = (self.timeStep or 0) - (cl.ts or 0)
		if (cl.exp or 0) == 0 and age >= 200 and (cl.fitness or 0) <= 0.6 then
			table.remove(self.classifiers, i)
		end
	end
end

function XCS:cullWeakFitnessBucket()
	local weak = {}
	for i, cl in ipairs(self.classifiers) do
		local f = cl.fitness or 0
		if f >= WEAK_CULL_CFG.fitLow and f <= WEAK_CULL_CFG.fitHigh then
			table.insert(weak, {idx = i, cl = cl})
		end
	end

	if #weak <= WEAK_CULL_CFG.maxKeepWeak then return end

	-- sort weakest first, then oldest first
	table.sort(weak, function(a, b)
		if (a.cl.fitness or 0) == (b.cl.fitness or 0) then
			return (a.cl.exp or 0) > (b.cl.exp or 0)
		end
		return (a.cl.fitness or 0) < (b.cl.fitness or 0)
	end)

	local toCull = #weak - WEAK_CULL_CFG.maxKeepWeak
	for k = 1, toCull do
		local cl = weak[k].cl
		local age = (self.timeStep or 0) - (cl.ts or 0)

		if (cl.n or 1) > WEAK_CULL_CFG.maxNumerosityForWeak then
			cl.n = WEAK_CULL_CFG.maxNumerosityForWeak
		elseif age >= WEAK_CULL_CFG.ageForHardCull then
			for j = #self.classifiers, 1, -1 do
				if self.classifiers[j] == cl then
					table.remove(self.classifiers, j)
					break
				end
			end
		end
	end
end

function XCS:performMaintenance()
	self:cullJunkByAge()
	self:cullWeakFitnessBucket()

	for _, cl in ipairs(self.classifiers) do
		-- cap action-set-size estimate for low-fitness rules so they don't
		-- get inflated base deletion votes
		if cl.fitness < 0.4 and cl.as > 3 then
			cl.as = cl.as * 0.9
		end
		if cl.fitness < 0.6 and cl.n > 3 then
			cl.n = math.max(1, cl.n - 1)
		end
	end

	-- roulette deletion until within population cap
	while true do
		local totalNumerosity = 0
		for _, cl in ipairs(self.classifiers) do
			totalNumerosity += (cl.n or 1)
		end
		if totalNumerosity <= XCS_CONFIG.max_population then return end

		local totalFitness = 0
		for _, cl in ipairs(self.classifiers) do
			totalFitness += (cl.fitness or 0)
		end
		local avgFitness = totalFitness / math.max(totalNumerosity, 1)

		local voteSum = 0
		local votes = {}
		for _, cl in ipairs(self.classifiers) do
			local vote = calculateDeletionVote(cl, avgFitness)
			votes[cl] = vote
			voteSum += vote
		end
		if voteSum <= 0 then return end

		local choicePoint = rand:NextNumber() * voteSum
		local acc = 0
		for i, cl in ipairs(self.classifiers) do
			acc += (votes[cl] or 0)
			if acc >= choicePoint then
				if (cl.n or 1) > 1 then
					cl.n -= 1
				else
					table.remove(self.classifiers, i)
				end
				break -- exactly one micro-deletion per iteration
			end
		end
	end
end

function XCS:insertChild(child, parent1, parent2)
	if XCS_CONFIG.doGASubsumption and parent1 and doesSubsume(parent1, child) then
		parent1.n = parent1.n + 1
		return
	elseif XCS_CONFIG.doGASubsumption and parent2 and doesSubsume(parent2, child) then
		parent2.n = parent2.n + 1
		return
	end

	local function sameCondition(c1, c2)
		for i = 1, #c1.condition do
			local a = c1.condition[i]
			local b = c2.condition[i]
			if a ~= "#" and b ~= "#" and a ~= b then
				return false
			end
		end
		return true
	end

	for _, existing in ipairs(self.classifiers) do
		if existing.actionSig == child.actionSig and sameCondition(existing, child) then
			existing.n = existing.n + 1
			return
		end
	end

	print("New classifier inserted")
	table.insert(self.classifiers, child)
	self:performMaintenance()
end

function XCS:generateCoveringClassifier(matchSet, state)
	local present = {}
	for _, c in ipairs(matchSet) do present[c.actionSig] = true end

	local candidates = {}
	for _, actionDef in ipairs(ACTIONS) do
		local act = { moveType = actionDef.moveType, jump = false }
		local sig = actionSignature(act)
		if not present[sig] then
			table.insert(candidates, act)
		end
	end

	local chosenAction = candidates[rand:NextInteger(1, #candidates)] or randomAction()
	local cl = { condition = {}, action = chosenAction, actionSig = actionSignature(chosenAction) }

	local VISION_BITS = 5 -- bits 1-5 are vision; the rest are pheromone bits

	for i = 1, #state do
		if i <= VISION_BITS and rand:NextNumber() < XCS_CONFIG.P_hash then
			cl.condition[i] = "#"
		else
			-- pheromone bits are never wildcarded, always copied exactly
			cl.condition[i] = state[i]
		end
	end

	cl.prediction = XCS_CONFIG.pI
	cl.error = XCS_CONFIG.eI
	cl.fitness = XCS_CONFIG.F_I
	cl.exp = 0
	cl.ts = self.timeStep
	cl.as = 1
	cl.n = 1

	return cl
end

function XCS:generateMatchSet(state)
	local M = {}
	local theta_mna_ = self.hasReachedGoal and XCS_CONFIG.theta_mna or XCS_CONFIG.theta_mna_explore

	while #M == 0 do
		for _, cl in ipairs(self.classifiers) do
			if #cl.condition == #state and tablesMatchWithWildcard(cl.condition, state) then
				table.insert(M, cl)
			end
		end

		local actionCount = {}
		for _, c in ipairs(M) do actionCount[c.actionSig] = true end
		local numActions = 0
		for _ in pairs(actionCount) do numActions += 1 end

		if numActions < theta_mna_ then
			local newCl = self:generateCoveringClassifier(M, state)
			table.insert(self.classifiers, newCl)
			table.insert(M, newCl)
			self:performMaintenance()
			M = {} -- rebuild match set from scratch
		end
	end

	return M
end

function XCS:predictionArrayFromMatchSet(matchSet)
	local sums = {}
	local weights = {}
	for _, c in ipairs(matchSet) do
		local k = c.actionSig
		sums[k] = (sums[k] or 0) + (c.prediction * c.fitness)
		weights[k] = (weights[k] or 0) + c.fitness
	end
	local PA = {}
	for k, s in pairs(sums) do
		PA[k] = s / weights[k]
	end
	return PA
end

-- epsilon-greedy action selection using PA; returns action table and actionSet
function XCS:selectActionFromMatch(matchSet, PA, currentCell)
	local epsilon = self.currentExplore or XCS_CONFIG.p_explore
	print("Exploration", epsilon)

	if rand:NextNumber() < epsilon then
		local randomCl = matchSet[rand:NextInteger(1, #matchSet)]
		local actionSet = {}
		for _, c in ipairs(matchSet) do
			if c.actionSig == randomCl.actionSig then
				table.insert(actionSet, c)
			end
		end
		return randomCl.action, actionSet
	end

	local bestSig, bestVal = nil, -math.huge
	for sig, val in pairs(PA) do
		if val > bestVal then
			bestVal, bestSig = val, sig
		end
	end

	if not bestSig then
		local c = matchSet[rand:NextInteger(1, #matchSet)]
		local actionSet = {}
		for _, cl in ipairs(matchSet) do
			if cl.actionSig == c.actionSig then
				table.insert(actionSet, cl)
			end
		end
		return c.action, actionSet
	end

	local actionSet = {}
	for _, c in ipairs(matchSet) do
		if c.actionSig == bestSig then
			table.insert(actionSet, c)
		end
	end

	if #actionSet == 0 then
		local c = matchSet[rand:NextInteger(1, #matchSet)]
		return c.action, { c }
	end

	return actionSet[1].action, actionSet
end

function XCS:runGA(actionSet, state)
	local avgTimeSinceGA = 0
	local totalNumerosity = 0

	for _, cl in ipairs(actionSet) do
		avgTimeSinceGA += (self.timeStep - cl.ts) * cl.n
		totalNumerosity += cl.n
	end
	avgTimeSinceGA = avgTimeSinceGA / math.max(totalNumerosity, 1)

	if avgTimeSinceGA <= XCS_CONFIG.theta_GA then return end

	for _, cl in ipairs(actionSet) do
		cl.ts = self.timeStep
	end

	local expandedSet = {}
	for _, cl in ipairs(actionSet) do
		table.insert(expandedSet, cl)
	end
	if #expandedSet < 2 then return end

	local parent1 = tournamentSelect(expandedSet, 2)
	local parent2 = tournamentSelect(expandedSet, 2)
	if not parent1 or not parent2 then
		print("GA: parent selection failed")
		return
	end
	if parent1 == parent2 then return end

	local child1 = cloneClassifier(parent1)
	local child2 = cloneClassifier(parent2)
	child1.n, child2.n = 1, 1
	child1.exp, child2.exp = 0, 0

	if rand:NextNumber() < XCS_CONFIG.chi then
		hybridCrossover(child1, child2)

		local avgP = (parent1.prediction + parent2.prediction) / 2
		local avgE = (parent1.error + parent2.error) / 2
		local avgF = (parent1.fitness + parent2.fitness) / 2

		child1.prediction, child1.error, child1.fitness = avgP, avgE, avgF
		child2.prediction, child2.error, child2.fitness = child1.prediction, child1.error, child1.fitness
	end

	child1.fitness = child1.fitness * 0.2
	child2.fitness = child2.fitness * 0.2

	mutateClassifier(child1, state)
	mutateClassifier(child2, state)
	child1.actionSig = actionSignature(child1.action)
	child2.actionSig = actionSignature(child2.action)

	self:insertChild(child1, parent1, parent2)
	self:insertChild(child2, parent1, parent2)
end

function XCS:updateQ(prevSet, reward, maxPred)
	if not prevSet or #prevSet == 0 then return end

	local target = reward + GAMMA * maxPred

	-- update prediction, error, action set size estimate, experience
	local actionSetSize = 0
	for _, cl in ipairs(prevSet) do
		actionSetSize += cl.n
	end

	for _, cl in ipairs(prevSet) do
		cl.exp = cl.exp + 1

		local beta = XCS_CONFIG.beta
		local diff = target - cl.prediction
		local absErr = math.abs(diff)

		if cl.exp < (1 / beta) then
			cl.prediction = cl.prediction + (diff / cl.exp)
			cl.error = cl.error + ((absErr - cl.error) / cl.exp)
			cl.as = cl.as + ((actionSetSize - cl.as) / cl.exp)
		else
			cl.prediction = cl.prediction + beta * diff
			cl.error = cl.error + beta * (absErr - cl.error)
			cl.as = cl.as + beta * (actionSetSize - cl.as)
		end
	end

	-- update fitness (requires the whole action set)
	local nu = 5
	local accuracySum = 0
	local accuracies = {}

	for _, cl in ipairs(prevSet) do
		local kappa
		if cl.error < XCS_CONFIG.epsilon0 then
			kappa = 1
		else
			kappa = ALPHA * (cl.error / XCS_CONFIG.epsilon0) ^ (-nu)
		end
		accuracies[cl] = kappa
		accuracySum += accuracies[cl] * cl.n
	end
	if accuracySum == 0 then accuracySum = 0.0001 end

	for _, cl in ipairs(prevSet) do
		local relativeAccuracy = (accuracies[cl] * cl.n) / accuracySum
		cl.fitness = cl.fitness + XCS_CONFIG.beta * (relativeAccuracy - cl.fitness)
	end

	if XCS_CONFIG.doActionSetSubsumption then
		self:actionSetSubsumption(prevSet)
	end
end

function XCS:printPheromoneMap()
	local cf, size = workspace.Maze:GetBoundingBox()
	local originX = cf.Position.X - size.X / 2
	local originZ = cf.Position.Z - size.Z / 2
	local gs = 4 -- gridSize

	local cols = math.floor(size.X / gs)
	local rows = math.floor(size.Z / gs)

	print("\n=== PHEROMONE MAP ===")
	for z = 0, rows - 1 do
		local line = ""
		for x = 0, cols - 1 do
			local key = string.format("%d,%d", x, z)
			local dirs = self.pheromone[key]
			if not dirs then
				line = line .. "  .  "
			else
				local best, bestVal = "?", -math.huge
				for dir, val in pairs(dirs) do
					if val > bestVal then
						bestVal = val
						best = dir
					end
				end
				local symbol = "."
				if bestVal > 2 then
					if best == "forward" then symbol = "^"
					elseif best == "backward" then symbol = "v"
					elseif best == "left" then symbol = "<"
					elseif best == "right" then symbol = ">"
					end
				elseif bestVal < -2 then
					symbol = "x" -- negative / dead end
				end
				line = line .. string.format(" %s%+.0f", symbol, bestVal)
			end
		end
		print(string.format("%2d|%s|", z, line))
	end
	print("Legend: ^v<>=good direction  x=bad  .=unknown")
	print("====================\n")
end

function XCS:printFitnessDebug()
	local buckets = {
		{min = 0,    max = 0.1,  count = 0, label = "dead   (0-0.1)"},
		{min = 0.1,  max = 0.3,  count = 0, label = "weak   (0.1-0.3)"},
		{min = 0.3,  max = 0.6,  count = 0, label = "medium (0.3-0.6)"},
		{min = 0.6,  max = 0.9,  count = 0, label = "good   (0.6-0.9)"},
		{min = 0.9,  max = 1.01, count = 0, label = "elite  (0.9-1.0)"},
	}

	local totalRewardEver = 0
	local zeroRewardRules = 0

	for _, cl in ipairs(self.classifiers) do
		for _, b in ipairs(buckets) do
			if cl.fitness >= b.min and cl.fitness < b.max then
				b.count += 1
				break
			end
		end
		if cl.prediction < 0.01 then
			zeroRewardRules += 1
		end
		totalRewardEver += cl.prediction
	end

	print("=== FITNESS DISTRIBUTION ===")
	for _, b in ipairs(buckets) do
		local bar = string.rep("#", b.count)
		print(string.format("  %s: %d %s", b.label, b.count, bar))
	end
	print(string.format("  Zero prediction rules: %d / %d", zeroRewardRules, #self.classifiers))
	print(string.format("  Avg prediction: %.3f", totalRewardEver / math.max(#self.classifiers, 1)))
	print("============================")
end

return XCS
