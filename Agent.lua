-- Agent: wraps one NPC's vision, per-episode memory and state-building
-- logic. Multiple agents share a single XCS instance.

local XCS = require(script.Parent.XCS)
local VisionSystem = require(game.ServerScriptService:WaitForChild("VisionSystem"))

local ACTIONS = XCS.ACTIONS

local Agent = {}
Agent.__index = Agent

function Agent.new(npc, xcs)
	local self = setmetatable({}, Agent)

	self.npc = npc
	self.vision = VisionSystem.new(npc)
	self.xcs = xcs -- shared XCS instance

	-- per-episode memory
	self.prevActionSet = nil
	self.prevReward = 0
	self.prevState = nil
	self.prevCell1 = nil
	self.lastCell = nil
	self.currentPath = {}
	self.sameCellSteps = 0
	self.lastActionBits = {0, 0}

	-- exploration memory
	self.visitedCells = {}
	self.totalCellsVisited = 0

	-- config
	self.gridSize = 4
	self.maxSameCellSteps = 10

	self.pendingUpdate = nil
	self.episodePath = {}

	return self
end

function Agent:resetEpisode()
	self.prevActionSet = nil
	self.prevReward = 0
	self.prevState = nil
	self.prevCell1 = nil
	self.lastCell = nil
	self.currentPath = {}
	self.sameCellSteps = 0
	self.lastActionBits = {0, 0}
	self.visitedCells = {}
	self.episodePath = {}
	-- visitedCells is intentionally NOT reset (long term memory)
end

function Agent:getGridKey(position)
	local cf, size = workspace.Maze:GetBoundingBox()
	local originX = cf.Position.X - size.X / 2
	local originZ = cf.Position.Z - size.Z / 2
	local x = math.floor((position.X - originX) / self.gridSize)
	local z = math.floor((position.Z - originZ) / self.gridSize)
	return string.format("%d,%d", x, z)
end

function Agent:isNewCell(position)
	local key = self:getGridKey(position)
	if not self.visitedCells[key] then
		self.visitedCells[key] = 1
		self.totalCellsVisited += 1
		return 1.0
	end

	local visits = self.visitedCells[key]
	self.visitedCells[key] += 1
	if visits >= 3 then return 0 end
	return 0.1
end

function Agent:updateOtherNpcs(allAgents)
	self.otherNpcs = {}
	for _, other in ipairs(allAgents) do
		if other ~= self then
			table.insert(self.otherNpcs, other.npc)
		end
	end
end

function Agent:buildState(agent)
	self.vision:updateVision(agent.otherNpcs)
	local visionData = self.vision:getVisionData()

	local state = {}
	for _, bit in ipairs(visionData) do
		table.insert(state, bit)
	end

	local pos = self.npc:GetPivot().Position
	local cell = self:getGridKey(pos)
	local phero = self.xcs:getPheroCell(cell)

	local bestDir, bestVal = nil, -math.huge
	for _, actionDef in ipairs(ACTIONS) do
		local v = phero[actionDef.moveType] or 0
		if v > bestVal then
			bestVal = v
			bestDir = actionDef.moveType
		end
	end

	local pheroSignal = bestVal > 2.0

	-- 2-bit encoding of the recommended pheromone direction
	local dirEncoding = {
		forward = {0, 0},
		right = {0, 1},
		left = {1, 0},
		backward = {1, 1},
	}

	if pheroSignal and bestDir then
		local enc = dirEncoding[bestDir]
		table.insert(state, enc[1])
		table.insert(state, enc[2])
	else
		table.insert(state, 0)
		table.insert(state, 0)
	end
	table.insert(state, pheroSignal and 1 or 0)

	return state, visionData
end

return Agent
