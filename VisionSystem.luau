local VisionSystem = {}
VisionSystem.__index = VisionSystem

local GRID_DISTANCE = 7-- max ray length
local NUM_RAYS = 5

local SEGMENTS = 4
local segmentLength = GRID_DISTANCE / SEGMENTS

local FOV = math.rad(180)

function VisionSystem.new(npc)
	local self = setmetatable({}, VisionSystem)
	self.npc = npc
	self.gridValues = {}
	return self
end

local function castRay(origin, direction, npc, extraFilter)
	local params = RaycastParams.new()
	params.FilterType = Enum.RaycastFilterType.Blacklist

	local filterList = {npc}
	if extraFilter then
		for _, obj in ipairs(extraFilter) do
			table.insert(filterList, obj)
		end
	end

	params.FilterDescendantsInstances = filterList
	return workspace:Raycast(origin, direction * GRID_DISTANCE, params)
end

function VisionSystem:updateVision(extraFilter)
	self.gridValues = {}
	self.gridLengthes = {}

	local humanoidRoot = self.npc:FindFirstChild("HumanoidRootPart")
	if not humanoidRoot then return self.gridValues end

	local centerPos = humanoidRoot.Position
	local forwardDir = humanoidRoot.CFrame.LookVector
	
	
	for i = 1, NUM_RAYS do
		local angle = -FOV/2 + (i-1)*(FOV/(NUM_RAYS-1))
		--local angle = -FOV/2 + (i-1) * (FOV / NUM_RAYS)
		local dir = (CFrame.Angles(0, angle, 0) * humanoidRoot.CFrame.LookVector)
		local result = castRay(centerPos, dir, self.npc, extraFilter)
	
		local bits = {}

		if result then
			local dist = (result.Position - centerPos).Magnitude
			self.gridLengthes[i] = dist
			if dist < GRID_DISTANCE then
				bits = 1
			else
				bits = 0 
			end
		else
			bits = 0
			self.gridLengthes[i] = GRID_DISTANCE
		end

		table.insert(self.gridValues, bits)
		
	end


	return self.gridValues, self.gridLengthes
end

function VisionSystem:getVisionData()
	local flatData = {}
	local flatData2 = {}
	for i = 1, NUM_RAYS * 2 do
		table.insert(flatData, self.gridValues[i])
		
	end
	for i = 1, NUM_RAYS do
		table.insert(flatData2, self.gridLengthes[i])
	end
	return flatData, flatData2
	
end

-- Visualize rays in workspace
function VisionSystem:visualizeRays()
	local humanoidRoot = self.npc:FindFirstChild("HumanoidRootPart")
	if not humanoidRoot then return end

	local centerPos = humanoidRoot.Position
	local forwardDir = humanoidRoot.CFrame.LookVector
	
	for i = 1, NUM_RAYS do
	    local angle = -FOV/2 + (i-1)*(FOV/(NUM_RAYS-1))
		--local angle = -FOV/2 + (i-1) * (FOV / NUM_RAYS)
		local dir = (CFrame.Angles(0, angle, 0) * humanoidRoot.CFrame.LookVector)
		local result = castRay(centerPos, dir, self.npc)
		
		

		local rayDist = result and (result.Position - centerPos).Magnitude or GRID_DISTANCE
		local rayDir = dir * rayDist

				-- Part for visualization
		local rayPart = Instance.new("Part")
		rayPart.Size = Vector3.new(0.05, 0.05, rayDist)
		rayPart.CFrame = CFrame.lookAt(centerPos, centerPos + rayDir) * CFrame.new(0, 0, -rayDist/2)
		rayPart.Anchored = true
		rayPart.CanCollide = false
		if result and rayDist < GRID_DISTANCE  then
			-- hit occurs after half distance > mark red
			rayPart.Color = Color3.new(1, 0, 0)
		else
			-- either no hit, or hit is close
			rayPart.Color = Color3.new(0, 1, 0)
		end
		rayPart.Parent = workspace

		game:GetService("Debris"):AddItem(rayPart, 0.1)
	end

end

return VisionSystem





