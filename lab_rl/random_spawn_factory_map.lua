---
--- Created by Ruben Glatt, December 2017
---

local random = require 'common.random'

local helper = require 'lab_rl.helper'
local pickups = require 'lab_rl.pickups'

local timeout = require 'decorators.timeout'
--local maze_gen = require 'dmlab.system.maze_generation'
local maze_gen = require 'lab_rl.map_maker'  -- Use when running first time?!


--[[ Creates a map where all objects spawn at random locations after reloading the map.
Keyword arguments:

*   `mapName` (string) - Name of map to load.
*   `episodeLengthSeconds` (number) - Episode length in seconds.
]]
local factory = {}

function factory.createLevelApi(kwargs)
    assert(kwargs.mapName and kwargs.episodeLengthSeconds)

    local map_name = kwargs.mapName
    local map_data = require('lab_rl.'..map_name)
    --local maze = maze_gen.MazeGeneration{entity = map_data.entity}
    local maze = maze_gen.makeMap(map_name, map_data.entity, map_data.variation) -- see maze_gen require

    local spawn_points = {}

    local api = {}

    function api:createPickup(class_name)
        return pickups.defaults[class_name]
    end

    function api:start(seed)
        api._count = 0
        api._finish_count = 0
        api._time_remaining = kwargs.episodeLengthSeconds
        random.seed(seed)

    end

    function api:updateSpawnVars(spawnVars)
        return {classname = spawnVars.classname,
                origin = table.remove(spawn_points, random.uniformInt(1, table.getn(spawn_points)))}
    end

    function api:nextMap()
        spawn_points = helper.copy_table(map_data.spawn_points)
        return map_name
    end

    timeout.decorate(api, kwargs.episodeLengthSeconds)
    return api
end

return factory



