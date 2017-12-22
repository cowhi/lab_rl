---
--- Created by Ruben Glatt, December 2017
---

local map_maker = require 'dmlab.system.map_maker'

local LEVEL_DATA = './.tmp/dmlab_level_data'
local make_map = {}

--- define possible pickups in map
local pickups = {
    A = 'apple_reward',           --- 1
    L = 'lemon_reward',           --- -1
    S = 'strawberry_reward',      --- 2
    F = 'fungi_reward',           --- -2
    G = 'goal',                   --- 10
}

function make_map.makeMap(mapName, mapEntityLayer, mapVariationsLayer)
  os.execute('mkdir -p ' .. LEVEL_DATA .. '/baselab')
  assert(mapName)
  map_maker:mapFromTextLevel{
      entityLayer = mapEntityLayer,
      variationsLayer = mapVariationsLayer,
      outputDir = LEVEL_DATA .. '/baselab',
      mapName = mapName,
      callback = function(i, j, c, maker)
        if pickups[c] then
          return maker:makeEntity(i, j, pickups[c])
        end
      end
  }
  return mapName
end

function make_map.commandLine(old_command_line)
  return old_command_line .. '+set sv_pure 0 +set fs_steampath ' .. LEVEL_DATA
end

function make_map.seedRng(value)
  map_maker:randomGen():seed(value)
end

function make_map.size()
    return map_maker:size()
end

return make_map
