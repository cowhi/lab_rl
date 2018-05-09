---
--- Created by ruben.
--- DateTime: 30/11/17 18:58
---

local helper = require 'lab_rl.helper'

local map = {}

--[[
    Description of entity objects:
    * --> wall
    A --> apple +2
    F --> fungi -2
    G --> goal +3
    H --> horizontal door
    I --> vertical door
    L --> lemon -2
    P --> player
    S --> strawberry +2
--]]
map.entity = [[
******
*G  L*
*    *
*    *
*A  P*
******
]]

--[[
    Define variation in texture to get different zones:
    'A' - 'Z' --> different zones
    '.' or ' ' --> no variation in texture
--]]
map.variation = [[
......
.AAAA.
.AAAA.
.AAAA.
.AAAA.
......
]]

--[[
    No variation
--]]
map.variation = {}


--[[
    Define all possible spawn positions
--]]
map.spawn_points = helper.create_spawn_points(select(2, map.entity:gsub('\n', '\n')))

return map