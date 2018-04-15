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
    F --> fungi -4
    G --> goal +1
    H --> horizontal door
    I --> vertical door
    L --> lemon -2
    P --> player
    S --> strawberry +4
--]]
map.entity = [[
********
*G     *
* LLLL *
*  LL  *
* LLLL *
*      *
*     P*
********
]]

--[[
    Define variation in texture to get different zones:
    'A' - 'Z' --> different zones
    '.' or ' ' --> no variation in texture
--]]
map.variation = [[
........
.AAAAAA.
.AAAAAA.
.AAAAAA.
.AAAAAA.
.AAAAAA.
.AAAAAA.
........
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
