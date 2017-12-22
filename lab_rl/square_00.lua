---
--- Created by ruben.
--- DateTime: 30/11/17 18:58
---

local helper = require 'lab_rl.helper'

local map = {}

--[[
    Description of entity objects:
    * --> wall
    A --> apple +1
    F --> fungi -2
    G --> goal +10
    H --> horizontal door
    I --> vertical door
    L --> lemon -1
    P --> player
    S --> strawberry +2
--]]
map.entity = [[
*************
*GAAAAAAAAAA*
*P          *
*           *
*           *
*           *
*           *
*           *
*           *
*           *
*           *
*           *
*************
]]

--[[
    Define variation in texture to get different zones:
    'A' - 'Z' --> different zones
    '.' or ' ' --> no variation in texture
--]]
map.variation = [[
.............
.AAAAAAAAAAA.
.AAAAAAAAAAA.
.AAAAAAAAAAA.
.AAAAAAAAAAA.
.AAAAAAAAAAA.
.AAAAAAAAAAA.
.AAAAAAAAAAA.
.AAAAAAAAAAA.
.AAAAAAAAAAA.
.AAAAAAAAAAA.
.............
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