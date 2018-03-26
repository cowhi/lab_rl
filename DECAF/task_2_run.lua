---
--- Created by Ruben Glatt, December 2017
---

local factory = require 'lab_rl.random_spawn_factory'

return factory.createLevelApi{
    mapName = 'task_2',
    episodeLengthSeconds = 60
}
