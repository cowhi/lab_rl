---
--- Created by Ruben Glatt, December 2017
---

local helper = {}

function helper.copy_table(t)
    local t2 = {}
    for k,v in pairs(t) do
        t2[k] = v
    end
    return t2
end

function helper.create_spawn_points(rows)
    local elements = 0
    local spawn_points = {}
    local row = rows - 1
    local columns = row
    while row > 2 do
        row = row - 1
        local column = columns
        while column > 2 do
            column = column - 1
            elements = elements + 1
            spawn_points[elements] = ''.. (column * 100) .. ' ' .. (row * 100) .. ' 20'
        end
    end
    return spawn_points
end

return helper