return [[
local socket = require("socket")
local client = socket.tcp()

local success, connection_err = client:connect("127.0.0.1", 12345)
if not success then
    print("Connection failed: " .. connection_err)
    return
end
client:settimeout(0) -- Non-blocking after connection

-- Credit to Henrik Ilgen (https://stackoverflow.com/a/6081639)
function serialize_table(val, name, skipnewlines, depth)
    skipnewlines = skipnewlines or false
    depth = depth or 0
    local tmp = string.rep(" ", depth)
    if name then
        tmp = tmp .. name .. " = "
    end
    if type(val) == "table" then
        tmp = tmp .. "{" .. (not skipnewlines and "\n" or "")
        for k, v in pairs(val) do
            tmp = tmp .. serialize_table(v, k, skipnewlines, depth + 1) .. "," .. (not skipnewlines and "\n" or "")
        end
        tmp = tmp .. string.rep(" ", depth) .. "}"
    elseif type(val) == "number" then
        tmp = tmp .. tostring(val)
    elseif type(val) == "string" then
        tmp = tmp .. string.format("%q", val)
    elseif type(val) == "boolean" then
        tmp = tmp .. (val and "true" or "false")
    else
        tmp = tmp .. '"[inserializeable datatype:' .. type(val) .. ']"'
    end
    return tmp
end

-- Set up channels for communication with the main thread
local to_network_channel = love.thread.getChannel("to_network")
local from_network_channel = love.thread.getChannel("from_network")

while true do
    -- Check for commands from the main thread
    local command = to_network_channel:pop()
    if command then
        if command.type == "send" then
            client:send(command.data .. "\n")
        end
        -- Additional command types can be added here if needed
    end

    -- Check for incoming data from the socket
    local data, err = client:receive()
    if data then
        from_network_channel:push(data)
    elseif err ~= "timeout" then
        print("Receive error: " .. err)
    end

    -- Sleep briefly to prevent 100% CPU usage
    socket.sleep(0.01) -- 10ms
end
]]
