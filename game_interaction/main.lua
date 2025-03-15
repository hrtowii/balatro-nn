--- asdasdaSTEAMODDED HEADER
--- MOD_NAME: Balatro Neural Network
--- MOD_ID: balatro_nn
--- MOD_AUTHOR: [htrowii<3]
--- MOD_DESCRIPTION: Balatro neural network to get data from and interact with the game

----------------------------------------------
------------MOD CODE -------------------------
-- local love = require("love")
-- B_NN = SMODS.current_mod
-- B_NN.tags = {}
-- -- https://github.com/V-rtualized/BalatroMultiplayer/blob/f5b64035fecb48f572e6f35b170c14bc2e8ca46e/Core.lua
-- function load_mp_file(file)
--     local chunk, err = SMODS.load_file(file, "balatro_nn")
--     if chunk then
--         local ok, func = pcall(chunk)
--         if ok then
--             return func
--         else
--             print("Failed to process file: " .. func, "MULTIPLAYER")
--         end
--     else
--         print("Failed to find or compile file: " .. tostring(err), "MULTIPLAYER")
--     end
--     return nil
-- end

-- local SOCKET = load_mp_file("./src/socket.lua")
-- NETWORKING_THREAD = love.thread.newThread(SOCKET)
-- NETWORKING_THREAD:start()

-- assert(SMODS.load_file("./src/hooks.lua"))()
-- assert(SMODS.load_file("./src/get_game_state.lua"))()

-- -- my own utils 
-- function love.keypressed(key, u)
--     if key == "-" then
--         debug.debug()
--     elseif key == ";" then
--         G.GAME.blind:disable()
--     -- elseif key == "0" then
--     --     G.FUNCS.toggle_shop()
--     end
-- end

----------------------------------------------
------------MOD CODE END----------------------
