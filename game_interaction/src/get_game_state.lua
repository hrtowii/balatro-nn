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

to_network_channel = love.thread.getChannel("to_network")
from_network_channel = love.thread.getChannel("from_network")

-- TODO: what state needs to be fetched? ->
-- blind selection, get:
-- blind: skip as a dictionary

-- entering playing stage
-- jokers, cards in hand, other stuff

-- entering shop stage:
-- G.shop_vouchers.cards
-- G.shop_booster.cards
-- G.shop_jokers.cards
-- current cash
-- reroll cost
function get_current_game_state()
    -- print("chips")
    -- print(inspect(G.GAME.blind))
    -- print("G.play")
    -- print(inspect(G.play.cards))
    local jokers = {}
    if G.jokers and G.jokers.cards then
        jokers = G.jokers
        print(inspect(G.jokers.cards))
    end
    local current_chips = G.GAME.chips
    local blind_chips = G.GAME.blind.chips 
    local blind_mult = G.GAME.blind.mult
    local blind_dollars = G.GAME.blind.dollars
    local hands_left = G.GAME.current_round.hands_left
    local discards_left = G.GAME.current_round.discards_left
    print(inspect(G.hand.cards[1]))


    print("Current Chips: " .. tostring(current_chips))
    print("Blind Chips: " .. tostring(blind_chips))
    print("Blind Multiplier: " .. tostring(blind_mult))
    print("Blind Dollars: " .. tostring(blind_dollars))
    print("Hands left: " .. tostring(hands_left))
    print("Discards left: " .. tostring(discards_left))
    print("Jokers: " .. tostring(jokers))
    return {
        score = 100,
        cards = { "Ace", "King" }
    }
end


-- actions:

-- blind selection:
-- view blinds,
-- select or skipping them

-- playing section:
-- current score, 
-- select -> use tarot / spectral OR play or discard cards
-- 
function apply_action(action)
    print("Applying action: " .. action)
end

local sr = Game.start_run
function Game:start_run(args)
    local ret = sr(self, args)
    local game_state = get_current_game_state()
    local state_str = serialize_table(game_state)

    to_network_channel:push({ type = "send", data = state_str })

    local action = from_network_channel:pop()
    if action then
        apply_action(action)
    end
    return ret
end

-- is function hooking that easy? the fuck lmao
-- https://github.com/MathIsFun0/Cryptid/blob/main/lib/overrides.lua#L132
-- https://forums.kleientertainment.com/forums/topic/129557-tutorial-function-hooking-and-you/
local gigo = Game.init_game_object
function Game:init_game_object()
	local g = gigo(self)

    -- local game_state = get_current_game_state()
    -- local state_str = serialize_table(game_state)

    -- to_network_channel:push({ type = "send", data = state_str })

    -- local action = from_network_channel:pop()
    -- if action then
    --     apply_action(action)
    -- end
    print("balatro_nn inited...")
	return g
end


-- local upd = Game.update
-- function Game:update(dt)
--     local ret = upd(self, dt)
    
--     return ret
-- end


local dft = Blind.defeat
function Blind:defeat(s)
    dft(self, s)
    print("meowwww blind defeated.......")
end

local disable = Blind.disable
function Blind:disable()
    disable(self)
    play_sound('glass'..math.random(1, 6), math.random()*0.2 + 0.9,0.5)
end

local cashout = G.FUNCS.cash_out
G.FUNCS.cash_out = function(e)
    local ret = cashout(e)
    print("Cash out!!!! :3:3:3")
    return ret
end


local pcfh = G.FUNCS.play_cards_from_highlighted
G.FUNCS.play_cards_from_highlighted = function(e)
    local ret = pcfh(e)
    local game_state = get_current_game_state()
    local state_str = serialize_table(game_state)

    to_network_channel:push({ type = "send", data = state_str })

    local action = from_network_channel:pop()
    if action then
        apply_action(action)
    end

    print("meowwwww playing card :3:3")
    return ret
end


local dcfh = G.FUNCS.discard_cards_from_highlighted
G.FUNCS.discard_cards_from_highlighted = function(e)
    local ret = dcfh(e)
    local game_state = get_current_game_state()
    local state_str = serialize_table(game_state)

    to_network_channel:push({ type = "send", data = state_str })

    local action = from_network_channel:pop()
    if action then
        apply_action(action)
    end

    print("meowwwww discarding card :3:3")
    return ret
end

local eval = G.FUNCS.evaluate_play
G.FUNCS.evaluate_play = function(e)
    print("meowww evaluating score :3:3")
    local ret = eval(e)
    local game_state = get_current_game_state()
    local state_str = serialize_table(game_state)

    to_network_channel:push({ type = "send", data = state_str })

    local action = from_network_channel:pop()
    return ret
end
