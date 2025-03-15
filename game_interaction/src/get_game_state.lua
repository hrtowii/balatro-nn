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
    print(tmp)
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

-- https://github.com/besteon/balatrobot/blob/main/src/utils.lua
function get_current_game_state()
    local game_state = {}
    -- BLIND selection -> get antes


    game_state.score = G.GAME.chips or 0

    -- CURRENT blind -> ingame
    if G.GAME and G.GAME.blind then
        game_state.blind = {
            chips = G.GAME.blind.chips or 0,
            mult = G.GAME.blind.mult or 0,
            dollars = G.GAME.blind.dollars or 0
        }
    end

    if G.GAME and G.GAME.current_round then
        game_state.round = {
            hands_left = G.GAME.current_round.hands_left or 0,
            discards_left = G.GAME.current_round.discards_left or 0
        }
    end

    game_state.jokers = {}
    if G.jokers and G.jokers.cards then
        for i, joker in ipairs(G.jokers.cards) do
            local chips_value = joker.ability.chips or 0
            if type(joker.ability.extra) == "number" then
                chips_value = joker.ability.extra
            end
            -- print(inspectDepth(joker.config))
            table.insert(game_state.jokers, {
                name = joker.label,
                blueprint = joker.config.center.blueprint_compat,
                ability = {
                    chips = chips_value,
                    x_mult = joker.ability.caino_xmult or joker.ability.x_mult or 0,
                    x_chips = joker.ability.x_chips or 0,
                    mult = joker.ability.mult or 0,
                    eternal = joker.ability.eternal or false
                },
                extra = joker.config.center.config.extra or {} -- can either be number or {extra = {Xmult = 4, every = 5, remaining = "5 remaining"} or a whole fucking mess jfc
            })
        end
    end

    game_state.cards = {}
    if G.hand and G.hand.cards then
        for i, card in ipairs(G.hand.cards) do
            local edition = {}
            local seal = {}
            if card then
                if card.edition then
                    edition = card.edition.type
                end
                if card.seal then
                    seal = card.seal
                end
                local chips = card:get_chip_bonus()
                -- print("chips for card: ".. tostring(chips))
                table.insert(game_state.cards, {
                    suit = card.base.suit,
                    rank = card.rank,
                    chips = chips,
                    seal = seal,
                    edition = edition
                })
            end
        end
    end

    game_state.shop = {}
    if G.GAME and G.shop then
        -- https://github.com/besteon/balatrobot/blob/main/src/utils.lua#L84 -> their card fetching data is really useless, i gotta research myself
        -- TODO: tarot packs / cards, spectral packs / cards, playing cards
        game_state.shop.reroll_cost = G.GAME.current_round.reroll_cost
        game_state.shop.cards = { }
        game_state.shop.boosters = { }
        game_state.shop.vouchers = { }

        for i = 1, #G.shop_jokers.cards do
            game_state.shop.cards[i] = G.shop_jokers.cards[i]
        end

        for i = 1, #G.shop_booster.cards do
            game_state.shop.boosters[i] = G.shop_booster.cards[i]
        end

        for i = 1, #G.shop_vouchers.cards do
            game_state.shop.vouchers[i] = G.shop_vouchers.cards[i]
        end
        print(inspectDepth(G.shop_booster))
    end
    print("game_state:" .. inspectDepth(game_state))
    return game_state
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
    local ret = dft(self, s)
    print("meowwww blind defeated.......")
    local game_state = get_current_game_state()
    local state_str = serialize_table(game_state)

    to_network_channel:push({ type = "send", data = state_str })

    local action = from_network_channel:pop()
    if action then
        apply_action(action)
    end
    return ret
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
    local game_state = get_current_game_state()
    local state_str = serialize_table(game_state)

    to_network_channel:push({ type = "send", data = state_str })

    local action = from_network_channel:pop()
    if action then
        apply_action(action)
    end
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
    print(e)
    local game_state = get_current_game_state()
    local state_str = serialize_table(game_state)

    to_network_channel:push({ type = "send", data = state_str })

    local action = from_network_channel:pop()
    return ret
end
