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
-- IN EVERY STAGE:
-- use / sell consumeables (sic.)

-- blind selection, get:
-- blind: skip tag as a dictionary
-- create a skip tag

-- entering playing stage
-- jokers, cards in hand, other stuff

-- entering shop stage:
-- G.shop_vouchers.cards
-- G.shop_booster.cards
-- G.shop_jokers.cards

-- current cash
-- reroll cost

-- https://github.com/besteon/balatrobot/blob/main/src/utils.lua
game_state = {}
function get_current_game_state()
    -- BLIND selection -> get antes
    game_state.ante = G.GAME.round_resets.ante or 0
    game_state.blind_choices = G.GAME.round_resets.blind_choices
    -- print(inspectDepth(G.P_BLINDS[game_state.blind_choices.Boss]))
    game_state.blind_info = {
        small = {
            dollars = G.P_BLINDS[game_state.blind_choices.Small].dollars,
            defeated = G.P_BLINDS[game_state.blind_choices.Small].defeated,
            name = G.P_BLINDS[game_state.blind_choices.Small].name,
            debuff_text = G.P_BLINDS[game_state.blind_choices.Small].debuff_text,
            debuff = G.P_BLINDS[game_state.blind_choices.Small].debuff,
            tag = B_NN.tags and B_NN.tags.small or nil
        },
        big = {
            dollars = G.P_BLINDS[game_state.blind_choices.Big].dollars,
            defeated = G.P_BLINDS[game_state.blind_choices.Big].defeated,
            name = G.P_BLINDS[game_state.blind_choices.Big].name,
            debuff_text = G.P_BLINDS[game_state.blind_choices.Big].debuff_text,
            debuff = G.P_BLINDS[game_state.blind_choices.Big].debuff,
            tag = B_NN.tags and B_NN.tags.big or nil
        },
        boss = {
            dollars = G.P_BLINDS[game_state.blind_choices.Boss].dollars,
            defeated = G.P_BLINDS[game_state.blind_choices.Boss].defeated,
            name = G.P_BLINDS[game_state.blind_choices.Boss].name,
            debuff_text = G.P_BLINDS[game_state.blind_choices.Boss].debuff_text,
            debuff = G.P_BLINDS[game_state.blind_choices.Boss].debuff
        }
    }
    
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
        -- G.GAME.current_round.hands_left = 1000
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
            -- if not joker.edition then
            --     joker:set_edition({negative = true}, true, true)
            -- end
            -- print(inspectDepth(joker.config))
            local edition = {}
            if joker.edition then
                edition = joker.edition.type
            end
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
                edition = edition,
                extra = joker.config.center.config.extra or {} -- can either be number or {extra = {Xmult = 4, every = 5, remaining = "5 remaining"} or a whole fucking mess jfc
            })
        end
    end

    game_state.cards = {}
    if G.hand and G.hand.cards then
        for i, card in ipairs(G.hand.cards) do
            if card then
                local chips = card:get_chip_bonus()
                -- print("chips for card: ".. tostring(chips))
                table.insert(game_state.cards, {
                    suit = card.base.suit,
                    rank = card.rank,
                    chips = chips,
                    seal = card.seal or {},
                    edition = card.edition or {}
                })
            end
        end
    end

    game_state.shop = {}
    if G.GAME and G.shop then
        -- https://github.com/besteon/balatrobot/blob/main/src/utils.lua#L84 -> their card fetching data is really useless, i gotta research myself
        -- TODO: tarot packs / cards, spectral packs / cards, playing cards
        game_state.shop.reroll_cost = G.GAME.current_round.reroll_cost
        game_state.shop.jokers = { }
        game_state.shop.boosters = { }
        game_state.shop.vouchers = { }

        if G.shop_jokers and G.shop_jokers.cards and type(G.shop_jokers.cards) == "table" then
            for i, card in ipairs(G.shop_jokers.cards) do
                local center = card.config.center or {}
                game_state.shop.jokers[i] = {
                    name = card.label,
                    config = {
                        set = center.set,
                        rarity = center.rarity,
                        effect = center.effect,
                        abilities = center.config or {},
                        blueprint_compat = center.blueprint_compat,
                    },
                    cost = card.cost or 0
                }
            end
        end

        if G.shop_booster and G.shop_booster.cards and type(G.shop_booster.cards) == "table"then
                for i, card in ipairs(G.shop_booster.cards) do
                    local center = card.config.center
                    game_state.shop.boosters[i] = {
                        name = card.label or center.name,
                        config = center.config or nil,
                        cost = card.cost or center.cost or 0,
                    }
                end
        end

        if G.shop_vouchers and G.shop_vouchers.cards and type(G.shop_vouchers.cards) == "table" then
                for i, card in ipairs(G.shop_vouchers.cards) do
                    local center = card.config.center
                    game_state.shop.vouchers[i] = {
                        name = card.label,
                        config = center.config or nil,
                        cost = card.cost or center.cost or 0,
                    }
                end
        end
    end
    -- print("game_state:" .. inspectDepth(game_state))
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

function B_NN:update()
    local game_state = get_current_game_state()
    local state_str = serialize_table(game_state)

    to_network_channel:push({ type = "send", data = state_str })

    -- local action = from_network_channel:pop()
    -- if action then
    --     apply_action(action)
    -- end
end