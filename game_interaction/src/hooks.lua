-- is function hooking that easy? the fuck lmao
-- https://github.com/MathIsFun0/Cryptid/blob/main/lib/overrides.lua#L132
-- https://forums.kleientertainment.com/forums/topic/129557-tutorial-function-hooking-and-you/
local sr = Game.start_run
function Game:start_run(args)
    local ret = sr(self, args)
    B_NN:update()
    return ret
end

-- we hook this to get the blind tags
local blind_tag = create_UIBox_blind_tag
function create_UIBox_blind_tag(blind_choice, run_info)
    local ret = blind_tag(blind_choice, run_info)
    if G and G.GAME then
        local _tag = Tag(G.GAME.round_resets.blind_tags[blind_choice], nil, blind_choice)
        local shortened_tag = {
            name = _tag.name,
            ability = _tag.ability,
        }
        print("tag tag tag :3:3")
        local key = string.lower(blind_choice)
        B_NN.tags[key] = shortened_tag
        -- print(inspect(_tag.ability))
        B_NN:update()
    end
    return ret
end

local skip_blind = G.FUNCS.skip_blind
G.FUNCS.skip_blind = function(e)
    local ret = skip_blind(e)
    print("SKIPPED BLIND!!! omg")
    B_NN:update()
    return ret
end

local select_blind = G.FUNCS.select_blind
G.FUNCS.select_blind = function(e)
    local ret = select_blind(e)
    print("SELECTED BLIND!!! :3")
    B_NN:update()
    return ret
end

local gigo = Game.init_game_object
function Game:init_game_object()
	local g = gigo(self)
    print("balatro_nn inited...")
	return g
end

local dft = Blind.defeat
function Blind:defeat(s)
    local ret = dft(self, s)
    print("meowwww blind defeated.......")
    B_NN:update()
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
    B_NN:update()
    return ret
end

local accumulated_dt = 0
local update_interval = 1
local update_shop = Game.update_shop
function Game:update_shop(dt)
    local ret = update_shop(self, dt)
    
    accumulated_dt = accumulated_dt + dt
    if accumulated_dt >= update_interval then
        B_NN:update()
        -- Set all joker costs to 0
        if G.shop_jokers and G.shop_jokers.cards then
            for i, card in ipairs(G.shop_jokers.cards) do
            card.cost = 0
            end
        end
        
        -- Set all booster costs to 0
        if G.shop_booster and G.shop_booster.cards then
            for i, card in ipairs(G.shop_booster.cards) do
            card.cost = 0
            end
        end
        
        -- Set all voucher costs to 0
        if G.shop_vouchers and G.shop_vouchers.cards then
            for i, card in ipairs(G.shop_vouchers.cards) do
            card.cost = 0
            end
        end
        accumulated_dt = accumulated_dt - update_interval
    end
    
    return ret
end

local pcfh = G.FUNCS.play_cards_from_highlighted
G.FUNCS.play_cards_from_highlighted = function(e)
    local ret = pcfh(e)
    B_NN:update()
    print("meowwwww playing card :3:3")
    return ret
end

local dcfh = G.FUNCS.discard_cards_from_highlighted
G.FUNCS.discard_cards_from_highlighted = function(e)
    local ret = dcfh(e)
    B_NN:update()
    print("meowwwww discarding card :3:3")
    return ret
end

local eval = G.FUNCS.evaluate_play
G.FUNCS.evaluate_play = function(e)
    print("meowww evaluating score :3:3")
    local ret = eval(e)
    B_NN:update()
    return ret
end
