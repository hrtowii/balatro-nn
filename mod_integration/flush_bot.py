from bot import Bot, Actions
from gamestates import cache_state
import time

import logging

# Set up logging for shop interactions and general debugging
logging.basicConfig(
    filename="flush_bot_log.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Tracks previously attempted purchases to avoid repeated buys
attempted_purchases = set()


# Plays flushes if possible
# otherwise keeps the most common suit
# Discarding the rest, or playing the rest if there are no discards left
class FlushBot(Bot):

    def skip_or_select_blind(self, G):
        cache_state("skip_or_select_blind", G)
        return [Actions.SELECT_BLIND]

    def select_cards_from_hand(self, G):
        logging.debug("Entering select_cards_from_hand")
        try:
            action = self._select_cards_from_hand(G)
            logging.debug(f"Action taken: {action}")
            return action
        except Exception as e:
            logging.error(f"Error in select_cards_from_hand: {e}")
            return [Actions.PLAY_HAND, [1]]  # Fallback action

    def _select_cards_from_hand(self, G):
        try:
            logging.debug(f"Game state before selecting cards: {G}")

            if "hand" not in G or not G["hand"]:
                logging.error("Hand is empty or missing from game state!")
                return [Actions.PLAY_HAND, [1]]  # Fallback

            suit_count = {suit: 0 for suit in ["Hearts", "Diamonds", "Clubs", "Spades"]}
            for card in G["hand"]:
                suit_count[card["suit"]] += 1

            most_common_suit = max(suit_count, key=suit_count.get)
            most_common_suit_count = suit_count[most_common_suit]

            if most_common_suit_count >= 5:
                flush_cards = [card for card in G["hand"] if card["suit"] == most_common_suit]
                flush_cards.sort(key=lambda x: x["value"], reverse=True)
                indices = [G["hand"].index(card) + 1 for card in flush_cards[:5]]
                action = [Actions.PLAY_HAND, indices]

                if any(i > len(G["hand"]) for i in indices):
                    logging.error(f"Invalid indices in PLAY_HAND action: {indices}")
                    return [Actions.PLAY_HAND, [1]]  # Fallback
                
                logging.debug(f"Playing flush: {action}")
                return action

            # Discarding non-suited cards
            discards = [card for card in G["hand"] if card["suit"] != most_common_suit]
            discards.sort(key=lambda x: x["value"], reverse=True)
            discards = discards[:5]
            indices = [G["hand"].index(card) + 1 for card in discards]

            if len(discards) > 0:
                action_type = Actions.DISCARD_HAND if G["current_round"]["discards_left"] > 0 else Actions.PLAY_HAND
                action = [action_type, indices]

                if any(i > len(G["hand"]) for i in indices):
                    logging.error(f"Invalid indices in DISCARD_HAND action: {indices}")
                    return [Actions.PLAY_HAND, [1]]  # Fallback
                
                logging.debug(f"Discarding non-suited cards: {action}")
                return action

            logging.warning("No valid flush or discard, playing first card by default.")
            return [Actions.PLAY_HAND, [1]]

        except Exception as e:
            logging.error(f"Exception in _select_cards_from_hand: {e}")
            return [Actions.PLAY_HAND, [1]]  # Fallback action

    def select_shop_action(self, G):
        global attempted_purchases, attempted_rerolls
        logging.info(f"Shop state received: {G}")

        # End shop action instantly if the number of jokers held is 5 or more
        if len(G.get("jokers", [])) >= 5:
            logging.info("5 or more jokers held, ending shop action instantly.")
            return [Actions.END_SHOP]

        if not hasattr(self, "rerolled_once"):
            self.rerolled_once = False  # Track if we've rerolled already

        specific_joker_cards = {
        "Joker", "Greedy Joker", "Lusty Joker", "Wrathful Joker", "Gluttonous Joker",
        "Droll Joker", "Crafty Joker", "Joker Stencil", "Banner", "Mystic Summit",
        "Loyalty Card", "Misprint", "Raised Fist", "Fibonacci", "Scary Face",
        "Abstract Joker", "Pareidolia", "Gros Michel", "Even Steven", "Odd Todd",
        "Scholar", "Supernova", "Burglar", "Blackboard", "Ice Cream", "Hiker",
        "Green Joker", "Cavendish", "Card Sharp", "Red Card",
        "Baron", "Midas Mask", "Photograph", "Erosion", "Baseball Card", "Bull",
        "Popcorn", "Ancient Joker", "Ramen", "Walkie Talkie", "Seltzer", "Castle",
        "Smiley Face", "Acrobat", "Sock and Buskin", "Swashbuckler", "Bloodstone",
        "Arrowhead", "Onyx Agate", "Showman", "Flower Pot", "Blueprint", "Wee Joker",
        "Merry Andy", "The Idol", "Seeing Double", "Hit the Road", "The Tribe",
        "Stuntman", "Brainstorm", "Shoot the Moon", "Bootstraps", "Triboulet",
        "Yorik", "Chicot"
        }

        if "shop" in G and "dollars" in G:
            dollars = G["dollars"]
            shop_cards = G["shop"].get("cards", [])

            logging.info(f"Current dollars: {dollars}, Available cards: {shop_cards}")

            # Try to buy a Joker if available
            for i, card in enumerate(shop_cards):
                if card["label"] in specific_joker_cards and card["label"] not in attempted_purchases and len(G.get("jokers", [])) < 5:
                    logging.info(f"Attempting to buy specific Joker: {card['label']}")
                    attempted_purchases.add(card["label"])
                    self.rerolled_once = True
                    return [Actions.BUY_CARD, [i + 1]]

            # If no Joker was found and we haven't rerolled yet, attempt a reroll
            if not self.rerolled_once:
                logging.info("No Jokers found in initial shop, rerolling once.")
                self.rerolled_once = True
                return [Actions.REROLL_SHOP]

        # If no purchase was made and reroll has already happened, end shop interaction
        logging.info("No valid purchase or reroll available, ending shop interaction.")
        if hasattr(self, "rerolled_once"):
            delattr(self, "rerolled_once")  # Removes the attribute from the object

        return [Actions.END_SHOP]


    def select_booster_action(self, G):
        return [Actions.SKIP_BOOSTER_PACK]

    def sell_jokers(self, G):
        # Jokers that should NOT be sold
        jokers_to_keep = {
        "Droll Joker", "Crafty Joker", "The Tribe", "Blueprint", "Brainstorm",
        "Cavendish", "Canio", "Hiker", "Blackboard", "Card Sharp", "Abstract Joker"
        }

        # Check the list of jokers sequentially
        if len(G["jokers"]) > 4:
            for i, joker in enumerate(G.get("jokers", []), start=1):
                if joker["label"] not in jokers_to_keep:
                    logging.info(f"Selling joker: {joker['label']} at position {i}")
                    return [Actions.SELL_JOKER, [i]]

            # If no joker was sold, return an empty sell action
            logging.info("No eligible jokers to sell.")
            return [Actions.SELL_JOKER, []]

        # If no joker was sold, return an empty sell action
        logging.info("No eligible jokers to sell.")
        return [Actions.SELL_JOKER, []]

    def rearrange_jokers(self, G):
        return [Actions.REARRANGE_JOKERS, []]

    def use_or_sell_consumables(self, G):
        return [Actions.USE_CONSUMABLE, []]

    def rearrange_consumables(self, G):
        return [Actions.REARRANGE_CONSUMABLES, []]

    def rearrange_hand(self, G):
        return [Actions.REARRANGE_HAND, []]


def benchmark_multi_instance():
    global t
    t = 0
    global first_time
    first_time = None

    # Benchmark the game states per second for different bot counts
    bot_counts = [1] # range(1, 21, 3)
    for bot_count in bot_counts:
        target_t = 50 * bot_count
        t = 0
        first_time = None

        bots = []
        for i in range(bot_count):
            mybot = FlushBot(
                deck="Blue Deck",
                stake=1,
                seed=None,
                challenge=None,
                bot_port=12348 + i,
            )

            bots.append(mybot)

        try:
            for bot in bots:
                bot.start_balatro_instance()
            time.sleep(20)

            start_time = time.time()
            while t < target_t:
                for bot in bots:
                    bot.run_step()
            end_time = time.time()

            t_per_sec = target_t / (end_time - start_time)
            print(f"Bot count: {bot_count}, t/sec: {t_per_sec}")
        finally:
            # Stop the bots
            for bot in bots:
                bot.stop_balatro_instance()


if __name__ == "__main__":
    benchmark_multi_instance()