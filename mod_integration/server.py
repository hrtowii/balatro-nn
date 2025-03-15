import socket
import re
import luadata
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass
class Edition:
    name: str = None  # Holographic, Negative, Polychrome, Foil

@dataclass
class Seal:
    color: str = None  # Red, Gold, Purple

@dataclass
class Card:
    suit: str = None
    rank: str = None
    chips: int = None
    edition: Optional[Edition] = None
    seal: Optional[Seal] = None
    
    def __post_init__(self):
        if isinstance(self.edition, dict):
            self.edition = Edition(**self.edition)
        if isinstance(self.seal, dict):
            self.seal = Seal(**self.seal)


@dataclass
class Blind:
    chips: int = 0
    mult: float = 0
    dollars: int = 0

@dataclass
class Round:
    hands_left: int = 0
    discards_left: int = 0

@dataclass
class Joker:
    ability: Any = None
    blueprint: bool = None
    name: str = None
    extra: Any = None # config.extra

@dataclass
class Shop:
    vouchers: List[Any] = None
    boosters: List[Any] = None
    jokers: List[Any] = None
    cash: int = 0
    reroll_cost: int = 0

@dataclass
class GameState:
    score: int = 0
    blind: Optional[Blind] = None
    round: Optional[Round] = None
    jokers: List[Joker] = None
    cards: List[Card] = None
    shop: Optional[Shop] = None
    
    def __post_init__(self):
        if self.jokers is None:
            self.jokers = []
        if self.cards is None:
            self.cards = []
            
        if isinstance(self.blind, dict):
            self.blind = Blind(**self.blind)
        
        if isinstance(self.round, dict):
            self.round = Round(**self.round)
            
        if isinstance(self.shop, dict):
            self.shop = Shop(**self.shop)
            
        if self.jokers and isinstance(self.jokers, list):
            self.jokers = [Joker(**j) if isinstance(j, dict) else j for j in self.jokers]
            
        # Convert cards list of dicts to list of Card objects
        if self.cards and isinstance(self.cards, list):
            self.cards = [Card(**c) if isinstance(c, dict) else c for c in self.cards]
    
    def determine_action(self) -> str:
        """Logic to determine the next action based on game state"""
        if self.score > 50:
            return "play_card"
        return "skip"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 12345))  # Localhost, port 12345
server.listen(1)
print("Waiting for connection...")
conn, addr = server.accept()
print("Connected by", addr)

while True:
    data = ""
    while True:
        chunk = conn.recv(4096).decode()
        if not chunk:
            break
        data += chunk
        if "\n" in chunk:
            break
    data = data.strip()
    if data:
        fixed_data = re.sub(r'(\s*)(\d+)\s*=', r'\1[\2] =', data)
        
        try:
            print(fixed_data)
            game_state_dict = luadata.unserialize(fixed_data)
            print("Raw game state dict:", game_state_dict)
            
            game_state = GameState(**game_state_dict)
            print("Parsed game state:", game_state)
            
            action = game_state.determine_action()
            conn.send((action + "\n").encode())
            
        except Exception as e:
            print("Error processing data:", e)
            conn.send("skip\n".encode())