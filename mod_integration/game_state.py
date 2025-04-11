import socket
import re
import luadata
from dataclasses import dataclass, field
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
    value: int = None
    name: str = None
    label: str = None
    card_key: str = None

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
class BlindInfo:
    small: Dict[str, Any] = field(default_factory=dict)
    big: Dict[str, Any] = field(default_factory=dict)
    boss: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Round:
    hands_left: int = 0
    discards_left: int = 0


@dataclass
class Joker:
    ability: Any = None
    blueprint: bool = None
    name: str = None
    label: str = None
    extra: Any = None  # config.center.extra
    edition: Optional[Edition] = None

    def __post_init__(self):
        if isinstance(self.edition, dict):
            self.edition = Edition(**self.edition)


@dataclass
class Shop:
    cards: List[Any] = None
    vouchers: List[Any] = None
    boosters: List[Any] = None
    jokers: List[Any] = None
    cash: int = 0
    reroll_cost: int = 0


@dataclass
class GameState:
    score: int = 0
    ante: int = 0
    round: int = ante * 3
    blind_choices: Dict[str, Any] = field(default_factory=dict)
    blind_info: Optional[BlindInfo] = None
    blind: Optional[Blind] = None
    current_round: Optional[Round] = None
    jokers: List[Joker] = None
    cards: List[Card] = None
    hand: List[Card] = None
    shop: Optional[Shop] = None
    state: Any = None
    waitingFor: Any = None
    waitingForAction: Any = None

    def __post_init__(self):
        if self.jokers is None:
            self.jokers = []
        if self.cards is None:
            self.cards = []
        if self.hand is None:
            self.hand = []
        if isinstance(self.blind_info, dict):
            self.blind_info = BlindInfo(**self.blind_info)

        if isinstance(self.blind, dict):
            self.blind = Blind(**self.blind)

        if isinstance(self.current_round, dict):
            self.current_round = Round(**self.current_round)

        if isinstance(self.shop, dict):
            self.shop = Shop(**self.shop)

        if self.jokers and isinstance(self.jokers, list):
            self.jokers = [
                Joker(**j) if isinstance(j, dict) else j for j in self.jokers
            ]

        if self.cards and isinstance(self.cards, list):
            self.cards = [Card(**c) if isinstance(c, dict) else c for c in self.cards]

        if self.hand and isinstance(self.hand, list):
            self.hand = [Card(**c) if isinstance(c, dict) else c for c in self.hand]
