import os
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.spaces import Tuple, Discrete, MultiDiscrete
from gym.error import DependencyNotInstalled

import pygame


def get_suit_and_num(card_num):
    if card_num == 0 or card_num == 53:
        return [0, card_num]

    suit = 0
    num = card_num % 13
    if num == 0:
        num = 13

    if card_num / 13 <= 1:
        suit = 0
    elif card_num / 26 <= 1:
        suit = 1
    elif card_num / 39 <= 1:
        suit = 2
    else:
        suit = 3

    return [suit, num]

def get_suit_char_from_val(suit_val):
    if suit_val == 0:
        return "H"
    elif suit_val == 1:
        return "D"
    elif suit_val == 2:
        return "S"
    elif suit_val == 3:
        return "C"

def get_card_char(card_num):
    if card_num == 1:
        return "A"
    elif card_num == 10:
        return "T"
    elif card_num == 11:
        return "J"
    elif card_num == 12:
        return "Q"
    elif card_num == 13:
        return "K"
    else:
        return str(int(card_num))

def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def get_shuffled_deck(np_random):
    return np_random.choice(range(1,53), 52, False)

def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]






def deck_to_suit_check(deck_cards_param, suit_cards_param):
    #check if it's not zero
    active_deck_card_index = 2
    active_deck_card = deck_cards_param[0,active_deck_card_index]
    while active_deck_card == 0 and active_deck_card_index >= 0:
        active_deck_card_index = active_deck_card_index - 1
        active_deck_card = deck_cards_param[0,active_deck_card_index]
    if active_deck_card == 0:
        #if it is still zero, then it used all 3, and don't update anything
        #(agent should learn it is a dumb move to do action 1 when
        #deck_cards[0,0] is 0/empty)
        return False, False, False
    else:
        active_deck_card_suit_and_num = get_suit_and_num(active_deck_card)
        #if the suit pile for this card is 1 below value of the card, transact
        if suit_cards_param[active_deck_card_suit_and_num[0]] + 1 == active_deck_card_suit_and_num[1]:
            return True, active_deck_card_suit_and_num[0], active_deck_card_index
    #otherwise, wasn't successful
    return False, False, False




def deck_to_pile_check(deck_cards_param, pile_cards_param, action_i):
    #check if it's not zero
    active_deck_card_index = 2
    active_deck_card = deck_cards_param[0,active_deck_card_index]
    while active_deck_card == 0 and active_deck_card_index >= 0:
        active_deck_card_index = active_deck_card_index - 1
        active_deck_card = deck_cards_param[0,active_deck_card_index]
    
    if active_deck_card == 0:
        #if it is still zero, then it used all 3, and don't update anything
        #(agent should learn it is a dumb move to do action 1 when
        #deck_cards[0,0] is 0/empty)
        return False, False, False, False
    else:
        active_deck_card_suit_and_num = get_suit_and_num(active_deck_card)
        a_suit = active_deck_card_suit_and_num[0]
        a_num = active_deck_card_suit_and_num[1]

        for i in range(13):
            #if on this row check, we are looking at rows less than the card number of the
            #active card in the deck we are trying to move
            if i <= a_num - 1:
                #if it not zero, return because that's an illegal move
                if pile_cards_param[i,action_i] != 0:
                    return False, False, False, False
            elif i == a_num:
                #else check the card above is different suit and 1 higher
                if pile_cards_param[i,action_i] > 0:
                    focus_pile_card_suit_and_num = get_suit_and_num(pile_cards_param[i,action_i])
                    f_suit = focus_pile_card_suit_and_num[0]
                    f_num = focus_pile_card_suit_and_num[1]
                    if ((a_suit <= 1 and f_suit >= 2) or (a_suit >= 2 and f_suit <= 1)) and int(f_num - 1) == int(a_num):
                        return True, (i-1), active_deck_card, active_deck_card_index
        #edge case, check if it was a king, and we already went through    
        if a_num == 13 and i == 12:
            return True, i, active_deck_card, active_deck_card_index
    return False, False, False, False



def suit_to_pile_check(suit_cards_param, pile_cards_param, pile_i, suit_j):
    #check if it's not zero
    active_suit_card = suit_cards_param[suit_j]
    if active_suit_card == 0:
        #if it is still zero, then it used all 3, and don't update anything
        #(agent should learn it is a dumb move to do action 1 when
        #deck_cards[0,0] is 0/empty)
        return False, False, False
    else:
        #get actual card value by multiply by 13*suit_j to get actual card value,
        #instead of just number
        #i.e. ace of hearts is still 1, ace of spades is 27, 3 of diamonds is 16
        active_suit_card = int(suit_cards_param[suit_j] + (13*suit_j))
        active_suit_card_suit_and_num = get_suit_and_num(active_suit_card)
        a_suit = active_suit_card_suit_and_num[0]
        a_num = active_suit_card_suit_and_num[1]

        for i in range(13):
            #if on this row check, we are looking at rows less than the card number of the
            #active card in the deck we are trying to move
            if i <= a_num - 1:
                #if it not zero, return because that's an illegal move
                if pile_cards_param[i,pile_i] != 0:
                    return False, False, False
            elif i == a_num:
                #else check the card above is different suit and 1 higher
                if pile_cards_param[i,pile_i] > 0:
                    focus_pile_card_suit_and_num = get_suit_and_num(pile_cards_param[i,pile_i])
                    f_suit = focus_pile_card_suit_and_num[0]
                    f_num = focus_pile_card_suit_and_num[1]
                    if ((a_suit <= 1 and f_suit >= 2) or (a_suit >= 2 and f_suit <= 1)) and int(f_num - 1) == int(a_num):
                        #multiply the card by suit to get the actual id (i.e. 1-52 value)
                        #active_suit_card_id = int(active_suit_card * (1 + suit_j))
                        return True, (i-1), active_suit_card
            
        if a_num == 13 and i == 12:
            #king edge case, we already checked all were zeros below
            #update pile_cards
            return True, i, active_suit_card
    return False, False, False


def pile_to_suit_check(pile_cards_param, suit_cards_param, pile_i):
    active_pile_card_index = 0
    #active pile card in this case is the bottom of the face up pile
    active_pile_card = pile_cards_param[active_pile_card_index,pile_i]
    #so check bottom, (could be ace), and work way through rows until find
    #a non-empty card
    while active_pile_card == 0 and active_pile_card_index < 12:
        active_pile_card_index += 1
        active_pile_card = pile_cards_param[active_pile_card_index,pile_i]
    #if it is still empty at the end of checking all the rows, this is illegal move
    if active_pile_card == 0:
        #if it is still zero, then it used all 3, and don't update anything
        #(agent should learn it is a dumb move to do action 1 when
        #deck_cards[0,0] is 0/empty)
        return False, False, False, False
    else:
        active_pile_card_suit_and_num = get_suit_and_num(active_pile_card)
        a_suit = active_pile_card_suit_and_num[0]
        a_num = active_pile_card_suit_and_num[1]
        #if this is a possible move
        if a_num == suit_cards_param[a_suit] + 1:
            #return possible, card num of bottom-most pile card, index of bottom-most pile card
            return True, a_suit, a_num, int(a_num-1)
    return False, False, False, False



def pile_to_pile_check(pile_cards_param, pile_i, pile_to_move_to_j, card_k):
    #check that this card exists
    if pile_cards_param[card_k,pile_i] > 0:
        t_suit = get_suit_and_num(pile_cards_param[card_k,pile_i])[0]

        #check that we aren't trying to move a king (index), cause that cause out of bounds
        if card_k == 12:
            #check that pile is empty
            if not pile_cards_param[:,pile_to_move_to_j].any():
                return True
        else:
            #check that card+1 in pile to move to is indeed there
            if pile_cards_param[card_k+1,pile_to_move_to_j] > 0:
                f_suit = get_suit_and_num(pile_cards_param[card_k+1,pile_to_move_to_j])[0]
                #check that it is opposite suit
                if (t_suit <= 1 and f_suit >= 2) or (t_suit >= 2 and f_suit <= 1):
                    #and every other one below is empty
                    #check that cards below in the pile to move are 0
                    for i in range(card_k+1):
                        #check it is not illegal
                        if pile_cards_param[i,pile_to_move_to_j] != 0:
                            return False
                    return True

    return False



class SolitaireWorldEnv(gym.Env):
    """
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.

    ### Description
    Card Values:

    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-9) have a value equal to their number.

    This game is played with an infinite deck (or with replacement).
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards.

    The player can request additional cards (hit, action=1) until they decide to stop (stick, action=0)
    or exceed 21 (bust, immediate loss).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust, the player wins.
    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.

    ### Action Space
    There are two actions: stick (0), and hit (1).

    ### Observation Space
    The observation consists of a 3-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    and whether the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (http://incompleteideas.net/book/the-book-2nd.html).

    ### Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:

        +1.5 (if <a href="#nat">natural</a> is True)

        +1 (if <a href="#nat">natural</a> is False)

    ### Arguments

    ```
    gym.make('Blackjack-v1', natural=False, sab=False)
    ```

    <a id="nat">`natural=False`</a>: Whether to give an additional reward for
    starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

    <a id="sab">`sab=False`</a>: Whether to follow the exact rules outlined in the book by
    Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
    If the player achieves a natural blackjack and the dealer does not, the player
    will win (i.e. get a reward of +1). The reverse rule does not apply.
    If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).

    ### Version History
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False):
        self.render_mode = render_mode

        #Remember, obs are only thing agent can see. Other self... things can
        #exist but not be in here, to manage underlying but unknown-to-agent state

        #8 decks to flip thru, 3 cards in a deck, can be 53 (when it initializes and
        #agent hasn't flipped through, but once it has just becomes 0-52 I guess
        decks_obs_shape = np.zeros((8,3), dtype=np.int8)
        decks_obs_shape.fill(54)
        #13 possible cards in a pile, 7 piles, 0 unknown, 1-52 card
        piles_obs_shape = np.zeros((13,7), dtype=np.int8)
        piles_obs_shape.fill(53)
        #piles behind has 7 piles, with as many as 6 cards behind
        piles_behind_obs_shape = np.zeros((6,7), dtype=np.int8)
        piles_behind_obs_shape.fill(54)

        piles_behind_actual_obs_shape = np.zeros((6,7), dtype=np.int8)
        piles_behind_actual_obs_shape.fill(53)

        self.action_space = Discrete(548)
        self.observation_space = spaces.Dict({
            "deck_position": Discrete(8),
            "decks_actual": MultiDiscrete(decks_obs_shape), #for testing
            "decks": MultiDiscrete(decks_obs_shape),
            "suits": MultiDiscrete(np.array([14,14,14,14])),
            "piles": MultiDiscrete(piles_obs_shape),
            "piles_behind": MultiDiscrete(piles_behind_obs_shape), #6 possible cards behind
            "piles_behind_actual": MultiDiscrete(piles_behind_actual_obs_shape) #for testing
        })


        self.render_mode = render_mode

    def action_mask(self, deck_cards_p, suit_cards_p, pile_cards_p):
        mask = np.zeros(548, dtype=np.int8)
        #can always tap the deck
        mask[0] = 1

        for i in range(1,548):
            if i == 1:
                can_deck_to_suit, _, _ = deck_to_suit_check(deck_cards_p, suit_cards_p)
                if can_deck_to_suit:
                    mask[i] = 1
                    # print("Can deck to suit, action = ", 1)
            elif i >= 2 and i <= 8:
                a_i = i - 2
                can_deck_to_pile, _, _, _ = deck_to_pile_check(deck_cards_p, pile_cards_p, a_i)
                if can_deck_to_pile:
                    mask[i] = 1
                    # print("can deck to pile: pile: ", a_i)
            elif i >= 9 and i <= 36:
                p_i = (i - 9) % 7
                s_j = (i - 9) // 7
                can_suit_to_pile, _, _ = suit_to_pile_check(suit_cards_p, pile_cards_p, p_i, s_j)
                if can_suit_to_pile:
                    mask[i] = 1
                    # print("can suit to pile: pile i: ", p_i)
                    # print("and s_j: ", s_j)
            elif i >= 37 and i <= 43:
                p_i = (i - 37)
                can_pile_to_suit, _, _, _ = pile_to_suit_check(pile_cards_p, suit_cards_p, p_i)
                if can_pile_to_suit:
                    mask[i] = 1
                    # print("can pile to suit, p_i: ", p_i)
            elif i >= 44 and i <= 547:
                p_i = (i - 44) // 72
                p_to_m = int(((i - 44 - (72*p_i)) // 12) + 1)
                p_to_m_j = (p_i + p_to_m) % 7
                c_k = ((i - 44) % 12) + 1

                can_pile_to_pile = pile_to_pile_check(pile_cards_p, p_i, p_to_m_j, c_k)

                if can_pile_to_pile:
                    mask[i] = 1
                    # print("can pile to pile, p_i: ", p_i)
                    # print("and pmj: ", p_to_m_j)
                    # print("and c_k: ", c_k)
        return mask



    def step(self, action):
        assert self.action_space.contains(action)

        terminated = False
        #game reward for initial part of step starts at zero, changes depending on action outcome
        game_reward = 0
        #agent reward however is -1 because we want to penalize it for steping, since time is a factor
        reward = -1

        #source: https://en.wikipedia.org/wiki/Klondike_(solitaire) Microsoft Windows Scoring section
        game_score_deck_to_pile = 5
        game_score_deck_to_suit = 10
        game_score_pile_to_suit = 10
        game_score_pile_card_reveal = 5
        game_score_suit_to_pile = -15
        game_score_deck_cycle = -20
        game_score_victory = 10000

        agent_reward_deck_to_pile = 5
        agent_reward_deck_to_suit = 15
        agent_reward_pile_to_suit = 15
        agent_reward_pile_card_reveal = 30
        agent_reward_suit_to_pile = -20
        agent_reward_deck_cycle = -10
        agent_reward_deck_flip = 0
        agent_reward_victory = 10000


        #action 0 is tapping deck
        if action == 0:
            #append first row to end, for underlying and known sets
            self.deck_cards = np.append(self.deck_cards, [self.deck_cards[0,:]], axis=0)
            self.deck_cards_known = np.append(self.deck_cards_known, [self.deck_cards_known[0,:]], axis=0)
            #then delete first row for both
            self.deck_cards = np.delete(self.deck_cards, (0), axis=0)
            self.deck_cards_known = np.delete(self.deck_cards_known, (0), axis=0)
            #and copy the top row of deck cards to deck cards known, since we know it now
            self.deck_cards_known[0,:] = self.deck_cards[0,:]

            if self.deck_position == 7:
                #flatten, put zeros at end, reshape
                #part of rules to how deck works, weird
                flat_deck = self.deck_cards.flatten()
                #count occurances of 0's
                num_zeros = np.count_nonzero(flat_deck == 0)
                #remove zeros
                flat_deck_no_zeros = np.delete(flat_deck, np.where(flat_deck == 0), axis = -1)
                #add them to end
                flat_deck_zeros = np.append(flat_deck_no_zeros, np.zeros(num_zeros, dtype=np.int8), axis = -1)
                #update state
                self.deck_cards = flat_deck_zeros.reshape((8,3))
                #at this point, since we have cycled through, we now have complete information and can just copy
                self.deck_cards_known = self.deck_cards
                self.deck_position = 0
                #no reward, unless its the 7th
                game_reward += game_score_deck_cycle
                reward += agent_reward_deck_cycle
            else:
                reward += agent_reward_deck_flip
                #increase deck position (so to know when to reset deck and penalize)
                self.deck_position = self.deck_position + 1
        
        #action 1 is tapping active deck card, attempting to sent to suit
        elif action == 1:
            can_deck_to_suit, suit_to_update, active_deck_card_index = deck_to_suit_check(self.deck_cards, self.suit_cards)
            if can_deck_to_suit:
                #update suit cards
                self.suit_cards[suit_to_update] = self.suit_cards[suit_to_update] + 1
                #set card val to 0 for empty at the given index
                self.deck_cards[0,active_deck_card_index] = 0
                self.deck_cards_known[0,active_deck_card_index] = 0
                game_reward += agent_reward_deck_to_suit
                reward += agent_reward_deck_to_suit

                #check if terminal/goal state, all suits at king
                if self.suit_cards[0] == 13 and self.suit_cards[1] == 13 and self.suit_cards[2] == 13 and self.suit_cards[3] == 13:
                    print("VICTORY!")
                    print(self.suit_cards)
                    terminated = True
                    game_reward += game_score_victory
                    reward += agent_reward_victory

        
        #actions 2-8 are trying to move top deck card to one of 7 other piles
        elif action >= 2 and action <= 8:
            action_i = action - 2
            can_deck_to_pile, pile_index_to_update, deck_card_val, deck_card_index = deck_to_pile_check(self.deck_cards, self.pile_cards, action_i)      
            if can_deck_to_pile:
                #update pile_cards
                self.pile_cards[pile_index_to_update,action_i] = deck_card_val
                #set card val to 0 for empty at the given index
                self.deck_cards[0,deck_card_index] = 0
                self.deck_cards_known[0,deck_card_index] = 0
                #deck to pile rewards
                game_reward += game_score_deck_to_pile
                reward += agent_reward_deck_to_pile
        
        #actions 9-15 are trying to move suit's 0 (hearts) to one of 7 other piles
        #actions 16-22 are trying to move suit's 1 (diamonds) to one of 7 other piles
        #...etc
        elif action >= 9 and action <= 36:
            pile_i = (action - 9) % 7
            suit_j = (action - 9) // 7

            can_suit_to_pile, pile_index_to_update, suit_card_val = suit_to_pile_check(self.suit_cards, self.pile_cards, pile_i, suit_j)

            if can_suit_to_pile:
                #update pile_cards
                self.pile_cards[pile_index_to_update,pile_i] = suit_card_val
                #set suit card val to one less
                self.suit_cards[suit_j] = int(self.suit_cards[suit_j] - 1)
                #suit to pile rewards (negative)
                game_reward += game_score_suit_to_pile
                reward += agent_reward_suit_to_pile
        
        #actions 37-43 are trying to move bottom-most card in one of 7 piles
        #to their given suits
        elif action >= 37 and action <= 43:
            pile_i = (action - 37)

            can_pile_to_suit, pile_card_suit, pile_card_num, pile_card_index = pile_to_suit_check(self.pile_cards, self.suit_cards, pile_i)

            if can_pile_to_suit:
                #update suit and pile cards
                self.suit_cards[pile_card_suit] = pile_card_num
                self.pile_cards[pile_card_index,pile_i] = 0
                #pile to suit reward
                game_reward += game_score_pile_to_suit
                reward += agent_reward_pile_to_suit
                #check if the pile is now empty, and there are upside-down cards behind it
                #if so, turn it up and add it to correct spot in pile_cards
                if not self.pile_cards[:,pile_i].any():
                    if self.pile_behind_cards[:,pile_i].any():
                        #start at bottom of pile behind cards
                        flip_pile_card_index = 0
                        flip_pile_card = self.pile_behind_cards[flip_pile_card_index,pile_i]
                        #print("initial flip pile card: ", flip_pile_card)
                        #find first instance of upside-down card
                        while flip_pile_card == 0:
                            flip_pile_card_index += 1
                            flip_pile_card = self.pile_behind_cards[flip_pile_card_index,pile_i]
                        #set the pile card at the given index
                        flip_pile_card_num = int(get_suit_and_num(flip_pile_card)[1] - 1)
                        self.pile_cards[flip_pile_card_num,pile_i] = flip_pile_card
                        #remove pile_behind_cards and _known
                        self.pile_behind_cards[flip_pile_card_index,pile_i] = 0
                        self.pile_behind_cards_known[flip_pile_card_index,pile_i] = 0
                        #pile card reveal reward
                        game_reward += game_score_pile_card_reveal
                        reward += agent_reward_pile_card_reveal       

                #check if terminal/goal state, all suits at king
                if self.suit_cards[0] == 13 and self.suit_cards[1] == 13 and self.suit_cards[2] == 13 and self.suit_cards[3] == 13:
                    print("VICTORY!")
                    print(self.suit_cards)
                    terminated = True
                    game_reward += game_score_victory
                    reward += agent_reward_victory

        #actions 44 to 547 are pile to pile moves, defined by the vars below
        elif action >= 44 and action <= 547:
            pile_i = (action - 44) // 72
            piles_to_move = int(((action - 44 - (72*pile_i)) // 12) + 1)
            #print("piles to move: ", piles_to_move)
            pile_to_move_to_j = (pile_i + piles_to_move) % 7
            card_k = ((action - 44) % 12) + 1

            can_pile_to_pile = pile_to_pile_check(self.pile_cards, pile_i, pile_to_move_to_j, card_k)

            if can_pile_to_pile:
                #if we have made it all the way through and not returned due to illegal move
                #add the bottom of from pile cards to destination pile cards
                for i in range(card_k+1):
                    self.pile_cards[i,pile_to_move_to_j] = self.pile_cards[i,pile_i]
                    self.pile_cards[i,pile_i] = 0
                    #check if the pile is now empty, and there are upside-down cards behind it
                    #if so, turn it up and add it to correct spot in pile_cards
                    if not self.pile_cards[:,pile_i].any():
                        if self.pile_behind_cards[:,pile_i].any():
                            #start at bottom of pile behind cards
                            flip_pile_card_index = 0
                            flip_pile_card = self.pile_behind_cards[flip_pile_card_index,pile_i]
                            #print("initial flip pile card: ", flip_pile_card)
                            #find first instance of upside-down card
                            while flip_pile_card == 0:
                                flip_pile_card_index += 1
                                flip_pile_card = self.pile_behind_cards[flip_pile_card_index,pile_i]
                            #set the pile card at the given index
                            flip_pile_card_num = int(get_suit_and_num(flip_pile_card)[1] - 1)
                            self.pile_cards[flip_pile_card_num,pile_i] = flip_pile_card
                            #remove card remove pile_behind_cards and _known
                            self.pile_behind_cards[flip_pile_card_index,pile_i] = 0
                            self.pile_behind_cards_known[flip_pile_card_index,pile_i] = 0
                            #pile card reveal reward
                            game_reward += game_score_pile_card_reveal
                            reward += agent_reward_pile_card_reveal

        self.game_episode_rewards += game_reward
        self.agent_episode_rewards += reward
        #try if agent rewards get too low, just terminate
        if self.agent_episode_rewards < -300:
            terminated = True

        if self.render_mode == "human":
            self.render()
        # return self._get_obs(), reward, terminated, False, {"action_mask": self.action_mask(self.deck_cards, self.suit_cards, self.pile_cards)}
        return self._get_obs(), reward, terminated, {"action_mask": self.action_mask(self.deck_cards, self.suit_cards, self.pile_cards)}


    def _get_obs(self):
        return {
            "deck_position": self.deck_position,
            "decks_actual": self.deck_cards,
            "decks": self.deck_cards_known,
            "suits": self.suit_cards,
            "piles": self.pile_cards,
            "piles_behind": self.pile_behind_cards_known,
            "piles_behind_actual": self.pile_behind_cards
        }
        #return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        #super().reset(seed=None)

        shuffled_deck = get_shuffled_deck(np.random)

        self.game_episode_rewards = 0
        self.agent_episode_rewards = 0

        self.deck_cards = shuffled_deck[:24].reshape((8,3))
        self.deck_cards_known = np.zeros((8,3), dtype=np.int8)
        self.deck_cards_known.fill(53)
        self.deck_cards_known[0,:] = self.deck_cards[0,:]
        #testing
        #self.deck_cards[0,0] = 1

        self.deck_position = 0

        self.suit_cards = np.zeros(4, dtype=np.int8)
        #testing
        #self.suit_cards[2] = 1

        self.pile_cards = np.zeros((13,7), dtype=np.int8)
        self.pile_behind_cards_known = np.zeros((6,7), dtype=np.int8)
        self.pile_behind_cards = np.zeros((6,7), dtype=np.int8)
        card_index = 24
        for i in range(7):
            bottom_pile_card = get_suit_and_num(shuffled_deck[card_index])
            self.pile_cards[bottom_pile_card[1]-1,i] = shuffled_deck[card_index]
            card_index += 1

            for j in range(i):
                self.pile_behind_cards_known[j,i] = 53
                self.pile_behind_cards[j,i] = shuffled_deck[card_index]
                card_index += 1

        #testing
        # self.pile_cards[0,0] = 0
        # self.pile_cards[1,0] = 0
        # self.pile_cards[2,0] = 0
        # self.pile_cards[3,0] = 0
        # self.pile_cards[4,0] = 0
        # self.pile_cards[5,0] = 0
        # self.pile_cards[6,0] = 0
        # self.pile_cards[7,0] = 0
        # self.pile_cards[8,0] = 0
        # self.pile_cards[9,0] = 0
        # self.pile_cards[10,0] = 0
        # self.pile_cards[11,0] = 0
        # self.pile_cards[12,0] = 0

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {"action_mask": self.action_mask(self.deck_cards, self.suit_cards, self.pile_cards)}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        #player_sum, dealer_card_value, usable_ace = self._get_obs()
        screen_width, screen_height = 1200, 640
        card_img_height = screen_height // 4
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 20

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        small_font = get_font(
            os.path.join("font", "Minecraft.ttf"), screen_height // 15
        )

        score_text = small_font.render(
            "Score: " + str(self.game_episode_rewards), True, white
        )
        score_text_rect = self.screen.blit(score_text, (spacing, 0))

        reward_text = small_font.render(
            "Reward: " + str(self.agent_episode_rewards), True, white
        )
        reward_text_rect = self.screen.blit(reward_text, (screen_width // 2, 0))

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))


        #UPSIDE DOWN DECK CARD
        hidden_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
        self.screen.blit(
            hidden_card_img,
            (
                0,
                score_text_rect.bottom + spacing,
            ),
        )
        #DECK CARDS
        if self.deck_cards[0,0] > 0:
            deck_card_one_suit_and_num = get_suit_and_num(self.deck_cards[0,0])
            deck_card_one_suit_char = get_suit_char_from_val(deck_card_one_suit_and_num[0])
            deck_card_one_num_char = get_card_char(deck_card_one_suit_and_num[1])
            deck_card_one_img = scale_card_img(
                get_image(
                    os.path.join(
                        "img",
                        f"{deck_card_one_suit_char}{deck_card_one_num_char}.png",
                    )
                )
            )
            deck_card_one_rect = self.screen.blit(
                deck_card_one_img,
                (
                    card_img_width + spacing,
                    score_text_rect.bottom + spacing,
                ),
            )

        if self.deck_cards[0,1] > 0:
            deck_card_two_suit_and_num = get_suit_and_num(self.deck_cards[0,1])
            deck_card_two_suit_char = get_suit_char_from_val(deck_card_two_suit_and_num[0])
            deck_card_two_num_char = get_card_char(deck_card_two_suit_and_num[1])
            deck_card_two_img = scale_card_img(
                get_image(
                    os.path.join(
                        "img",
                        f"{deck_card_two_suit_char}{deck_card_two_num_char}.png",
                    )
                )
            )
            deck_card_two_rect = self.screen.blit(
                deck_card_two_img,
                (
                    1.5*card_img_width + spacing,
                    score_text_rect.bottom + spacing,
                ),
            )

        if self.deck_cards[0,2] > 0:
            deck_card_three_suit_and_num = get_suit_and_num(self.deck_cards[0,2])
            deck_card_three_suit_char = get_suit_char_from_val(deck_card_three_suit_and_num[0])
            deck_card_three_num_char = get_card_char(deck_card_three_suit_and_num[1])
            deck_card_three_img = scale_card_img(
                get_image(
                    os.path.join(
                        "img",
                        f"{deck_card_three_suit_char}{deck_card_three_num_char}.png",
                    )
                )
            )
            deck_card_three_rect = self.screen.blit(
                deck_card_three_img,
                (
                    2*card_img_width + spacing,
                    score_text_rect.bottom + spacing,
                ),
            )


        #SUIT CARDS
        if self.suit_cards[3] > 0:
            suit_card_four_suit_char = get_suit_char_from_val(3)
            suit_card_four_num_char = get_card_char(self.suit_cards[3])
            suit_card_four_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
            if suit_card_four_suit_char != "Card":
                suit_card_four_img = scale_card_img(
                    get_image(
                        os.path.join(
                            "img",
                            f"{suit_card_four_suit_char}{suit_card_four_num_char}.png",
                        )
                    )
                )
            self.screen.blit(
                suit_card_four_img,
                (
                    screen_width - card_img_width,
                    score_text_rect.bottom + spacing,
                ),
            )

        if self.suit_cards[2] > 0:
            suit_card_three_suit_char = get_suit_char_from_val(2)
            suit_card_three_num_char = get_card_char(self.suit_cards[2])
            suit_card_three_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
            if suit_card_three_suit_char != "Card":
                suit_card_three_img = scale_card_img(
                    get_image(
                        os.path.join(
                            "img",
                            f"{suit_card_three_suit_char}{suit_card_three_num_char}.png",
                        )
                    )
                )
            self.screen.blit(
                suit_card_three_img,
                (
                    screen_width - 2*card_img_width,
                    score_text_rect.bottom + spacing,
                ),
            )

        if self.suit_cards[1] > 0:
            suit_card_two_suit_char = get_suit_char_from_val(1)
            suit_card_two_num_char = get_card_char(self.suit_cards[1])
            suit_card_two_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
            if suit_card_two_suit_char != "Card":
                suit_card_two_img = scale_card_img(
                    get_image(
                        os.path.join(
                            "img",
                            f"{suit_card_two_suit_char}{suit_card_two_num_char}.png",
                        )
                    )
                )
            self.screen.blit(
                suit_card_two_img,
                (
                    screen_width - 3*card_img_width,
                    score_text_rect.bottom + spacing,
                ),
            )

        if self.suit_cards[0] > 0:
            suit_card_one_suit_char = get_suit_char_from_val(0)
            suit_card_one_num_char = get_card_char(self.suit_cards[0])
            suit_card_one_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
            if suit_card_one_suit_char != "Card":
                suit_card_one_img = scale_card_img(
                    get_image(
                        os.path.join(
                            "img",
                            f"{suit_card_one_suit_char}{suit_card_one_num_char}.png",
                        )
                    )
                )
            self.screen.blit(
                suit_card_one_img,
                (
                    screen_width - 4*card_img_width,
                    score_text_rect.bottom + spacing,
                ),
            )


        #PILE CARDS

        for i in range(int(self.pile_cards.shape[1])):
            spacing_counter = 0
            #place behind cards
            for j in range(6):
                if self.pile_behind_cards[j,i] != 0:
                    pile_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
                    self.screen.blit(
                        pile_card_img,
                        (
                            i * (card_img_width + 50),
                            (screen_height // 2.75) + (spacing // 2) + spacing_counter,
                        ),
                    )
                    spacing_counter += 10

            for j in range(12,-1,-1):
                if self.pile_cards[j,i] > 0:
                    pile_card_suit_and_num = get_suit_and_num(self.pile_cards[j,i])
                    pile_card_suit_char = get_suit_char_from_val(pile_card_suit_and_num[0])
                    pile_card_num_char = get_card_char(pile_card_suit_and_num[1])
                    #pile_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
                    #if pile_card_suit_char != "Card":
                    pile_card_img = scale_card_img(
                        get_image(
                            os.path.join(
                                "img",
                                f"{pile_card_suit_char}{pile_card_num_char}.png",
                            )
                        )
                    )
                    self.screen.blit(
                        pile_card_img,
                        (
                            i * (card_img_width + 50),
                            (screen_height // 2.75) + (spacing // 2) + spacing_counter,
                        ),
                    )
                    spacing_counter += 30


        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


    def close(self):
        if hasattr(self, "screen"):
            print("closing")
            #import pygame
            print("imported")
            try:
                pygame.display.quit()
                pygame.quit()
            except:
                print("no pygame")


# Pixel art from Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)