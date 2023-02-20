import datetime
import pathlib

import gym
import numpy
import torch

from .abstract_game import AbstractGame


import os
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.spaces import Tuple, Discrete, MultiDiscrete
from gym.error import DependencyNotInstalled

#credit: https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
def first_nonzero_index(array):
    """Return the index of the first non-zero element of array. If all elements are zero, return -1."""
    
    fnzi = -1 # first non-zero index
    indices = np.flatnonzero(array)
       
    if (len(indices) > 0):
        fnzi = indices[0]
        
    return fnzi

#gets suit number given card num (worried about it always returning 0, shouldnt it be -1?)
def get_suit(card_num):
    if card_num / 13 <= 1:
        return 0
    elif card_num / 26 <= 1:
        return 1
    elif card_num / 39 <= 1:
        return 2
    else:
        return 3
    return 0

#gets suit and card num, given card num
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

#gets character from the suit (for displaying images)
def get_suit_char_from_val(suit_val):
    if suit_val == 0:
        return "H"
    elif suit_val == 1:
        return "D"
    elif suit_val == 2:
        return "S"
    elif suit_val == 3:
        return "C"

#gets character from the card num (for displaying images)
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

#returns an np array size 52 with the numbers 1-53(non-inc), False-resampled
def get_shuffled_deck(np_random):
    return np_random.choice(range(1,53), 52, False)

#checks if can move card from deck to suit
def deck_to_suit_check(deck_cards_param, suit_cards_param, highest_nonzero_deck):
    #-1 is encoded in this variable as the row is entirely empty, so cant move anything from empty deck top row
    if highest_nonzero_deck > -1:
        #make sure it can slot into the pile properly
        active_deck_card = deck_cards_param[0, highest_nonzero_deck]
        active_deck_card_suit_and_num = get_suit_and_num(active_deck_card)
        a_suit = active_deck_card_suit_and_num[0]
        a_num = active_deck_card_suit_and_num[1]
        #if suit cards is one below active deck card number, then can add to suits
        if suit_cards_param[a_suit] + 1 == a_num:
            return True, a_suit
    return False, False

#checks if can move from deck to pile
def deck_to_pile_check(deck_cards_param, pile_cards_param, pile_i, highest_nonzero_deck):
    #-1 is encoded in this variable as the row is entirely empty, so cant move anything from empty deck top row
    if highest_nonzero_deck > -1:
        #get suit and num of the deck card
        c_suit_and_num = get_suit_and_num(deck_cards_param[0, highest_nonzero_deck])
        c_suit = c_suit_and_num[0]
        c_num = c_suit_and_num[1]

        #check if deck card is king, if so just check if pile bottom most is empty
        if c_num == 13:
            if pile_cards_param[0,pile_i] == 0:
                return True
        else:

            #check pile bottom most is not empty
            if pile_cards_param[0, pile_i] > 0:
                
                #check the pile you are trying to move to's bottom most card is one lower and different suit
                t_suit_and_num = get_suit_and_num(pile_cards_param[0,pile_i])
                
                #check opposite suits and current card is one lower than target
                if ((c_suit <= 1 and t_suit_and_num[0] >= 2) or (c_suit >= 2 and t_suit_and_num[0] <= 1)) and (c_num + 1 == t_suit_and_num[1]):
                    return True  

    return False


#checks if can move from suit down to pile
def suit_to_pile_check(suit_cards_param, pile_cards_param, pile_i, suit_j):
    #check suit card you are trying to move exists, i.e. greater than zero
    #and also greater than one because why consider moving ace down?
    c_num = suit_cards_param[suit_j]
    if c_num > 1:
        
        #first, check it isnt a king edge case
        if c_num == 13:
            
            #check if pile trying to move to is empty
            if pile_cards_param[0, pile_i] == 0:
                return True
        
        else:
            
            #check pile bottom most is not empty
            if pile_cards_param[0, pile_i] > 0:
                
                #check the pile you are trying to move to's bottom most card is one lower and different suit
                t_suit_and_num = get_suit_and_num(pile_cards_param[0,pile_i])
                
                #check opposite suits and current card is one lower than target
                if ((suit_j <= 1 and t_suit_and_num[0] >= 2) or (suit_j >= 2 and t_suit_and_num[0] <= 1)) and (c_num + 1 == t_suit_and_num[1]):
                    return True
    
    return False

#check if can move from pile to suit
def pile_to_suit_check(pile_cards_param, suit_cards_param, pile_i):
    #check there is a card in the pile
    if pile_cards_param[0,pile_i] > 0:
        #get suit and number of bottom most pile card
        a_suit_and_num = get_suit_and_num(pile_cards_param[0,pile_i])
        #check that the card is indeed one higher than what is current in its suit
        if a_suit_and_num[1] - 1 == suit_cards_param[a_suit_and_num[0]]:
            return True, a_suit_and_num[0]
    return False, False


#check that agent can move some card index (and all the cards below it) to
#another pile
#index 0 is lowest card
def pile_to_pile_check(pile_cards_param, pile_i, pile_to_move_to_j, card_k, highest_nonzero_pile_i):
    #check current pile is not empty
    if highest_nonzero_pile_i > -1:
        #check that there is a card which is a number 
        #of cards below the top one you want to move
        #i.e. if the pile is ace,2,3,4, and card_k=7, that is out of bounds
        #max would be card_k = 3, which would move ace. card_k = 0 moves card 4
        #only 12 actions because in k,q,j,10,9,8,7,6,5,4,3,2,ace, it would only
        #ever make sense to go down to the 2 and move that, otherwise
        #send the ace to the suits
        #highest nonzero pile i here should be 3
        if card_k <= highest_nonzero_pile_i:
            #get the index of the current card you want to move. In the above example
            #c_k = 3 would mean we move index 0, the ace, exactly. c_k=0 means we move
            #index 3, the 4, exactly
            index_to_move = highest_nonzero_pile_i - card_k
            #get the current top card trying to move's suit and num
            c_suit_and_num = get_suit_and_num(pile_cards_param[index_to_move,pile_i])

            #if the current card we want to move is a king, check the target pile is empty
            if c_suit_and_num[1] == 13:
                if not pile_cards_param[:,pile_to_move_to_j].any():
                    return True
            else:
                #else its some other card, so check the pile we want to move to has
                #card at index 0, and that it is one lower and different suit than current
                if pile_cards_param[0,pile_to_move_to_j] > 0:
                    t_suit_and_num = get_suit_and_num(pile_cards_param[0,pile_to_move_to_j])
                    #check diff suit and one higher
                    if ((c_suit_and_num[0] <= 1 and t_suit_and_num[0] >= 2) or (c_suit_and_num[0] >= 2 and t_suit_and_num[0] <= 1)) and (c_suit_and_num[1] + 1 == t_suit_and_num[1]):
                        return True

    return False


#solitaire world environment class
class SolitaireWorldEnv(gym.Env):

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

        self._seed = 1314

        self.render_mode = render_mode

    #returns array of 0s and 1s, of length num actions,
    #where if the value at the index in the array for the action is 1
    #the action is valid
    def action_mask(self, deck_cards_p, suit_cards_p, pile_cards_p):
        mask = np.zeros(548, dtype=np.int8)
        #can always tap the deck
        mask[0] = 1
        #gets the highest nonzero index by rows, for the first row of the deck array
        #credit: https://stackoverflow.com/questions/67921398/find-the-index-of-last-non-zero-value-per-column-of-a-2-d-array
        #deck first nonzero, obviously we only care about row 0, hence [0] at end
        highest_nonzero_deck_am = np.where(np.count_nonzero(deck_cards_p, axis=1)==0, -1, (deck_cards_p.shape[1]-1) - np.argmin(deck_cards_p[:,::-1]==0, axis=1))[0]
        #need highest nonzero, since needed for checking pile to pile actions
        highest_nonzeros_piles_am = np.where(np.count_nonzero(pile_cards_p, axis=0)==0, -1, (pile_cards_p.shape[0]-1) - np.argmin(pile_cards_p[::-1,:]==0, axis=0))

        #for each action, besides 0 because always possible to tap deck
        for i in range(1,548):
            if i == 1:
                can_deck_to_suit, _ = deck_to_suit_check(deck_cards_p, suit_cards_p, highest_nonzero_deck_am)
                if can_deck_to_suit:
                    mask[i] = 1

            elif i >= 2 and i <= 8:
                p_i = i - 2
                can_deck_to_pile = deck_to_pile_check(deck_cards_p, pile_cards_p, p_i, highest_nonzero_deck_am)
                if can_deck_to_pile:
                    mask[i] = 1

            elif i >= 9 and i <= 36:
                p_i = (i - 9) % 7
                s_j = (i - 9) // 7
                can_suit_to_pile = suit_to_pile_check(suit_cards_p, pile_cards_p, p_i, s_j)
                if can_suit_to_pile:
                    mask[i] = 1

            elif i >= 37 and i <= 43:
                p_i = (i - 37)
                can_pile_to_suit, _ = pile_to_suit_check(pile_cards_p, suit_cards_p, p_i)
                if can_pile_to_suit:
                    mask[i] = 1

            elif i >= 44 and i <= 547:
                p_i = (i - 44) // 72
                p_to_m = int(((i - 44 - (72*p_i)) // 12) + 1)
                p_to_m_j = (p_i + p_to_m) % 7
                c_k = (i - 44) % 12
                can_pile_to_pile = pile_to_pile_check(pile_cards_p, p_i, p_to_m_j, c_k, highest_nonzeros_piles_am[p_i])
                if can_pile_to_pile:
                    mask[i] = 1

        return mask

    #gym step func
    def step(self, action):
        assert self.action_space.contains(action)

        terminated = False
        #game reward for initial part of step starts at zero, changes depending on action outcome
        game_reward = 0
        #agent reward however is -1 because we want to penalize it for steping, since time is a factor
        reward = 0

        #source: https://en.wikipedia.org/wiki/Klondike_(solitaire) Microsoft Windows Scoring section
        game_score_deck_to_pile = 5
        game_score_deck_to_suit = 10
        game_score_pile_to_suit = 10
        game_score_pile_card_reveal = 5
        game_score_suit_to_pile = -15
        game_score_deck_cycle = -20
        game_score_victory = 10000

        agent_reward_deck_to_pile = 5
        agent_reward_deck_to_suit = 5
        agent_reward_pile_to_suit = 5
        agent_reward_pile_to_pile = 0
        agent_reward_pile_card_reveal = 0
        agent_reward_suit_to_pile = 0
        agent_reward_deck_cycle = 0
        agent_reward_deck_flip = 0
        agent_reward_victory = 0


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
            highest_nonzero_deck = np.where(np.count_nonzero(self.deck_cards, axis=1)==0, -1, (self.deck_cards.shape[1]-1) - np.argmin(self.deck_cards[:,::-1]==0, axis=1))[0]
            can_deck_to_suit, deck_card_suit = deck_to_suit_check(self.deck_cards, self.suit_cards, highest_nonzero_deck)
            if can_deck_to_suit:
                #update suit cards
                self.suit_cards[deck_card_suit] = self.suit_cards[deck_card_suit] + 1
                #set card val to 0 for empty at the given index
                self.deck_cards[0,highest_nonzero_deck] = 0
                self.deck_cards_known[0,highest_nonzero_deck] = 0
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
            pile_i = action - 2
            highest_nonzero_deck = np.where(np.count_nonzero(self.deck_cards, axis=1)==0, -1, (self.deck_cards.shape[1]-1) - np.argmin(self.deck_cards[:,::-1]==0, axis=1))[0]

            can_deck_to_pile = deck_to_pile_check(self.deck_cards, self.pile_cards, pile_i, highest_nonzero_deck)      
            if can_deck_to_pile:
                #shift all cards one index up to make room for this at bottom (index 0)
                self.pile_cards[1:,pile_i] = self.pile_cards[:-1,pile_i]
                #set bottom most index card to be the one we added, since we can now
                self.pile_cards[0,pile_i] = self.deck_cards[0,highest_nonzero_deck]

                #set deck card val to 0 for empty at the given index
                self.deck_cards[0,highest_nonzero_deck] = 0
                self.deck_cards_known[0,highest_nonzero_deck] = 0

                #deck to pile rewards
                game_reward += game_score_deck_to_pile
                reward += agent_reward_deck_to_pile
        
        #actions 9-15 are trying to move suit's 0 (hearts) to one of 7 other piles
        #actions 16-22 are trying to move suit's 1 (diamonds) to one of 7 other piles
        #...etc
        elif action >= 9 and action <= 36:
            pile_i = (action - 9) % 7
            suit_j = (action - 9) // 7

            can_suit_to_pile = suit_to_pile_check(self.suit_cards, self.pile_cards, pile_i, suit_j)

            if can_suit_to_pile:
                #shift pile cards up, to make room for suit card to add
                self.pile_cards[1:,pile_i] = self.pile_cards[:-1,pile_i]
                #set bottom most index card to be the one we added, since we can now
                #have to multiply by suit_j*13 to get original card val back from suit and num
                self.pile_cards[0,pile_i] = int((suit_j * 13) + self.suit_cards[suit_j])

                #reduce suit card val, since we took a card off it
                self.suit_cards[suit_j] -= 1

                #suit to pile rewards (negative)
                game_reward += game_score_suit_to_pile
                reward += agent_reward_suit_to_pile
       
        #actions 37-43 are trying to move bottom-most card in one of 7 piles
        #to their given suits
        elif action >= 37 and action <= 43:
            pile_i = (action - 37)

            can_pile_to_suit, pile_card_suit = pile_to_suit_check(self.pile_cards, self.suit_cards, pile_i)

            if can_pile_to_suit:
                #update suit cards
                self.suit_cards[pile_card_suit] += 1

                #update pile cards
                #(set cards from beginning to one from end to the cards from 1 to end)
                self.pile_cards[:-1,pile_i] = self.pile_cards[1:,pile_i]
                #and set end card to 0, since we removed a card from the pile
                self.pile_cards[-1,pile_i] = 0

                #pile to suit reward
                game_reward += game_score_pile_to_suit
                reward += agent_reward_pile_to_suit

                #check if the pile is now empty, and there are upside-down cards behind it
                #if so, turn it up and add it to correct spot in pile_cards
                if not self.pile_cards[:,pile_i].any():
                    #get highest nonzero for pile behind card
                    highest_nonzeros_piles_behind = np.where(np.count_nonzero(self.pile_behind_cards, axis=0)==0, -1, (self.pile_behind_cards.shape[0]-1) - np.argmin(self.pile_behind_cards[::-1,:]==0, axis=0))[pile_i]
                    if highest_nonzeros_piles_behind > -1:
                        self.pile_cards[0, pile_i] = self.pile_behind_cards[highest_nonzeros_piles_behind,pile_i]
                        self.pile_behind_cards[highest_nonzeros_piles_behind, pile_i] = 0
                        self.pile_behind_cards_known[highest_nonzeros_piles_behind, pile_i] = 0
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
            #set agent reward any time take this step
            reward += agent_reward_pile_to_pile

            pile_i = (action - 44) // 72
            piles_to_move = int(((action - 44 - (72*pile_i)) // 12) + 1)
            #print("piles to move: ", piles_to_move)
            pile_to_move_to_j = (pile_i + piles_to_move) % 7
            card_k = (action - 44) % 12

            highest_nonzeros_piles = np.where(np.count_nonzero(self.pile_cards, axis=0)==0, -1, (self.pile_cards.shape[0]-1) - np.argmin(self.pile_cards[::-1,:]==0, axis=0))
            highest_nonzero_pile_i = highest_nonzeros_piles[pile_i]
            can_pile_to_pile = pile_to_pile_check(self.pile_cards, pile_i, pile_to_move_to_j, card_k, highest_nonzero_pile_i)

            if can_pile_to_pile:
                #if can pile to pile,
                #card_k = 0, highest_nonzero_pile_i = 4, we are moving 4 cards to pile j, and these cards go under the others at pile j
                #so pile j needs to shift cards up by highest_nzpi - c_k, i.e. 4
                #+1 because ...?
                index_to_move = highest_nonzero_pile_i - card_k + 1

                #first, shift cards up
                self.pile_cards[index_to_move:,pile_to_move_to_j] = self.pile_cards[:(13 - index_to_move),pile_to_move_to_j]
                #then add in cards from pile i
                self.pile_cards[:index_to_move,pile_to_move_to_j] = self.pile_cards[:index_to_move,pile_i]
                #finally shift pile i cards down
                self.pile_cards[:(13 - index_to_move),pile_i] = self.pile_cards[index_to_move:,pile_i]
                #and zero out rest of pile
                self.pile_cards[(13 - index_to_move):,pile_i] = 0 

                #if pile is now empty that we removed some cards, i.e. we removed top card because card k is 0, meaning no cards below it selected as top
                if card_k == 0:
                    #get highest nonzero for pile behind card
                    highest_nonzeros_piles_behind = np.where(np.count_nonzero(self.pile_behind_cards, axis=0)==0, -1, (self.pile_behind_cards.shape[0]-1) - np.argmin(self.pile_behind_cards[::-1,:]==0, axis=0))[pile_i]
                    if highest_nonzeros_piles_behind > -1:
                        self.pile_cards[0, pile_i] = self.pile_behind_cards[highest_nonzeros_piles_behind,pile_i]
                        self.pile_behind_cards[highest_nonzeros_piles_behind, pile_i] = 0
                        self.pile_behind_cards_known[highest_nonzeros_piles_behind, pile_i] = 0
                        game_reward += game_score_pile_card_reveal
                        reward += agent_reward_pile_card_reveal

        self.game_episode_rewards += game_reward
        self.agent_episode_rewards += reward
        self.episode_steps += 1

        current_action_mask = self.action_mask(self.deck_cards, self.suit_cards, self.pile_cards)
        self.current_legal_actions = np.nonzero(current_action_mask)[0].tolist()

        #try if agent rewards get too low, just terminate
        # if self.episode_steps > 30:
        #     terminated = True

        if self.render_mode == "human":
            self.render()
        # return self._get_obs(), reward, terminated, False, {"action_mask": self.action_mask(self.deck_cards, self.suit_cards, self.pile_cards)}
        return self._get_obs(), reward, terminated, {"action_mask": current_action_mask}


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

    def seed(self, seed_val):
        self._seed = seed_val

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        shuffled_deck = get_shuffled_deck(self.np_random)

        self.game_episode_rewards = 0
        self.agent_episode_rewards = 0
        self.episode_steps = 0

        self.deck_cards = shuffled_deck[:24].reshape((8,3))
        self.deck_cards_known = np.zeros((8,3), dtype=np.int8)
        self.deck_cards_known.fill(53)
        self.deck_cards_known[0,:] = self.deck_cards[0,:]
        #testing
        # self.deck_cards[0,2] = 9
        # self.deck_cards_known[0,2] = 9
        # self.deck_cards[0,1] = 15
        # self.deck_cards_known[0,1] = 15
        # self.deck_cards[0,0] = 28
        # self.deck_cards_known[0,0] = 28

        self.deck_position = 0

        self.suit_cards = np.zeros(4, dtype=np.int8)
        #testing
        # self.suit_cards[0] = 3
        # self.suit_cards[1] = 3
        # self.suit_cards[2] = 3
        # self.suit_cards[3] = 3

        #NOTE: SHOULD try to MAKE THIS MORE EFFICIENT
        self.pile_cards = np.zeros((13,7), dtype=np.int8)
        self.pile_behind_cards_known = np.zeros((6,7), dtype=np.int8)
        self.pile_behind_cards = np.zeros((6,7), dtype=np.int8)
        card_index = 24
        for i in range(7):
            bottom_pile_card = get_suit_and_num(shuffled_deck[card_index])
            self.pile_cards[0,i] = shuffled_deck[card_index]
            card_index += 1

            for j in range(i):
                self.pile_behind_cards_known[j,i] = 53
                self.pile_behind_cards[j,i] = shuffled_deck[card_index]
                card_index += 1

        #testing
        # self.pile_cards[0,0] = 4
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

        # self.pile_cards[0,1] = 26
        # self.pile_cards[1,1] = 0
        # self.pile_cards[2,1] = 0
        # self.pile_cards[3,1] = 0
        # self.pile_cards[4,1] = 0
        # self.pile_cards[5,1] = 0
        # self.pile_cards[6,1] = 0
        # self.pile_cards[7,1] = 0
        # self.pile_cards[8,1] = 0
        # self.pile_cards[9,1] = 0
        # self.pile_cards[10,1] = 0
        # self.pile_cards[11,1] = 0
        # self.pile_cards[12,1] = 0

        # self.pile_behind_cards[0,1] = 0
        # self.pile_behind_cards_known[0,1] = 0

        current_action_mask = self.action_mask(self.deck_cards, self.suit_cards, self.pile_cards)
        self.current_legal_actions = np.nonzero(current_action_mask)[0].tolist()

        if self.render_mode == "human":
            self.render()
        # return self._get_obs(), {"action_mask": self.action_mask(self.deck_cards, self.suit_cards, self.pile_cards)}
        return self._get_obs()

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
            import pygame

            pygame.display.quit()
            pygame.quit()



class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
        print("rewards: +5 for good (should find pile to suit suit to piles), Thirty STEPS, fc, sgd, clipped grad norm 1")

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 1
        # self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (21, 23, 7)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(548))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        # self.selfplay_on_gpu = torch.cuda.is_available()
        self.max_moves = 30  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        print("num sims: 50")
        self.discount = 0.999  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 1.25
        self.root_exploration_fraction = 0.25
        print("root dirichlet 1.25")
        print("root explor fraction 0.25")

        # UCB formula
        self.pb_c_base = 500
        self.pb_c_init = 1.25
        print("pb_c_base: 500")
        print("pb_c_init: 1.25")



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        print("support size: 10")

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 2  # Number of blocks in the ResNet
        self.channels = 8  # Number of channels in the ResNet
        self.reduced_channels_reward = 4  # Number of channels in reward head
        self.reduced_channels_value = 4  # Number of channels in value head
        self.reduced_channels_policy = 8  # Number of channels in policy head
        self.resnet_fc_reward_layers = [32, 32]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [32, 32]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [32, 32]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 128
        self.fc_representation_layers = [128]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [128]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [128]  # Define the hidden layers in the reward network
        self.fc_value_layers = [128]  # Define the hidden layers in the value network
        self.fc_policy_layers = [128]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 20000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 100  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        print("value weight loss: 0.25")
        print("CUDA???")
        print(torch.__version__)
        print(torch.backends.cudnn.enabled)
        print(torch.cuda.is_available())
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        print("lr: 0.005, 0.1/50000")
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 0.1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 50000



        ### Replay Buffer
        self.replay_buffer_size = 30  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 30  # Number of game moves to keep for every batch element
        print("unroll steps: 30")
        self.td_steps = 5  # Number of steps in the future to take into account for calculating the target value
        print("td steps: 5")
        print("per true, alpha 0.5")
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False
        # self.reanalyse_on_gpu = torch.cuda.is_available()



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        # self.env = gym.make("CartPole-v1")
        # self.env = SolitaireWorldEnv(render_mode="human")
        self.env = SolitaireWorldEnv()
        if seed is not None:
            self.env.seed(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, _ = self.env.step(action)
        # obs_out = self.get_one_d_obs(observation)
        obs_out = self.get_three_d_obs(observation)
       
        return obs_out, reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        # return list(range(2))
        return self.env.current_legal_actions
        # return list(range(548))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation = self.env.reset()
        # obs_out = self.get_one_d_obs(observation)
        obs_out = self.get_three_d_obs(observation)
        return obs_out

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def suits_func(self, item):
        if item == 53:
            return -1
        elif item == 0:
            return 0
        elif item / 13 <= 1:
            return 1
        elif item / 26 <= 1:
            return 2
        elif item / 39 <= 1:
            return 3
        else:
            return 4

    def card_num_func(self, item):
        if item == 0:
            return 0
        
        if item % 13 == 0:
            return 13
        else:
            return item % 13

    def get_one_d_obs(self, obs):
        suits = obs['suits'].flatten()
        suits_suits = np.zeros(4,dtype=np.int8)
        if suits[0] > 0:
            suits_suits[0] = 1
        if suits[1] > 0:
            suits_suits[1] = 2
        if suits[2] > 0:
            suits_suits[2] = 3
        if suits[3] > 0:
            suits_suits[3] = 4
        decks = obs['decks_actual'].flatten()
        decks_suits = np.vectorize(self.suits_func)(decks)
        decks_card_num = np.vectorize(self.card_num_func)(decks)
        piles = obs['piles'].flatten()
        piles_suits = np.vectorize(self.suits_func)(piles)
        piles_card_num = np.vectorize(self.card_num_func)(piles)
        piles_behind = (obs['piles_behind'] != 0).sum(0)

        obs_arr = np.concatenate((np.array([obs["deck_position"]]),suits,suits_suits,decks_suits,decks_card_num,piles_suits,piles_card_num,piles_behind))

        return numpy.array([[obs_arr]])

    def get_three_d_obs(self, obs):
        deck_num = obs["deck_position"]

        decks_me = obs['decks'].flatten()
        top_deck_rows = decks_me[3:].reshape(3,7)

        suits_me = obs['suits'].flatten()
        #retrieve original card numbers from suits
        #since they are encoded only 0-13, with 0
        #being empty and 1-13 being card num, but
        #diamond four is 17, but encoded as 4 in the
        #index[1] aka index 2
        if suits_me[1] > 0:
            suits_me[1] = 13 + suits_me[1]
        if suits_me[2] > 0:
            suits_me[2] = (13*2) + suits_me[2]
        if suits_me[3] > 0:
            suits_me[3] = (13*3) + suits_me[3]
        middle_deck_suit_row = numpy.concatenate((decks_me[:3],suits_me), axis=None).reshape(1,7)

        piles_me = obs['piles']
        piles_behind_me = obs['piles_behind']
        
        full_array = np.vstack((top_deck_rows,middle_deck_suit_row,piles_me,piles_behind_me))
        
        aces_channel = np.zeros_like(full_array, dtype=np.int8)
        twos_channel = np.zeros_like(full_array, dtype=np.int8)
        threes_channel = np.zeros_like(full_array, dtype=np.int8)
        fours_channel = np.zeros_like(full_array, dtype=np.int8)
        fives_channel = np.zeros_like(full_array, dtype=np.int8)
        sixes_channel = np.zeros_like(full_array, dtype=np.int8)
        sevens_channel = np.zeros_like(full_array, dtype=np.int8)
        eights_channel = np.zeros_like(full_array, dtype=np.int8)
        nines_channel = np.zeros_like(full_array, dtype=np.int8)
        tens_channel = np.zeros_like(full_array, dtype=np.int8)
        jacks_channel = np.zeros_like(full_array, dtype=np.int8)
        queens_channel = np.zeros_like(full_array, dtype=np.int8)
        kings_channel = np.zeros_like(full_array, dtype=np.int8)
        hearts_channel = np.zeros_like(full_array, dtype=np.int8)
        diamonds_channel = np.zeros_like(full_array, dtype=np.int8)
        spades_channel = np.zeros_like(full_array, dtype=np.int8)
        clubs_channel = np.zeros_like(full_array, dtype=np.int8)
        empty_channel = np.zeros_like(full_array, dtype=np.int8)
        unknown_channel = np.zeros_like(full_array, dtype=np.int8)
        deck_num_channel = np.zeros_like(full_array, dtype=np.int8)



        aces_channel[(full_array == 1) | (full_array == 14) | (full_array == 27) | (full_array == 40)] = 1
        twos_channel[(full_array == 2) | (full_array == 15) | (full_array == 28) | (full_array == 41)] = 1
        threes_channel[(full_array == 3) | (full_array == 16) | (full_array == 29) | (full_array == 42)] = 1
        fours_channel[(full_array == 4) | (full_array == 17) | (full_array == 30) | (full_array == 43)] = 1
        fives_channel[(full_array == 5) | (full_array == 18) | (full_array == 31) | (full_array == 44)] = 1
        sixes_channel[(full_array == 6) | (full_array == 19) | (full_array == 32) | (full_array == 45)] = 1
        sevens_channel[(full_array == 7) | (full_array == 20) | (full_array == 33) | (full_array == 46)] = 1
        eights_channel[(full_array == 8) | (full_array == 21) | (full_array == 34) | (full_array == 47)] = 1
        nines_channel[(full_array == 9) | (full_array == 22) | (full_array == 35) | (full_array == 48)] = 1
        tens_channel[(full_array == 10) | (full_array == 23) | (full_array == 36) | (full_array == 49)] = 1
        jacks_channel[(full_array == 11) | (full_array == 24) | (full_array == 37) | (full_array == 50)] = 1
        queens_channel[(full_array == 12) | (full_array == 25) | (full_array == 38) | (full_array == 51)] = 1
        kings_channel[(full_array == 13) | (full_array == 26) | (full_array == 39) | (full_array == 52)] = 1

        hearts_channel[(full_array >= 1) & (full_array <= 13)] = 1
        diamonds_channel[(full_array >= 14) & (full_array <= 26)] = 1
        spades_channel[(full_array >= 27) & (full_array <= 39)] = 1
        clubs_channel[(full_array >= 40) & (full_array <= 52)] = 1

        empty_channel[full_array == 0] = 1
        unknown_channel[full_array == 53] = 1

        if deck_num < 7:
            deck_num_channel[0, deck_num] = 1
        else:
            deck_num_channel[22, 6] = 1


        ones_cnn_channel = np.ones_like(full_array, dtype=np.int8)


        obs_me = np.array([aces_channel,twos_channel,threes_channel,
                fours_channel,fives_channel,sixes_channel,
                sevens_channel,eights_channel,nines_channel,
                tens_channel,jacks_channel,queens_channel,
                kings_channel,hearts_channel,diamonds_channel,
                spades_channel,clubs_channel,empty_channel,
                unknown_channel,deck_num_channel,ones_cnn_channel])
        return obs_me

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return str(action_number)
        # actions = {
        #     0: "Push cart to the left",
        #     1: "Push cart to the right",
        # }
        # return f"{action_number}. {actions[action_number]}"
