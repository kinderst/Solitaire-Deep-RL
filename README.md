# Solitaire Deep Reinforcement Learning

The repo contains different approaches used to try to (better) solve the classic game of 3-draw Klondike Solitaire

## Environment

The environment was created as an OpenAI Gym environment, and contains all the logic needed to play 3-draw Klondike solitaire, as well as view it in Pygame with actual cards in the classic solitaire arrangement.

![Solitaire Screen](/img/solitaire-screen.PNG)

### State Space

The state space of the environment is defined by the cards that make up the various components of the game:
- Deck, is the deck you draw from i.e. with 3 cards
- Suits, are the goal set you are trying to fill with all Kings
- Piles, are the piles you form of cards, which lie above the
- Piles Behind, which are the upside down cards underneath piles

Cards are defined by 1-52, and their position in the array define their location in the game

### Action Space

The action space of the environment is defined as such:

- action 0 is tapping deck
- action 1 is tapping active deck card, attempting to send to suit
- actions 2-8 are trying to move top deck card to one of 7 other piles
- actions 9-15 are trying to move suit's 0 (hearts) to one of 7 other piles
- actions 16-22 are trying to move suit's 1 (diamonds) to one of 7 other piles
- actions 23-29 are trying to move suit's 2 (spades) to one of 7 other piles
- actions 30-36 are trying to move suit's 3 (clubs) to one of 7 other piles
- actions 37-43 are trying to move bottom-most card in one of 7 piles to their given suits
- actions 44 to 547 are pile to pile moves, for each 12 cards (no need to move ace), you can move them to 6 other piles, for each of the 7 piles

## Deep Reinforcement Learning Approaches

Several Deep Reinforcement Learning approaches have been applied to Solitaire. Work for all of them is still in progress

### Proximal Policy Optimization

The first naive attempt was using [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347), which is a policy gradient method, and should generally learn good actions from bad. It shouldn't be expected to find complicated paths -- unless by random chance -- as it will take a lot of learning to build into the algorithm the importance of the horizon in Solitaire. This is because for it to understand that playing one suboptimal move (such as removing a card from the Suits and putting it into a Pile) would be better than a currently better move, like putting a card from the deck into a pile, would be very hard and take a lot of time, and even then we wouldn't be guaranteed to find it.

The implementation is based off the Keras approach. Here is a video of a typical PPO agent under similar hyperparameters and reward structure as defined in the scripts:

https://user-images.githubusercontent.com/11727102/220267610-60eb639d-9028-4df4-a042-b435411abd35.mp4


First of all, not all games of solitaire are able to be solved. However, it's clear that the agent is not optimal, but does a sufficient job. It will often get stuck doing some action that is positively rewarding, but not as good as others that may open up other cards.

The best agent was able to win about 10% of games, which is slightly below human-level. More detailed results can be seen in the notebook script, or experimented on there. I believe it is certainly possible to get far above the 10% win rate with PPO, but will take experimentation with hyperparameter and reward space, along with how you define your action space.

PPO should really only be considered a baseline here, as a policy gradient method won't be able to find complex paths needed to open up more cards in the future. Compared to something like tree-based search, PPO's horizon is far too shallow, especially given the complexity of the state space, to form a state-of-the-art policy.

### Deep Q Learning

The next approach was [Deep Q Learning](https://arxiv.org/abs/1312.5602), a value-based approach hoping to leverage the fact that some state-action pairs may be better than others long term. Unfortunately, the horizon problem is stil an issue here as well, as state-actions pair have the horizon implicitly baked into the q-value and it probably won't have the ability to see far enough while generalizing the state-space.

For this implementation, I used PyTorch instead, but staying with a fully connected network (though there is an implementation for a ResNet using the card numbers and suits as channels in the notebook script). Results can be seen in the graph of the rewards over episodes below. The positive reward spikes represent solutions to a solitaire game, as it gets rewarded more when it wins. Sometimes there are just no moves, and it gets penalized for cycling through the deck or moving pile cards uselessly, so there are also highly negative spikes as well

![DQN Results](/img/dqn-learning.PNG)

DQN seems to be slightly better than PPO, winning about 15% of the time. This would align with expectations, as more finely calculating precise state-action pairs would probably be better than just usually picking good actions like PPO, though certainly some randomness is needed to break from repetitions. That is also why the episilon greedy parameter is kept during action selection at inference time as well, as that was how it was optimal when it was trained on, and because otherwise it suffers from repetition

Below is a video of the DQN agent solving a game of Solitare

https://user-images.githubusercontent.com/11727102/220307302-31844891-a4ed-45e7-b895-f8857b622354.mp4

You can see that the agent is struggling going in circles, until it moves a 9 that was under a 10 to a different 10, allowing that 10 to be added to the suits, and unlocking a new card, thus solving the puzzle (around 57 second mark). This actually probably just happened by chance as that action was available for some moves, and it's unlikely the state-space changing slightly like it did would cause that, and therefore was probably one of the epsilon random actions. Under my experiments, lowing episilon too low actually really hurt the win rate, probably for reasons like this.

### MuZero and EfficientZero

From the amazing results on chess and other board games that AlphaZero had, it is clear that MCTS is probably going to be state-of-the-art for Solitaire as well. Intuitively it makes sense, as planning is critical to open up more cards, and that can be hard to capture through simply state-action pairs.

[MuZero](https://arxiv.org/abs/1911.08265) and [EfficientZero](https://arxiv.org/abs/2111.00210) both rely on learned models of the environment. Unlike Chess, no perfect simulator for Solitaire exists, as it is unknown what the upside-down cards are. While some engineering efforts could be made to create AlphaZero with this small amount of non-determinism added in, MuZero is gaining popularity in the field as the learned model makes it far more applicable to more environments. And, surprisingly, the learned model hardly suffers compared to the AlphaZero's, as MuZero was similar in elo in Chess to AlphaZero, but the authors explain more in detail [here](https://arxiv.org/abs/1911.08265)

The idea here is to apply the MuZero algorithm to the chess environment. This is still a work in progress as results have been sub optimal, but current progress is contained in the folder
