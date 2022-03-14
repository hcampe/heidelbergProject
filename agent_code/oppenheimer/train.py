from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

#Custom Events
#...

existing_model="oppenheimer.pt"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    if(len(self.transitions)==0): return

    _, oldscore, old_bombs_left, (old_x, old_y) = old_game_state['self']
    _, new_score, new_bombs_left, (new_x, new_y) = new_game_state['self']
    old_coins = old_game_state['coins']
    new_coins = new_game_state['coins']

    #calculate state indices
    if(len(old_coins)>0):
        oldxdist = old_coins[0][0]-old_x
        oldydist = old_coins[0][1]-old_y
    else:
        oldxdist = 0
        oldydist = 0

    if(len(new_coins)>0):
        newxdist = new_coins[0][0]-new_x
        newydist = new_coins[0][1]-new_y
    else:
        newxdist = 0
        newydist = 0

    oldxodd = old_x%2
    oldyodd = old_y%2
    newxodd = new_x%2
    newyodd = new_y%2

    if(old_bombs_left): oldbomb = 1
    else: oldbomb = 0
    if(new_bombs_left): newbomb = 1
    else: newbomb = 0

    oldstatenumber = ((oldxdist+14)+29*(oldydist+14))+841*oldxodd+2*841*oldyodd + 4*841*oldbomb # row index corresponding to state, 841=29*29 rel states
    newstatenumber = ((newxdist+14)+29*(newydist+14))+841*newxodd+2*841*newyodd + 4*841*newbomb# row index corresponding to state, 


    #learn here
    alpha = 0.5 # learning rate (hyperparameter)
    gamma = 0.9 # discount (hyperparameter)

    r = reward_from_events(self, events) # reward for this timestep
    a = ACTIONS.index(self_action) # column index corresponding to action taken in this timestep

    #prints for debugging
    #print(f"action: {self_action}, actionindex: {a}, reward: {r}, old model score: {self.model[oldstatenumber]}, s: {oldstatenumber} ")
    #print(f"new state: {self.model[newstatenumber,:]}, s: {newstatenumber}")

    self.model[oldstatenumber,a]+=alpha*(r+gamma*max(self.model[newstatenumber,:])-self.model[oldstatenumber,a]) # Q-learning

    #print(f"updated model score: {self.model[oldstatenumber]}")

    # Store the model
    with open(existing_model, "wb") as file:
        pickle.dump(self.model, file)

    self.transitions.append(Transition(old_game_state, self_action, new_game_state, reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(last_game_state, last_action, None, reward_from_events(self, events)))


    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE) # delete transitions of previous game

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        #e.KILLED_SELF: -10,
        e.INVALID_ACTION: -100,
        e.WAITED: -100,
        e.BOMB_DROPPED: -100,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
