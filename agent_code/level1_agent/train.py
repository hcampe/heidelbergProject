from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .state_to_features  import state_to_features as stf
from .rewards import reward_from_events as rfe

import random

import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400  # keep only ... last transitions // full game?
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 20
N_STEP = 5

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

EPSILON_START = 0.9
EPSILON_MIN = 0.05
EXPLORATION_DECAY = 0.96
GAMMA = 0.95 # do modify
LEARNING_RATE = 0.01

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT'] # level1 agent does not do bombs



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.epsilon = EPSILON_START

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.count = 0


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

    immediate_reward = rfe(self,events)
    self_action_int = ACTIONS.index(self_action)
 
    self.transitions.append(Transition(stf(old_game_state), self_action_int, stf(new_game_state), rfe(self, events)))
        
    self.count += 1

    if len(self.transitions) < BATCH_SIZE:
        return

    if self.count%20 == 0: # only every 20 steps of game ??
        batch_sample = random.sample(self.transitions, BATCH_SIZE)
        X = np.zeros((BATCH_SIZE,len(stf(old_game_state))))
        Y = np.zeros((BATCH_SIZE,len(ACTIONS)))
        for i, (old_state, action, new_state, immed_reward) in enumerate(batch_sample):
            Y[i,action] = immed_reward
            if new_state is not None:
                if self.isFit:
                    Y[i,action] += GAMMA * np.amax(self.model.predict(new_state.reshape(1,-1))) # all other Y=0??
            if old_state is not None:          
                X[i] = old_state
                #print(self.transitions[i])    
        self.model.fit(X,Y)
        self.isFit = True
        self.epsilon *= EXPLORATION_DECAY
        self.epsilon = max(self.epsilon,EPSILON_MIN)
        



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
    last_action = ACTIONS.index(last_action)
    self.transitions.append(Transition(stf(last_game_state), last_action, None, rfe(self, events)))

    # update Q with full game memory:
    X = np.zeros((TRANSITION_HISTORY_SIZE,len(stf(last_game_state))))
    Y = np.zeros((TRANSITION_HISTORY_SIZE,len(ACTIONS)))
    n = N_STEP
    for t, (old_state, action, new_state, immed_reward) in enumerate(self.transitions):
        if t + n >= TRANSITION_HISTORY_SIZE:
            n -= 1
            #n = min(np.abs(n),0)
        for tprime in range(n+1):
            tt = int(t + tprime)
            #print(action)
            Y[t,action] += GAMMA**(tprime)*self.transitions[tt][3]
        if self.transitions[t+n][2] is not None:
            Y[t,action] += GAMMA**n * np.amax(self.model.predict(self.transitions[t+n][0].reshape(1,-1)))
        X[t] = old_state

    self.model.fit(X,Y)
    self.epsilon *= EXPLORATION_DECAY
    self.epsilon = max(self.epsilon,EPSILON_MIN)

    # Store the model
    with open("saved_model.sav", "wb") as file:
        pickle.dump(self.model, file)


