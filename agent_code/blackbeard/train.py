from collections import namedtuple, deque

import pickle
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features, ACTDICT, N_BATCH
import settings as s

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # here: set up np arrays to store the features and rewards
    # for every step of the current game
    # they are stored in different np arrays dep. on the action
    self.oldState = [np.empty((N_BATCH, 5))] * 6
    self.newState = [np.empty((N_BATCH, 5))] * 6
    self.rewards  = [np.empty(N_BATCH)] * 6
    self.numUses  = [0] * 6 # counts how often an action has been performed


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self:            This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state:  The state that was passed to the last call of `act`.
    :param self_action:     The action that you took.
    :param new_game_state:  The state the agent is in now.
    :param events:          The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    self.logger.debug(f"events: {events}, old game state: {old_game_state}, new game state: {new_game_state}")

    if new_game_state['step'] < 2:
        return

    a = ACTDICT[self_action]
    self.oldState[a][self.numUses[a],:] = state_to_features(old_game_state)
    self.newState[a][self.numUses[a],:] = state_to_features(new_game_state)
    self.rewards[a][self.numUses[a]] = reward_from_events(self, events)
    self.numUses[a] += 1

    if self.numUses[a] == N_BATCH: # if enough data has been collected, train the model
        self.model.trainAction(a, self.oldState[a], self.newState[a], self.rewards[a])
        # reset the arrays:
        self.oldState[a] = np.empty((N_BATCH,5))
        self.newState[a] = np.empty((N_BATCH,5))
        self.rewards[a]  = np.empty(N_BATCH)
        self.numUses[a]      = 0


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

    # enter final stuff into the arrs
#    i = new_game_state['step']
#    self.oldState[i,:] = state_to_features(old_game_state)
#    self.gameActions[i]    = ACTDICT[self_action]
#    self.rewards[i]    = reward_from_events(self, events)


    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    # reset the arrs
    self.oldState = [np.empty((N_BATCH, 5))] * 6
    self.newState = [np.empty((N_BATCH, 5))] * 6
    self.rewards  = [np.empty(N_BATCH)] * 6
    self.numUses  = [0] * 6 # counts how often an action has been performed




def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    #game_rewards = {
    #    e.COIN_COLLECTED: 5,
    #    e.INVALID_ACTION: -1,
    #    e.KILLED_SELF: -1,
    #    e.GOT_KILLED: -1,
    #}
    game_rewards = { e.MOVED_LEFT: 100 }
    reward_sum = -5
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
