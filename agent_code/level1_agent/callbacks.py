import os
import pickle
import random
from sklearn.ensemble import RandomForestRegressor as RFR

from .state_to_features import state_to_features as stf


import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT'] # level1 agent does not do bombs



def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if self.train or not os.path.isfile("saved_model.sav"):
        self.logger.info("Setting up model from scratch.")
        self.model = RFR()
        self.isFit = False # initial guess??
    else:
        self.logger.info("Loading model from saved state.")
        with open("saved_model.sav", "rb") as file:
            self.model = pickle.load(file)
            self.isFit = True




def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    X = stf(game_state).reshape(1,-1)

    if self.train and random.random() < self.epsilon or not self.isFit: # inititally choose randomly???
        self.logger.debug("Random action according to eps greedy alg.")
        action = np.random.choice(ACTIONS) 
    
    else:
        self.logger.debug("Choosing action according to current model.")
        a_prob = self.model.predict(X)[0]
        #a_prob = a_prob/np.sum(a_prob)
        #action = np.random.choice(ACTIONS,p=a_prob)
        action = np.argmax(a_prob)
        action = ACTIONS[action]

    return action













