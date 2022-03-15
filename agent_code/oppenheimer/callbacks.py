import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
np.set_printoptions(threshold=np.inf) #verbose print of matrix

existing_model = "oppenheimer.pt"

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
    train_existing_model = 1 # continue training existing model
    
    if (self.train and not train_existing_model) or not os.path.isfile(existing_model):
        self.logger.info("Setting up model from scratch.")
        weights = np.zeros((2*4*29*29,len(ACTIONS)))
        self.model = weights
    else:
        self.logger.info("Loading model from saved state.")
        with open(existing_model, "rb") as file:
            self.model = pickle.load(file)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # get state and coin information
    _, score, bombs_left, (x, y) = game_state['self']
    coins = game_state['coins']

    #exploration
    random_prob = .1 
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    #exploitation
    self.logger.debug("Querying model for action.")

    #calculate state index
    if(len(game_state['coins'])>0):
        #determine which coin is closest to agent
        coindist = coins-np.array((x,y))
        coinindex = np.argmin(np.linalg.norm(coindist,axis=1))
        xdist = coindist[coinindex][0]
        ydist = coindist[coinindex][1]
    else:
        xdist = 0
        ydist = 0
        
    xodd = x%2
    yodd = y%2

    if(bombs_left): oldbomb = 1
    else: oldbomb = 0

    statenumber = ((xdist+14)+29*(ydist+14))+841*xodd+2*841*yodd + 4*841*oldbomb

    #Speed up learning by picking random move if multiple moves have same Q Value instead of first in ACTIONS list.
    permut = np.random.permutation(range(6))
    permutatedstate = [self.model[statenumber,i] for i in permut]
    index = np.argmax(permutatedstate)

    return ACTIONS[permut[index]]

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
