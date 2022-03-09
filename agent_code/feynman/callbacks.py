import os
import pickle
import random

import numpy as np
from sklearn.ensemble import RandomForestRegressor

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTDICT = { ACTIONS[i] : i for i in range(len(ACTIONS)) }

GAMMA = .9
ALPHA = .3

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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
        beta = np.zeros((6, 5)) # #actions x #features
        self.model = Model(beta)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model.policy(game_state))
    return ACTIONS[self.model.hardChoice(game_state)]


def walls_around(game_state: dict) -> np.array:
    """
    this should tell the agent where there are walls i.e. where in which directions
    it is not possible to move

    :param game_state:  A dictionary describing the current game board.
    :return: np.array [UP,RIGHT,DOWN,LEFT] with entries 0 if no wall and 1 if wall
    """
    x, y = game_state['self'][-1] # agents position
    field = game_state['field']

    walls = -np.array([field[x,y+1], field[x,y-1], field[x-1,y], field[x+1,y]])
    
    return walls


def compass_to_closest_coin(game_state: dict) -> float:
    """
    tells the agent the direction of the closest coin.

    :param game_state:  A dictionary describing the current game board.
    :return: a float that represents the angle (rad) of the direction to the
        closest coin, where the ‘north’ points UP
    """
    x, y = game_state['self'][-1] # agents position
    coins = np.array(game_state['coins'])
    distToCoins = (coins[:,0] - x)**2 + (coins[:,1] - y)**2
    closest = np.argmin(distToCoins)
    xC, yC = coins[closest]

    if yC==y:
        phi = np.pi/2 if xC > x else 1.5*np.pi
    else:
        phi = np.arctan((xC - x)/(yC - y))
    return phi

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    The features will b normalised and centralised i.e. in [-.5, .5]

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

#    # For example, you could construct several channels of equal shape, ...
#    channels = []
#    channels.append(...)
#    # concatenate them as a feature tensor (they must have the same shape), ...
#    stacked_channels = np.stack(channels)
#    # and return them as a vector

#    return stacked_channels.reshape(-1)

    features = np.array([])
    # idea 1: return where there are walls around
    features = np.append(features, walls_around(game_state))
    # idea 2: return direction of closest coin
    features = np.append(features, [compass_to_closest_coin(game_state)/(2*np.pi)])

    return features - .5


class Model:
    """
    Feynman’s model to find a policy as to where to move in a given state of the
    game. uses TD QRL and a regression forest for learning

    it is necessary to implement this as a class since we want to be able to save
    the model. I would not know any other way.
    """
    def __init__(self, beta):
        self.forests = [RandomForestRegressor() for a in ACTIONS]
        self.alreadyFit = [False for a in ACTIONS] # stores whether the forest
                                            # for action a has already been fit

    def hardChoice(self, features) -> int:
        """
        returns a hard choice on the action to take, choice = argmax policy

        :game_state:  A dictionary describing the current game board.
        :returns:     the index of the action to take
        """
    
        Qvals = [0 if not self.alreadyFit[a] else self.forests[a].predict(features) \
                                                   for a in range(len(ACTIONS))]
        return np.argmax(Qvals)

    def train(self, features, actions, rewards):
        """
        trains the model using the random forest and TD Q RL
        
        :features:      the features as seen by the agent in every step
        :actions:       actions took by the agent in every step
        :rewards:       rewards handed out to the agent in every step
        :returns:       0 if successfull?
        """
        
        # get the Y values (TD Q RL)
        Y = rewards[:-1] + GAMMA * np.max([0 if not self.alreadyFit[a] else \
                                                self.forests[a].predict(features[1:,:]) \
                                                            for a in range(len(ACTIONS))])
        # fit the forests to the new data
        for a in range(len(ACTIONS)):
            X = features[:-1][actions[:-1]==a]
            if len(X) == 0: continue
            self.forests[a].partial_fit(X,Y[actions[:-1]==a])
            self.alreadyFit[a] = True


        return 0

    def getBeta(self):
        return self.beta

