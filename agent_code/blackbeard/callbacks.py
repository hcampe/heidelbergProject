import os
import pickle
import random

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTDICT = { ACTIONS[i] : i for i in range(len(ACTIONS)) }

N_BATCH = 10
GAMMA = .6
ALPHA = .1

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
        beta = np.zeros((6, 5)) # #actions x #features
        self.model = BlackbeardsModel(beta)
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
    # Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return ACTIONS[self.model.hardChoice(game_state)]


def walls_around(game_state: dict) -> np.array:
    """
    this should tell the agent where there are walls i.e. where in which directions
    it is not possible to move

    :param game_state:  A dictionary describing the current game board.
    :return: np.array [UP,RIGHT,DOWN,LEFT] with entries 0 if no wall and 1 if wall
    """
    x, y  = game_state['self'][-1] # agent’s position
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


class BlackbeardsModel:
    """
    blackbeard’s model to find a policy as to where to move in a given state of the
    game. uses TD QRL and linear regression for learning

    it is necessary to implement this as a class since we want to be able to save
    the model. I would not know any other way.
    """
    def __init__(self, beta):
        """beta represents the linear regression, for each action a,
        beta[a,:] is the weight vector"""
        self.beta = beta

    def hardChoice(self, game_state: dict) -> int:
        """
        returns a hard choice on the action to take, choice = argmax policy

        :game_state:  A dictionary describing the current game board.
        :returns:     the index of the action to take
        """

        features = state_to_features(game_state)
        Q = np.einsum('i,ji->j', features, self.beta)

        return np.argmax(Q)

    def train(self, features, actions, rewards):
        """
        trains the model using batch gradient descent
        
        :features:      the features as seen by the agent in every step
        :actions:       actions took by the agent in every step
        :rewards:       rewards handed out to the agent in every step
        :returns:       0 if successfull?
        """
        
        # calculate Q / Y
        Qold = np.einsum('ki,ji->kj', features, self.beta)
        Q = rewards[:-1] + GAMMA*np.max(Qold[1:,:], axis=1)
        #print(f"Q = {Q}")
        #print(f"features[:5,:] = {features[:5,:]}")
        print(f"actions[:5] = {actions[:5]}")
        #print(f"rewards[:5] = {rewards[:5]}")

        for a in range(len(ACTIONS)):
            X = features[:-1][actions[:-1]==a]
            Y = Q[actions[:-1]==a]
            Nbatch = len(Y)
            print(f"for action {a}, Nbatch = {Nbatch}")
            if Nbatch < 10: continue
            print(f"X = {X}, Y = {Y}")
            self.beta[a,:] += ALPHA/Nbatch * X.T @ (Y - X@self.beta[a,:])
            print(f"update! beta[{a},:] = {self.beta[a,:]}")

        return 0

    def trainAction(self, action, oldState, newState, rewards):
        X = oldState
        Y = rewards + GAMMA*np.max(np.einsum('ij,j->i', newState, self.beta[action,:]))
        self.beta[action,:] += ALPHA/N_BATCH * X.T @ (Y - X@self.beta[action,:])
        print(f"updated! beta[{action},:] = {self.beta[action]}")
        
    def getBeta(self):
        return self.beta

