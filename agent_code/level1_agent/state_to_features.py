import numpy as np



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

    own_x, own_y = game_state['self'][3]

    field = game_state['field']

    coins = game_state['coins']
    coins = np.asarray(coins)

    features = np.zeros(4)

    for i, (x,y) in enumerate([(own_x,own_y + 1), (own_x + 1, own_y), (own_x, own_y - 1), (own_x - 1, own_y)]):
        if field[x,y] == -1:
            features[i] = -1

        else:
            d = np.sum(np.abs(coins - np.array((x,y))),axis=1).min()

            features[i] = d


    return features

