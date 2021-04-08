import random
from mcts_py import PyHex, PySearch
from net import wrap_for_rust

# Pits player1 against player2. Each player is a function from board state to a valid action
# Returns 1 if player1 wins, else 0
def pit(game, player1, player2, verbose=False):
    player, next_player = player1, player2
    while game.terminal_value() is None:
        if verbose:
            print(game.to_string())

        action = player(game)
        game = game.next_state(action)
        player, next_player = next_player, player
    result = 1 if player is player2 else 0
    if verbose:
        print(game.to_string())
        print('Game over. Player {} wins'.format(2 - result))
    return result

# random_players takes a valid action with equal probability
def random_player(game):
    valids = game.valid_actions()
    return random.choice(valids)

def human_player(game):
    valids = game.valid_actions()
    action = None
    while action not in valids:
        if action is not None:
            print('Valid actions are: ', valids)
        try:
            action = int(input('Please input a valid action:'))
        except ValueError:
            action = -1
    return action

def create_mcts_player(net, mcts_iterations=1000):
    search = PySearch()
    net = wrap_for_rust(net)
    def player(game):
        return search.get_action(game, mcts_iterations, net)
    return player

def create_shallow_player(net):
    net = wrap_for_rust(net)
    def player(game):
        valids = game.valid_actions()
        _, policy = net(game.to_array())
        for i in range(len(policy)):
            if i not in valids:
                policy[i] = 0
        options = sorted(enumerate(policy), key=lambda e: -e[1])
        print(options[:4])
        return options[0][0]
    return player

if __name__ == '__main__':
    player1 = human_player
    player2 = create_mcts_player(None)
    pit(PyHex(5), player1, player2, verbose=True)
