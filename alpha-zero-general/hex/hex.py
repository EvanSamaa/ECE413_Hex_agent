import numpy as np
import random

EMPTY = -1

edge_nodes = [(-1, -2), (-3, -4)]

class Board:
    def __init__(self, size):
        self.size = size
        self.turn = 0
        self.board = np.int8([EMPTY] * (size ** 2))

        self.edges = { i: get_edges(i, size) for i in range(size**2) }
        self.edges[edge_nodes[0][0]] = [i for i in range(size)]
        self.edges[edge_nodes[0][1]] = [i + (size - 1) * size for i in range(size)]
        self.edges[edge_nodes[1][0]] = [i * size for i in range(size)]
        self.edges[edge_nodes[1][1]] = [i * size + (size - 1) for i in range(size)]

    def __str__(self):
        s = ''
        for i in range(self.size):
            s += ' ' * i
            for j in range(self.size):
                s += (['-', 'x', 'o'])[self.board[i * self.size + j] + 1] + ' '
            s += '\n'
        return s

    def actions(self):
        return [i for i in range(self.size ** 2) if self.board[i] == EMPTY]

    def do(self, action):
        self.board[action] = self.turn
        self.turn = 1 - self.turn

    def eval(self):
        for turn, (start, end) in enumerate(edge_nodes):
            if find_path(self.board, self.edges, start, end, turn):
                return turn
        return None

    def serialize(self):
        if self.turn == 0:
            return self.board.copy()
        else:
            transposed = np.reshape(np.transpose(np.reshape(self.board, (self.size, self.size))), (self.size ** 2,))
            return np.choose(transposed + 1, np.int8([EMPTY, 1, 0]))

def get_edges(i, n):
    y = i // n
    x = i % n
    nbrs = [ r * n + c for r, c in [(y-1, x), (y-1, x+1), (y, x-1), (y, x+1), (y+1, x-1), (y+1, x)] if 0 <= r < n and 0 <= c < n ]

    if y == 0:
        nbrs.append(edge_nodes[0][0])
    if y == n - 1:
        nbrs.append(edge_nodes[0][1])
    if x == 0:
        nbrs.append(edge_nodes[1][0])
    if x == n - 1:
        nbrs.append(edge_nodes[1][1])

    return nbrs

def find_path(nodes, edges, begin, end, val):
    todo = [begin]
    visited = set()
    while len(todo):
        nxt = todo.pop()
        visited.add(nxt)
        for nbr in edges[nxt]:
            if nbr == end:
                return True
            if nodes[nbr] == val and nbr not in visited:
                todo.append(nbr)
    return False

def gen_samples(size, n_games):
    samples = []

    for _ in range(n_games):
        board = Board(size)
        val = None
        game_samples = []
        while val is None:
            act = random.choice(board.actions())
            board.do(act)
            val = board.eval()
            # serialization is noramlized, so always 0 or -1
            game_samples.append((board.serialize(), 0 if val is None else -1))
        samples.extend(game_samples[-2:])

    return samples

if __name__ == '__main__':
    N = 7
    n_games = 1000000
    bs = 1000
    samples = []
    for i in range(n_games // bs):
        samples.extend(gen_samples(N, bs))
        print('{} / {}'.format((i+1) * bs, n_games))
    X = np.int8([d for d, _ in samples])
    Y = np.int8([l for _, l in samples])

    filename = './data/hex/eval{}'.format(N)

    print('Saving to {}'.format(filename))
    np.savez(filename, X=X, Y=Y)


