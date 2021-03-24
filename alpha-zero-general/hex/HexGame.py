from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
import numpy as np

EMPTY = 0
edge_nodes = [(1, -1, -2), (-1, -3, -4)] # (player, node1, node2)

class HexGame(Game):
    square_content = {
        -1: "O",
        +0: "-",
        +1: "X"
    }

    @staticmethod
    def getSquarePiece(piece):
        return HexGame.square_content[piece]

    def __init__(self, n):
        self.n = n
        # Initialize the connection graph
        self.edges = { i: get_edges(i, n) for i in range(n**2) }
        self.edges[edge_nodes[0][1]] = [i for i in range(n)]
        self.edges[edge_nodes[0][2]] = [i + (n - 1) * n for i in range(n)]
        self.edges[edge_nodes[1][1]] = [i * n for i in range(n)]
        self.edges[edge_nodes[1][2]] = [i * n + (n - 1) for i in range(n)]

    def getInitBoard(self):
        # return initial board (numpy board)
        return np.zeros((self.n, self.n))

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b[move] = player
        return (b, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        moves = np.zeros(board.shape)
        moves[board == 0] = 1
        return np.reshape(moves, (-1,))

    def getGameEnded(self, board, current_player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        flat_board = board.reshape((-1,))
        for (player, start, end) in edge_nodes:
            if find_path(flat_board, self.edges, start, end, player):
                return player * current_player
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        # Transpose the board for player 1, they want to connect left and right
        return board if player == 1 else -board.T

    def getSymmetries(self, board, pi):
        # TODO: enable symmetries?
        # mirror, rotational
        # assert(len(pi) == self.n**2+1)  # 1 for pass
        # pi_board = np.reshape(pi[:-1], (self.n, self.n))
        # l = []

        # for i in range(1, 5):
        #     for j in [True, False]:
        #         newB = np.rot90(board, i)
        #         newPi = np.rot90(pi_board, i)
        #         if j:
        #             newB = np.fliplr(newB)
        #             newPi = np.fliplr(newPi)
        #         l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        # return l
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        s = ''
        for i in range(self.n):
            s += ' ' * i
            for j in range(self.n):
                s += HexGame.square_content[int(board[i,j])] + ' '
            s += '\n'
        return s

def get_edges(i, n):
    y = i // n
    x = i % n
    nbrs = [ r * n + c for r, c in [(y-1, x), (y-1, x+1), (y, x-1), (y, x+1), (y+1, x-1), (y+1, x)] if 0 <= r < n and 0 <= c < n ]

    if y == 0:
        nbrs.append(edge_nodes[0][1])
    if y == n - 1:
        nbrs.append(edge_nodes[0][2])
    if x == 0:
        nbrs.append(edge_nodes[1][1])
    if x == n - 1:
        nbrs.append(edge_nodes[1][2])

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