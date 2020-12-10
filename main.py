
import numpy as np


class eightTilePuzzle():
    def __init__(self,initial):
        self.state = initial
        self.goal = np.array([[1, 2, 3], [4,5,6], [7,8,0]])
        self.path = []

        def _find_empty_tile(self):
            '''
            Finds the location of the current state empty tile, represented by 0

            :return: a tuple of ints, representing the row index and column index
            '''
            state = self.state
            row, col = np.where(state == 0)
            return row.item(), col.item()

        def _get_legal_moves(self):
            '''
            Checks the legal options on the board. For example, if the empty tile is on the first row,
            then we can't switch with the tile above (because there is no tile above).

            :return: A list of tuples, each one includes a row and column indices representing a tile location we are
                    able to replace from this current state
            '''
            row, col = self._find_empty_tile()
            legal_moves = []
            if row > 0:
                legal_moves.append((row - 1, col))
            if col > 0:
                legal_moves.append((row, col - 1))
            if row < 2:
                legal_moves.append((row + 1, col))
            if col < 2:
                legal_moves.append((row, col + 1))
            return legal_moves

        def _swap(self,move):
            '''
            Creates a new flattened state (vector form) according to the current swap choice

            :param move: a tuple of row and column indices, representing the location we are going to swap with the
                        empty tile
            :return: a flattened array (1d shaped puzzle), after swapping.
            '''
            new_state = self.state.copy()
            val_row, val_col = move
            empty_row,empty_col = self._find_empty_tile()
            swapped_value = new_state[val_row,val_col]
            new_state[empty_row,empty_col] = swapped_value
            new_state[val_row,val_col] = 0

            return new_state.reshape(9)



        def create_possible_future_states(self):
            '''
            :return: a list of arrays, each one representing a possible state we can go to from the current state
            '''
            moves = self._get_legal_moves()
            future_states = []
            for i in range(len(moves)):
                move = moves[i]
                new_state = self._swap(move)
                future_states.append(new_state)

            return future_states

        def move_state(self,new_state):
            '''
            Doesn't return anything, just makes sure we are tracking the progress and changing the current state
            within the class

            :param new_state: The state chosen as the next one

            '''
            self.path.append(self.state)
            self.state = new_state
