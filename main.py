
import numpy as np


class EightTilePuzzle:
    '''
    The board class, keeps tab on the current path, state and goal
    '''
    def __init__(self,initial):
        self.state = initial
        self.goal = np.array([[1, 2, 3], [4,5,6], [7,8,0]])
        self.path = [initial]

    def is_solvable(self):
        '''
        Makes sure the initial board state is solvable
        :return: boolean, true if solvable
        '''
        inv_count = 0

        for i in range(0,3):
            pos = i+1
            for j in range(pos,3):
                if(self.state[j,i] > 0 and self.state[j,i] > self.state[i,j]):
                    inv_count += 1
            return inv_count % 2 == 0

    def is_win(self):
        '''
        Checks if the current state is the goal state
        :return: boolean, true if current state and goal state are completely equal
        '''
        return np.sum(np.equal(self.state,self.goal)) == 9

    def get_path(self):
        return self.path

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
        new_state[val_row,val_col], new_state[empty_row,empty_col] = 0, new_state[val_row,val_col]

        return new_state



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

        self.state = new_state
        self.path.append(self.state)


class Node:
    '''
    Node class for each part of the search tree
    '''
    def __init__(self,state, parent=None, cost=0):
        self.state = state
        self.parent = parent
        self.children = []
        self.cost = cost
        self.alive = True

    def is_root(self):
        return self.parent is None

    def is_alive(self):
        return self.alive

    def kill_node(self):
        '''
        Makes sure we never expand to this node again
        '''
        self.alive = False

    def get_cost(self):
        return self.cost

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def no_return(self):
        '''
        The idea here is that we should never return to the exact state of one of our ancestors. If we want to do that,
        it is better to simply go back to that ancestor and do a different choice, as we prefer less actions. Also,
        allowing this can easily lead to endless loops

        :return: boolean, if one of the node's ancestors are also in this state
        '''
        current_state = self.state
        node = self
        flag = True
        while node.get_parent() is not None:
            node = node.get_parent()
            if np.sum(np.equal(node.state,current_state)) == 9:
                flag = False
                break
        return flag



class SearchTree:
    '''
    Search tree of nodes, follows the current node we are searching.
    Not sure if this class is necessary, but it helps to divide the responsibilities more sensibly
    '''

    def __init__(self, root):
        self.nodes = [Node(root)]
        self.current_node = Node(root)


    def __len__(self):
        return len(self.nodes)

    def add_as_parent_child(self,node):
        '''
        Because we add new nodes every time, we also need to add the as children to their parent
        :param node: the new created node

        Doesn't return anything, simply adds the new node as a child to it's parent. To be honest, not sure this is
        a good way to do this, but whatever.
        '''
        node.get_parent().get_children().append(node)

    def get_current_node(self):
        return self.current_node

    def change_search_node(self,node):
        '''
        Moves the search to a different node (whether to expand or to back track)
        :param node: the new node for the current search
        '''
        self.current_node = node


    def add_node(self, node):
        '''
        Ads a node to the search tree
        :param node: the node to add
        '''
        self.add_as_parent_child(node)
        self.nodes.append(node)

    def examine_addition_to_search(self,state,puzzle,heuristic):
        '''
        We can't simply add a node just because it's a possible move on the board. We need to make sure we don't run
        into infinite loops (see no_return function for Node class)
        :param state: The new state which is a candidate for a new node
        :param puzzle: the board configuration class
        :return: Either the new node for the search process, or None if the state is a bad candidate (creates a return)
        '''

        if heuristic is True:
            state_cost = distance_heuristic(state, puzzle.goal)
        else:
            state_cost = misplaced_cost_Yoav(state, puzzle.goal)

        potential_node = Node(state, self.current_node, state_cost)

        if potential_node.no_return():
            self.add_node(potential_node)
            return potential_node
        else:
            return None

    def try_expand_search(self,puzzle,algo,heuristic):
        '''
        Attempt to proceed to a new child node in the search. At the moment we always choose the child with the least
        cost by some heuristic. If several nodes have the same cost, we will go for the one on the left (also a type of
        heuristic I guess, but that's how numpy.argmin works, so I just left it this way)
        :param puzzle: the board class for current configuration and new state candidates
        :param algo: the algorithm used
        :return: Either the chosen node for the next search expansion, or None if no children are an option (all dead
        nodes, or ones that create a return)
        '''

        open_list = []
        cost_list = []
        if len(self.get_current_node().get_children()) == 0:
            children = puzzle.create_possible_future_states()

            for i in range(len(children)):

                node = self.examine_addition_to_search(children[i], puzzle,heuristic)
                if node is not None:
                    open_list.append(node)
                    cost_list.append(node.get_cost())

            if len(open_list) > 0:
                best_node_index = np.argmin(cost_list)

                self.change_search_node(open_list[best_node_index])

                algo.increase_LB(self.get_current_node().get_cost()+1)
                puzzle.move_state(self.get_current_node().state)

                return self.get_current_node()

        else:
            children = self.get_current_node().get_children()
            for i in range(len(children)):
                child = children[i]
                if child.is_alive():
                    open_list.append(child)
                    cost_list.append(child.get_cost())

            if len(open_list) > 0:
                best_node_index = np.argmin(cost_list)

                self.change_search_node(open_list[best_node_index])

                algo.increase_LB(self.get_current_node().get_cost() + 1)
                puzzle.move_state(self.get_current_node().state)

                return self.get_current_node()
            else:
                return None

    def back_track(self,puzzle,algo):
        '''
        Goes backwards one Node in the search tree, also reduces the cost of the current node for the algorithm, and
        removes the last action from the puzzle ("sorry, I'm actually not going to do that move!" sort of thing)
        :param puzzle: puzzle class
        :param algo: algorithm class
        :return: the new node for the search, which is actually always the parent node, unless it's the root, then None.
        '''
        if self.get_current_node().get_parent() is None:
            return None
        else:
            old_node = self.get_current_node()
            algo.decrease_LB(old_node.get_cost())
            old_node.kill_node()
            self.change_search_node(old_node.get_parent())
            puzzle.state = self.get_current_node().state
            puzzle.get_path().pop(-1)

            return self.get_current_node()




    def __repr__(self):
        '''
        Just a way to represent the tree. Not a very good one, once it's large :-)
        '''
        return f"NonBinTree({self.val}, parent is {self.parent}): {self.nodes}"



class BnB:
    '''
    Algorithm class, not sure about this one yet. Might change drastically.
    '''
    def __init__(self):
        self.UB = np.inf
        self.LB = 0
        self.optimal_path = []

    def increase_LB(self,cost):
        self.LB += cost

    def decrease_LB(self,cost):
        self.LB -= cost

    def reset_LB(self):
        self.LB = 0

    def save_solution(self,path):
        self.UB = self.LB
        self.reset_LB()
        self.optimal_path = path


def misplaced_cost(state,goal):
    '''
    A cost heuristic of number of misplaced tiles on the board
    :param state: the offered new state
    :param goal: the goal state
    :return: sum of misplaced tiles
    '''
    return np.sum(~np.equal(state,goal))

def misplaced_cost_Yoav(state,goal):
    '''
    A cost heuristic of number of misplaced tiles on the board with Yoav adjustments
    :param state: the offered new state
    :param goal: the goal state
    :return: sum of misplaced tiles
    '''
    return np.sum(np.multiply((~np.equal(state,goal)),np.array([[3,3,3],[2,2,2],[1,1,1]])))

def distance_heuristic(state,goal):
    total_weighted_distance = 0
    for i in range(0, 3):
        for j in range(0,3):
            if state[i,j] != goal[i,j]:
                cur_row, cur_col = np.where(state == goal[i,j])
                cur_row, cur_col = cur_row.item(), cur_col.item()

                total_weighted_distance += (abs(cur_row - i) + abs(cur_col - j))

    return total_weighted_distance


def random_puzzle():
    '''
    :return: a randomized 8 tile puzzle
    '''
    return np.random.choice([0,1,2,3,4,5,6,7,8],9,replace=False).reshape((3,3))

def solve(puzzle,tree,algo,heuristic):
    iterations = 0  # Tracking how many times we expanded to a child node
    win = puzzle.is_win()  # Making sure we are not in the goal state
    if win:
        print('Cleared the Puzzle on iteration %d' % iterations)
        algo.save_solution(puzzle.path)
    done = False  # Not really used yet. In the future, we should not stop when we find the first solution, only after
    #  we made sure there is no other solution that is more optimal. We don't dop that yet.

    print("initial state is")
    print(tree.get_current_node().state)
    while not win:  # Search until you win
        new_node = tree.try_expand_search(puzzle, algo,heuristic)  # Depth first - do we have a child to expand to?


        if new_node is not None:

            # We can expand to a child node!
            iterations += 1
            win = puzzle.is_win() # Did we win?




        else:
            # We can't expand, then we should go back up the tree
            new_node = tree.back_track(puzzle, algo)

            if new_node is None:
                # This means we are at the root node, and the algorithm should finish (not truly implemented yet)
                done = True

        if iterations %1000 == 0:  # Just a logger for sanity check, should be removed when we are happy
            print("at iteration %d cost is %d" %(iterations, algo.LB))
            print(tree.get_current_node().state)

        if win: # Also a fanecy logger, might be removed in the future, but's it's fun.
            print('Cleared the Puzzle on iteration %d' %iterations)
            print(tree.get_current_node().state)
            algo.save_solution(puzzle.path)

    done = True
    return algo.optimal_path

if __name__ == "__main__":


    np.random.seed(2) # Whatever seed we want for generating puzzles

    puzzle = EightTilePuzzle(random_puzzle())
    solvable = puzzle.is_solvable()
    while not solvable:
        puzzle = EightTilePuzzle(random_puzzle())
        solvable = puzzle.is_solvable()

    tree = SearchTree(puzzle.state)
    algo = BnB()
    distance_heuristic = True
    path = solve(puzzle,tree,algo,distance_heuristic)
    print("optimal solution is")
    for i in range(len(path)):
        print(path[i])
        if i < len(path) -1:
            print("    |")
            print("    |")
            print("   \\\'/")


