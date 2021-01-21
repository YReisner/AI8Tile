
import numpy as np
import matplotlib.pyplot as plt

class EightTilePuzzle:
    '''
    The board class, keeps tab on the current path, state and goal
    '''
    def __init__(self,initial):
        self.initial_State = initial
        self.state = initial
        self.goal = np.array([[1, 2, 3], [4,5,6], [7,8,0]])
        self.path = [initial]

    def is_solvable(self):
        '''
        Makes sure the initial board state is solvable
        :return: boolean, true if solvable
        '''
        inv_count = 0

        flat = self.state.reshape(9)
        for i in range(9):
            if flat[i] != 0:
                inv_count += sum(flat[i:]>flat[i])
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
        self.g = cost
        self.h = cost
        self.alive = True

    def is_root(self):
        return self.parent is None

    def is_alive(self):
        return self.alive

    def kill_node(self):
        '''
        Makes sure we never expand to this node again relevant for B&B algorithm
        '''
        self.alive = False

    def get_cost(self):
        return self.h+self.g

    def get_g(self):
        return self.g

    def get_h(self):
        return self.h

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def calculate_g(self):
        '''
        g is simply the number of steps it takes to get to this Node, so we simply count the number of parents until
        the root node. Not very efficient, but works :-)
        :return: number of steps to Node
        '''
        g = 0
        node = self
        while node.get_parent() is not None:
            node = node.get_parent()
            g += 1
        return g

    def no_return(self):
        '''
        The idea here is that we should never return to the exact state of one of our ancestors. If we want to do that,
        it is better to simply go back to that ancestor and make a different choice, as we prefer less actions. Also,
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
        self.open_nodes = {}


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

    def get_open_nodes(self):
        return self.open_nodes




    def __repr__(self):
        '''
        Just a way to represent the tree. Not a very good one, once it's large :-)
        '''
        return f"SearchTree(current state is {self.current_node.state}, number of open nodes is {len(self.open_nodes)})"



class BnB:
    '''
    Algorithm class, not sure about this one yet. Might change drastically.
    '''
    def __init__(self,UB_init):
        self.UB = UB_init
        self.LB = 0
        self.optimal_path = []
        self.comparisons = 0

    def increase_LB(self):
        self.LB += 1

    def increase_comparisons(self,nodes):
        self.comparisons += nodes

    def decrease_LB(self):
        self.LB -= 1

    def reset_LB(self):
        self.LB = 0

    def save_solution(self,path):
        self.UB = self.LB
        self.optimal_path = path.copy()

    def examine_addition_to_search(self,tree,state,puzzle,heuristic):
        '''
        We can't simply add a node just because it's a possible move on the board. We need to make sure we don't run
        into infinite loops (see no_return function for Node class)
        :param state: The new state which is a candidate for a new node
        :param puzzle: the board configuration class
        :return: Either the new node for the search process, or None if the state is a bad candidate (creates a return)
        '''

        if heuristic:
            state_cost = distance_heuristic(state, puzzle.goal)
        else:
            state_cost = misplaced_cost(state, puzzle.goal)

        potential_node = Node(state, tree.current_node, state_cost)

        if potential_node.no_return():
            tree.add_node(potential_node)
            return potential_node
        else:
            return None

    def try_expand_search(self,puzzle,tree,heuristic):
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
        if len(tree.get_current_node().get_children()) == 0:
            children = puzzle.create_possible_future_states()

            for i in range(len(children)):

                node = self.examine_addition_to_search(tree, children[i], puzzle,heuristic)
                if node is not None:
                    open_list.append(node)
                    cost_list.append(node.get_h())

            if len(open_list) > 0:
                self.increase_comparisons(len(open_list))
                best_node_index = np.argmin(cost_list)

                if self.LB+open_list[best_node_index].get_h() < self.UB:

                    tree.change_search_node(open_list[best_node_index])

                    self.increase_LB()
                    puzzle.move_state(tree.get_current_node().state)

                    return tree.get_current_node()

        else:
            children = tree.get_current_node().get_children()
            for i in range(len(children)):
                child = children[i]
                if child.is_alive():
                    open_list.append(child)
                    cost_list.append(child.get_cost())

            if len(open_list) > 0:
                self.increase_comparisons(len(open_list))
                best_node_index = np.argmin(cost_list)
                if self.LB + open_list[best_node_index].get_h() < self.UB:
                    tree.change_search_node(open_list[best_node_index])

                    self.increase_LB()
                    puzzle.move_state(tree.get_current_node().state)

                    return tree.get_current_node()

        return None

    def back_track(self,puzzle,tree):
        '''
        Goes backwards one Node in the search tree, also reduces the cost of the current node for the algorithm, and
        removes the last action from the puzzle ("sorry, I'm actually not going to do that move!" sort of thing)
        :param puzzle: puzzle class
        :param algo: algorithm class
        :return: the new node for the search, which is actually always the parent node, unless it's the root, then None.
        '''
        if tree.get_current_node().get_parent() is None:
            return None
        else:
            old_node = tree.get_current_node()
            self.decrease_LB()
            old_node.kill_node()
            tree.change_search_node(old_node.get_parent())
            puzzle.state = tree.get_current_node().state
            puzzle.get_path().pop(-1)


            return tree.get_current_node()

    def solve(self, puzzle,  heuristic):
        win_iterations = []  # for graphing purposes
        win_moves = []  # for graphing purposes
        tree = SearchTree(puzzle.state)
        win = puzzle.is_win()  # Making sure we are not in the goal state
        done = False
        if win:
            print('Cleared the Puzzle in %d comparisons' % self.comparisons)
            self.save_solution(puzzle.path)
            done = True  # If the initial node is the goal, obviously it is also the optimal solution



        print("initial state is")
        print(tree.get_current_node().state)
        while not done:
            backTrack = False
            win = False
            while not win:  # Search until you win
                new_node = self.try_expand_search(puzzle, tree, heuristic)  # Depth first - do we have a child to expand to?

                if new_node is not None:

                    # We can expand to a child node!
                    win = puzzle.is_win()  # Did we win?
                else:
                    new_node = self.back_track(puzzle, tree)

                    if new_node is None:
                        # This means we are at the root node, and the algorithm should finish (not truly implemented yet)
                        done = True
                        break

                if backTrack:
                    # We can't expand, then we should go back up the tree
                    new_node = self.back_track(puzzle, tree)

                    if new_node is None:
                        # This means we are at the root node, and the algorithm should finish (not truly implemented yet)
                        done = True
                        break

                if self.comparisons % 100 == 0:  # Just a logger for sanity check, should be removed when we are happy
                    print("at %d comparisons number of moves is %d" % (self.comparisons, self.LB))
                    print(tree.get_current_node().state)

                if win:  # Also a fancy logger, might be removed in the future, but's it's fun.
                    print('Cleared the Puzzle on comparisons %d in %d moves' % (self.comparisons,self.LB))
                    print(tree.get_current_node().state)
                    self.save_solution(puzzle.path)
                    win_iterations.append(self.comparisons)
                    win_moves.append(len(puzzle.path)-1)



        return self.optimal_path, self.comparisons, win_iterations, win_moves

def trace_path(node):
    '''
    Traces the path from the best node backwards
    :param node: the optimal node (represents the goal state and how we got to it)
    :return: moves taken to get to the node, and the entire path
    '''
    path = [node.state]
    while node.get_parent() is not None:
        node = node.get_parent()
        path.insert(0,node.state)
    return len(path)-1,path

def a_star(puzzle,heuristic):
    '''
    Solves the 8 tile problem with A* algorithm
    :param puzzle: the board
    :param heuristic: Boolean. Heuristic used, at the moment it's either distance (True) or misplaced tiles (False)
    :return: the optimal path to solution, and number of iterations. We also return the number of moves, but that can
    also be easily computer by len(optimal_path)-1
    '''
    tree = SearchTree(puzzle.state) # start a search tree
    comparisons = 0
    state_iterations = []
    state_moves = []
    index = 0  # Index to keep track of tree nodes :-)
    children = puzzle.create_possible_future_states() # Let's add some nodes
    comparisons += len(children)
    for i in range(len(children)):
        child = children[i]
        node = Node(child,tree.get_current_node())
        if heuristic: # True means use distance heuristic
            node.h = distance_heuristic(child,puzzle.goal)
        else:
            node.h = misplaced_cost(child,puzzle.goal)
        node.g = node.calculate_g() # Essentially the number of moves needed to get to state from initial state

        tree.add_node(node) # add to search tree
        comparisons += 1
        index +=1  # Change index for open nodes dict
        tree.get_open_nodes()[index] = (node.get_cost(),node) # save cost-node tuple for easy search-space

    while not puzzle.is_win():
        if comparisons % 100 == 0:  # Just a logger for sanity check, should be removed when we are happy
            print("at %d comparisons  number of moves is %d" % (comparisons, tree.get_current_node().g))
            print(tree.get_current_node().state)

        min_cost = np.inf
        min_node = None
        min_key = None
        open_nodes = tree.get_open_nodes()
        comparisons += len(children)
        for key in open_nodes.keys(): # Which of the open nodes is the best option to expand to?
            # This could have been done much more efficiently with a different data-structure, but as long as the
            # problem isn't too large, it works fine.
            if open_nodes[key][0] < min_cost:

                min_cost = open_nodes[key][0]
                min_node = open_nodes[key][1]
                min_key = key
        tree.change_search_node(min_node)
        puzzle.move_state(min_node.state)
        del open_nodes[min_key] # If I moved to a node, I remove it from the open node list
        if comparisons % 10 == 0:
            state_iterations.append(comparisons)
            state_moves.append(min_node.g)

        children = puzzle.create_possible_future_states()
        for i in range(len(children)):
            child = children[i]
            node = Node(child, tree.get_current_node())
            if node.no_return():
                if heuristic:
                    node.h = distance_heuristic(child, puzzle.goal)
                else:
                    node.h = misplaced_cost(child, puzzle.goal)
                node.g = node.calculate_g()

                tree.add_node(node)
                comparisons += 1
                index += 1
                tree.get_open_nodes()[index] = (node.get_cost(), node)

    moves, path = trace_path(tree.get_current_node())
    state_iterations.append(comparisons)
    state_moves.append(tree.get_current_node().g)
    return comparisons, path, moves, state_iterations, state_moves

def misplaced_cost(state,goal):
    '''
    A cost heuristic of number of misplaced tiles on the board
    :param state: the offered new state
    :param goal: the goal state
    :return: sum of misplaced tiles
    '''
    return np.sum(~np.equal(state,goal))

def distance_heuristic(state,goal):
    '''
    Calculates a Manhattan distance heuristic. The sum of number of "free" moves you need to move each tile to his
    right spot. This means we are not only penalizing for a misplaced tile, we penalize more if the tile is far away
    from his correct spot. This heuristic is admissible because we cannot actually make "free" moves, which means the
    road to victory will always require more moves.
    :param state: the current state
    :param goal: the goal state
    :return: Int, the sum of tile distances from their goal location
    '''
    total_distance = 0
    for i in range(0, 3):
        for j in range(0,3):
            if state[i,j] != goal[i,j]:
                cur_row, cur_col = np.where(state == goal[i,j])
                cur_row, cur_col = cur_row.item(), cur_col.item()

                total_distance += (abs(cur_row - i) + abs(cur_col - j))

    return total_distance


def random_puzzle():
    '''
    :return: a randomized 8 tile puzzle
    '''
    return np.random.choice([0,1,2,3,4,5,6,7,8],9,replace=False).reshape((3,3))



if __name__ == "__main__":


    np.random.seed(30) # Whatever seed we want for generating puzzles

    puzzle = EightTilePuzzle(random_puzzle()) # Generate puzzle
    solvable = puzzle.is_solvable() # Is the puzzle solvable?

    while not solvable:
        puzzle = EightTilePuzzle(random_puzzle())
        solvable = puzzle.is_solvable()

    algorithm = 'C' # Choosing the algorithm to run. A for A*, B for B&B

    use_distance_heuristic = True

    if algorithm == 'A':
        for h in (True,False): # For loop was created for graph comparing heuristics within A*
            use_distance_heuristic = h
            if h:
                iterations, path, moves, state_iterations_h, state_moves_h = a_star(puzzle, use_distance_heuristic)
            else:
                puzzle = EightTilePuzzle(path[0])
                iterations, path, moves, state_iterations_m, state_moves_m = a_star(puzzle, use_distance_heuristic)
        plt.plot(state_iterations_h, state_moves_h,label='Distance Heuristic',color='blue')
        plt.plot(state_iterations_m, state_moves_m, label='Misplaced Heuristic', color='green')
        plt.legend()
        plt.title("A* solution graph")
        plt.xlabel('Number of comparisons')
        plt.ylabel('Number of moves for victory')
        plt.show()
    if algorithm == 'B':
        algo = BnB()
        path,iterations, win_iterations,win_moves = algo.solve(puzzle,use_distance_heuristic)
        moves = len(path)-1
        plt.plot(win_iterations, win_moves)
        plt.title("Branch & Bound solution graph")
        plt.xlabel('Number of comparisons')
        plt.ylabel('Number of moves for victory')
        plt.show()

    else:
        bnb = BnB(np.inf)
        print("Starting BnB algorithm")
        bnb_path,bnb_iterations, bnb_win_iterations,bnb_win_moves = bnb.solve(puzzle,use_distance_heuristic)
        puzzle_new = EightTilePuzzle(puzzle.initial_State)
        print("Starting A* algorithm")
        A_iterations, path, A_moves, A_state_iterations, A_state_moves = a_star(puzzle_new, use_distance_heuristic)

        print('Plotting')
        plt.plot(bnb_win_iterations, bnb_win_moves,label='BnB',color='blue')
        plt.plot(A_state_iterations, A_state_moves, label='A*', color='green')
        plt.legend()
        plt.title("BnB vs A*")
        plt.xlabel('Number of comparisons')
        plt.ylabel('Number of moves for victory')
        plt.show()
    print("optimal solution is")
    for i in range(len(path)):
        print(path[i])
        if i < len(path) -1:
            print("    |")
            print("    |")
            print("   \\\'/")
    print('Solution took %d comparisons and %d moves for A_star' %(A_iterations,A_moves))
    print('Solution took %d comparisons and %d moves for BnB' % (bnb_iterations, len(bnb_path)-1))

