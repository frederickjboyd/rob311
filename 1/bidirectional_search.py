from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem


def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by your search
                 max_frontier_size: maximum frontier size during search
        """
    # Initialize variables and queues for each direction
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []
    discovered_start = [False] * len(problem.V)
    discovered_end = [False] * len(problem.V)
    parents_start = [None] * len(problem.V)
    parents_end = [None] * len(problem.V)
    Q_start = deque()
    Q_end = deque()
    Q_neighbours_start = []
    Q_neighbours_start_parents = {}

    # Add initial state to start queue
    init_state_start = None
    if type(problem.init_state) == list:
        init_state_start = problem.init_state[0]
    else:
        init_state_start = problem.init_state
    Q_start.append(init_state_start)
    discovered_start[init_state_start] = True
    parents_start[init_state_start] = -1

    # Add initial state to end queue
    init_state_end = problem.goal_states[0]
    Q_end.append(init_state_end)
    discovered_end[init_state_end] = True
    parents_end[init_state_end] = -1

    # Check for trivial case
    node_intersection = set(Q_start) & set(Q_end)
    neighbouring_node_intersection_start = set(Q_end) & set(Q_neighbours_start)

    while len(node_intersection) == 0 and len(neighbouring_node_intersection_start) == 0 and len(Q_start) != 0 and len(Q_end) != 0:
        search_iteration(problem, Q_start, discovered_start, parents_start)
        num_nodes_expanded += 1

        # Check again for intersections and exit from loop if at least one is found
        node_intersection = set(Q_start) & set(Q_end)
        if len(node_intersection) != 0:
            break

        search_iteration(problem, Q_end, discovered_end, parents_end)
        num_nodes_expanded += 1

        # Update max frontier size
        if max_frontier_size < max(len(Q_start), len(Q_end)):
            max_frontier_size = max(len(Q_start), len(Q_end))

        node_intersection = set(Q_start) & set(Q_end)

        # Get intersection of Q_end elements with Q_start elements' neighbouring nodes
        Q_neighbours_start = []
        Q_neighbours_start_parents = {}
        for node in Q_start:
            Q_neighbours_start += problem.neighbours[node]
            # Keep track of each neighbour's parents
            Q_neighbours_start_parents[node] = problem.neighbours[node]
        neighbouring_node_intersection_start = set(
            Q_end) & set(Q_neighbours_start)

    # Determine where intersection occurred and get intersecting node
    intersect = None
    if len(neighbouring_node_intersection_start) != 0:
        intersect = list(neighbouring_node_intersection_start)[0]
        # Get parent from dictionary (so we don't have to perform more iterations)
        parents = list(Q_neighbours_start_parents.keys())
        children = list(Q_neighbours_start_parents.values())
        index = None
        for i, child_nodes in enumerate(children):
            if intersect in child_nodes:
                index = i
        parents_start[intersect] = parents[index]
    else:
        intersect = list(node_intersection)[0]

    # Construct partial path from start
    curr_node = parents_start[intersect]
    path_start = []
    path_start.append(curr_node)
    while curr_node != init_state_start:
        curr_node = parents_start[curr_node]
        path_start.append(curr_node)
    path_start.reverse()

    # Construct partial path from end
    curr_node = intersect
    path_end = []
    path_end.append(curr_node)
    while curr_node != init_state_end:
        curr_node = parents_end[curr_node]
        path_end.append(curr_node)

    path = path_start + path_end

    return path, num_nodes_expanded, max_frontier_size


def search_iteration(problem: GraphSearchProblem, Q: deque, discovered: list, parents: list):
    """
    Helper function to handle a single iteration of BFS
    """
    if len(Q) == 0:
        return
    node = Q.popleft()
    for neighbouring_node in problem.neighbours[node]:
        is_discovered = discovered[neighbouring_node]
        # CHeck if it has already been discovered
        if is_discovered != True:
            # If not, mark it as discovered and add it to the queue
            discovered[neighbouring_node] = True
            parents[neighbouring_node] = node
            Q.append(neighbouring_node)


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print(num_nodes_expanded)
    print(max_frontier_size)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt(
        'stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print(num_nodes_expanded)
    print(max_frontier_size)

    E_twitter = np.load('twitter_edges_project_01.npy')
    V_twitter = np.unique(E_twitter)
    twitter_problem = GraphSearchProblem([59999], 0, V_twitter, E_twitter)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(
        twitter_problem)
    correct = twitter_problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print(num_nodes_expanded)
    print(max_frontier_size)

    E_twitter = np.load('twitter_edges_project_01.npy')
    V_twitter = np.unique(E_twitter)
    goal_states = [29999]
    init_state = 64848
    twitter_problem = GraphSearchProblem(
        goal_states, init_state, V_twitter, E_twitter)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(
        twitter_problem)
    correct = twitter_problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print(num_nodes_expanded)
    print(max_frontier_size)

    E_twitter = np.load('twitter_edges_project_01.npy')
    V_twitter = np.unique(E_twitter)
    goal_states = [79999]
    init_state = 0
    twitter_problem = GraphSearchProblem(
        goal_states, init_state, V_twitter, E_twitter)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(
        twitter_problem)
    correct = twitter_problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print(num_nodes_expanded)
    print(max_frontier_size)

    # Be sure to compare with breadth_first_search!
