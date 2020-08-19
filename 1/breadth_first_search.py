from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem


def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    # Initialize variables and queue
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []
    discovered = [False] * len(problem.V)  # Keep track of discovered nodes
    parents = [None] * len(problem.V)  # Parents of each node
    Q = deque()

    # Add initial state to queue
    init_state = None
    if type(problem.init_state) == list:
        init_state = problem.init_state[0]
    else:
        init_state = problem.init_state
    Q.append(init_state)
    discovered[init_state] = True
    parents[init_state] = -1

    while len(Q) != 0:
        # Update the max frontier size
        if len(Q) > max_frontier_size:
            max_frontier_size = len(Q)
        # Get a node from the queue
        node = Q.popleft()
        num_nodes_expanded += 1
        for neighbouring_node in problem.neighbours[node]:
            is_discovered = discovered[neighbouring_node]
            # Check if it has already been discovered
            if is_discovered != True:
                # If not, mark it as discovered and add it to the queue
                discovered[neighbouring_node] = True
                parents[neighbouring_node] = node
                Q.append(neighbouring_node)

    # Construct BFS path by working backwards from the goal state
    curr_node = problem.goal_states[0]
    path.append(curr_node)
    while curr_node != init_state:
        curr_node = parents[curr_node]
        path.append(curr_node)

    # Since we worked our way backwards from the goal state, we need to reverse the order
    path.reverse()

    return path, num_nodes_expanded, max_frontier_size


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
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print(num_nodes_expanded)
    print(max_frontier_size)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print(num_nodes_expanded)
    print(max_frontier_size)

    E_twitter = np.load('twitter_edges_project_01.npy')
    V_twitter = np.unique(E_twitter)
    twitter_problem = GraphSearchProblem([59999], 0, V_twitter, E_twitter)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(
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
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(
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
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(
        twitter_problem)
    correct = twitter_problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print(num_nodes_expanded)
    print(max_frontier_size)
