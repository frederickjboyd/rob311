import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem
import matplotlib.pyplot as plt


def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    # Initialize variables and priority queue
    num_nodes_expanded = 0
    max_frontier_size = 0
    path = []
    parents = {}
    priority_queue = queue.PriorityQueue()  # Priority number is f(x)
    # Path length to get from start to state with corresponding index
    g = [np.inf] * (problem.M * problem.N)

    # print('-----Initial State-----')
    # print('init_state:', problem.init_state)
    # x, y = problem.get_position(problem.init_state)
    # print('x:', x)
    # print('y:', y)
    # print('-----Goal State-----')
    # print('goal_state:', problem.goal_states[0])
    # x, y = problem.get_position(problem.goal_states[0])
    # print('x:', x)
    # print('y:', y)

    # Calculate heuristic for initial state and add it to priority queue
    init_state = problem.init_state
    goal_state = problem.goal_states[0]
    init_state_heuristic = problem.heuristic(init_state)
    priority_queue.put([init_state_heuristic, init_state])
    parents[init_state] = -1
    g[init_state] = 0

    # Keep looping while priority queue is not empty
    while priority_queue.empty() != True:
        # Update max frontier size
        if priority_queue.qsize() > max_frontier_size:
            max_frontier_size = priority_queue.qsize()

        # Get node with lowest f(x) value from the priority queue
        f, curr_node = priority_queue.get()
        num_nodes_expanded += 1

        if curr_node == goal_state:
            break

        # Get actions for current node
        # get_actions takes care of obstacle detection
        actions = problem.get_actions(curr_node)

        for neighbour in actions:
            neighbour_state = neighbour[1]
            # Find distance from start to neighbour
            g_temp = g[curr_node] + 1
            # Check if we have found a shorter path to neighbour
            if g_temp < g[neighbour_state]:
                parents[neighbour_state] = curr_node
                # We have found a shorter path, so update g(x)
                g[neighbour_state] = g_temp
                # Calculate f(x) and add neighbour to priority queue
                f_temp = g[neighbour_state] + \
                    problem.heuristic(neighbour_state)
                priority_queue.put([f_temp, neighbour_state])

    # We now reconstruct the path
    try:
        curr_node = goal_state
        path.append(curr_node)
        while curr_node != init_state:
            curr_node = parents[curr_node]
            path.append(curr_node)
    except:
        path = []
        num_nodes_expanded = -1
        max_frontier_size = -1

    # Since we worked our way backwards from the goal state, we need to reverse the order
    path.reverse()

    return path, num_nodes_expanded, max_frontier_size


def a_star_search_modified(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    # Initialize variables and priority queue
    num_nodes_expanded = 0
    max_frontier_size = 0
    path = []
    unique_nodes_found = []
    parents = {}
    priority_queue = queue.PriorityQueue()  # Priority number is f(x)
    # Path length to get from start to state with corresponding index
    g = [np.inf] * (problem.M * problem.N)

    # print('-----Initial State-----')
    # print('init_state:', problem.init_state)
    # x, y = problem.get_position(problem.init_state)
    # print('x:', x)
    # print('y:', y)
    # print('-----Goal State-----')
    # print('goal_state:', problem.goal_states[0])
    # x, y = problem.get_position(problem.goal_states[0])
    # print('x:', x)
    # print('y:', y)

    # Calculate heuristic for initial state and add it to priority queue
    init_state = problem.init_state
    goal_state = problem.goal_states[0]
    init_state_heuristic = problem.heuristic(init_state)
    priority_queue.put([init_state_heuristic, init_state])
    parents[init_state] = -1
    g[init_state] = 0

    # Keep looping while priority queue is not empty
    while priority_queue.empty() != True:
        # Update max frontier size
        if priority_queue.qsize() > max_frontier_size:
            max_frontier_size = priority_queue.qsize()

        # Get node with lowest f(x) value from the priority queue
        f, curr_node = priority_queue.get()
        num_nodes_expanded += 1

        if curr_node == goal_state:
            break

        # Get actions for current node
        # get_actions takes care of obstacle detection
        actions = problem.get_actions(curr_node)

        for neighbour in actions:
            neighbour_state = neighbour[1]

            # Keep track of the number of unique nodes found (not expanded)
            # Check if neighbour has been previously found
            if len(set([neighbour_state]) & set(unique_nodes_found)) == 0:
                unique_nodes_found.append(neighbour_state)

            # Find distance from start to neighbour
            g_temp = g[curr_node] + 1
            # Check if we have found a shorter path to neighbour
            if g_temp < g[neighbour_state]:
                parents[neighbour_state] = curr_node
                # We have found a shorter path, so update g(x)
                g[neighbour_state] = g_temp
                # Calculate f(x) and add neighbour to priority queue
                f_temp = g[neighbour_state] + \
                    problem.heuristic(neighbour_state)
                priority_queue.put([f_temp, neighbour_state])

    # We now reconstruct the path
    try:
        curr_node = goal_state
        path.append(curr_node)
        while curr_node != init_state:
            curr_node = parents[curr_node]
            path.append(curr_node)
    except:
        path = []
        num_nodes_expanded = -1
        max_frontier_size = -1

    # Since we worked our way backwards from the goal state, we need to reverse the order
    path.reverse()

    return path, num_nodes_expanded, max_frontier_size, len(unique_nodes_found)


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.3
    transition_end_probability = 0.4
    peak_nodes_expanded_probability = 0.45
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


def find_search_phase_transition(N: int):
    """
    Helper function for question 2 to plot the "hardness" of grid problems
    """
    num_runs = 100
    p_start = 0.1
    p_end = 0.9
    p_step = 0.05
    p_range = np.arange(p_start, p_end + p_step, p_step)
    # Keep track of percentage of problems that were solvable for each p_occ value
    solvable_array = []
    # Iterate over all the p_occ values we need to test
    for p in p_range:
        print('============')
        print('p:', p)
        num_runs_solvable = 0
        num_nodes_accum = 0
        # Perform a fixed numer of runs for each p_occ value
        for n in range(num_runs):
            problem = get_random_grid_problem(p, N, N)
            path, num_nodes_expanded, max_frontier_size, num_nodes_found = a_star_search_modified(
                problem)
            # Check if problem was solved
            solvable = num_nodes_expanded != -1 and max_frontier_size != -1
            if solvable:
                num_runs_solvable += 1
        solvable_array.append(num_runs_solvable / num_runs)
        print('num_runs:', num_runs)
        print('num_runs_solvable:', num_runs_solvable)
        print('num_runs_not_solvable:', num_runs - num_runs_solvable)

    plt.plot(p_range, solvable_array)
    plt.title('N = ' + str(N))
    plt.xlabel('p_occ')
    plt.ylabel('Solvable Problems (%)')
    plt.savefig('a-star-hardness-' + str(N), dpi=300)
    plt.close()


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 10
    N = 10
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(
        problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    # print('num_nodes_found:', num_nodes_found)
    problem.plot_solution(path)

    # find_search_phase_transition(100)

    # Experiment and compare with BFS
