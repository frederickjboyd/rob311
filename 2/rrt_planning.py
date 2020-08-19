"""
    Problem 3 Template file
"""
import random
import math

import numpy as np

"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees (RRT)
for a problem setup given by the RRT_DUBINS_PROMLEM class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file rrt_planning.py. Your implementation
   can be tested by running RRT_DUBINS_PROBLEM.PY (check the main function).
2. Read all class and function documentation in RRT_DUBINS_PROBLEM carefully.
   There are plenty of helper function in the class for ease of implementation.
3. Your solution must meet all the conditions specificed below.
4. Below are some do's and don'ts for this problem.

Conditions
-------------------
There are some conditions to be satisfied for an acceptable solution.
These may or may not be verified by the marking script.

1. Solution loop must not run for more that a certain number of random points
   (Specified by a class member called MAX_ITER). This is mainly a safety
   measure to avoid time-out related issues and will be generously set.
2. The planning function must return a list of nodes that represent a collision free path
   from start node to the goal node. The path states (path_x, path_y, path_yaw)
   specified by each node must be a dubins-style path and traverse from node i-1 -> node i.
   (READ the documentation of the node to understand the terminology)
3. The returned path should have the start node at index 0 and goal node at index -1,
   while parent node for node i from the list should be node i-1 from the list, ie,
   the path should be a valid list of nodes with dubin-style path connecting the nodes.
   (READ the documentation of the node to understand the terminology)
4. The node locations must not lie outside the map boundaries specified by
   RRT_DUBINS_PROBLEM.map_area

DO(s) and DONT(s)
-------------------
1. Rename the file to rrt_planning.py for submission.
2. Do not change change the PLANNING function signature.
3. Do not import anything other than what is already imported in this file.
4. You can write more function in this file in order to reduce code repitition
   but these function can only be used inside the PLANNING function.
   (since only the planning function will be imported)
"""


def planning(rrt_dubins, display_map=False):
    """
        Execute RRT planning using dubins-style paths. Make sure to populate the node_lis

        Inputs
        -------------
        rrt_dubins  - (RRT_DUBINS_PROBLEM) Class conatining the planning
                      problem specification
        display_map - (boolean) flag for animation on or off (OPTIONAL)

        Outputs
        --------------
        (list of nodes) This must be a valid list of connected nodes that form
                        a path from start to goal node

        NOTE: In order for rrt_dubins.draw_graph function to work properly, it is important
        to populate rrt_dubins.nodes_list with all valid RRT nodes.
    """
    # Fix Randon Number Generator seed
    random.seed(1)

    path = []

    # LOOP for max iterations
    i = 0
    while i < rrt_dubins.max_iter:
        i += 1

        # Generate a random vehicle state (x, y, yaw)
        random_x = random.uniform(rrt_dubins.x_lim[0], rrt_dubins.x_lim[1])
        random_y = random.uniform(rrt_dubins.y_lim[0], rrt_dubins.y_lim[1])
        random_yaw = random.uniform(-math.pi, math.pi)

        # Determine goal
        goal_state = rrt_dubins.goal

        smallest_distance = np.infty
        closest_node = None
        for node in rrt_dubins.node_list:
            # Calculate Euclidean distance
            curr_distance = math.sqrt(
                (node.x - random_x) ** 2 + (node.y - random_y) ** 2)
            if curr_distance < smallest_distance:
                smallest_distance = curr_distance
                closest_node = node

        # Find an existing node nearest to the random vehicle state
        new_node = rrt_dubins.propogate(
            closest_node, rrt_dubins.Node(random_x, random_y, random_yaw))

        # Check if the path between nearest node and random state has obstacle collision
        # Add the node to nodes_list if it is valid
        if rrt_dubins.check_collision(new_node):
            rrt_dubins.node_list.append(new_node)  # Storing all valid nodes

            # Propogate to goal
            propogation = rrt_dubins.propogate(new_node, goal_state)

            # Check for obstacles
            if rrt_dubins.check_collision(propogation):
                prev_node = propogation
                while prev_node != None:
                    path.append(prev_node)
                    prev_node = prev_node.parent
                # Since we are tracing our way back, reverse to get correct order
                path.reverse()
                break

        # Draw current view of the map
        # PRESS ESCAPE TO EXIT
        if display_map:
            rrt_dubins.draw_graph()

        # Check if new_node is close to goal
        if True:
            print("Iters:", i, ", number of nodes:", len(rrt_dubins.node_list))

    if i == rrt_dubins.max_iter:
        print('reached max iterations')

    # Return path, which is a list of nodes leading to the goal
    return path
