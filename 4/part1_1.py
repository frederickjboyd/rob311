# part1_1.py: Project 4 Part 1 script
#
# --
# Artificial Intelligence
# ROB 311 Winter 2020
# Programming Project 4
#
# --
# University of Toronto Institute for Aerospace Studies
# Stars Lab
#
# Course Instructor:
# Dr. Jonathan Kelly
# jkelly@utias.utoronto.ca
#
# Teaching Assistant:
# Matthew Giamou
# mathhew.giamau@robotics.utias.utoronto.ca
#
# Abhinav Grover
# abhinav.grover@robotics.utias.utoronto.ca


###
# Imports
###

import numpy as np
from mdp_cleaning_task import cleaning_env

# WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the method  get_transition_model which creates the
    transition probability matrix for the cleaning robot problem described in the
    project document.
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def get_transition_model(env: cleaning_env) -> np.ndarray:
    """
    get_transition_model method creates a table of size (SxSxA) that represents the
    probability of the agent going from s1 to s2 while taking action a
    e.g. P[s1,s2,a] = 0.5
    This is the method that will be used by the cleaning environment (described in the
    project document) for populating its transition probability table

    Inputs
    --------------
        env: The cleaning environment

    Outputs
    --------------
        P: Matrix of size (SxSxA) specifying all of the transition probabilities.
    """

    P = np.zeros([len(env.states), len(env.states), len(env.actions)])

    # Define constants because writing 0 and 1 for left and right is very unintuitive
    LEFT = 0
    RIGHT = 1

    # Get number of states and actions
    num_states = len(env.states)

    # Impossible for robot to move more than one square at a time
    # As such, the entire grid will not be filled
    # Fill out positions between start and end points
    for i in range(1, num_states - 1):
        # Probabilities of robot actually moving left
        P[i, i - 1, LEFT] = 0.8
        P[i, i - 1, RIGHT] = 0.05
        # Probabilities of robot actually staying put
        P[i, i, LEFT] = 0.15
        P[i, i, RIGHT] = 0.15
        # Probabilities of robot actually moving right
        P[i, i + 1, LEFT] = 0.05
        P[i, i + 1, RIGHT] = 0.8

    return P
