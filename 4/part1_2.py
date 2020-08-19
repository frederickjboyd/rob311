# part1_2.py: Project 4 Part 1 script
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
from mdp_env import mdp_env
from mdp_agent import mdp_agent

# WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the value_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def value_iteration(env: mdp_env, agent: mdp_agent, eps: float, max_iter=1000) -> np.ndarray:
    """
    value_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 653). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs
    ---------------
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        eps:   Max error allowed in the utility of a state
        max_iter: Max iterations for the algorithm

    Outputs
    ---------------
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    policy = np.empty_like(env.states)
    print('policy.shape:', policy.shape)
    agent.utility = np.zeros([len(env.states), 1])

    # Define constants
    num_states = len(env.states)
    temp_utility = np.zeros((num_states, 1))

    for i in range(max_iter):
        # Reset delta
        delta = 0

        agent.utility = temp_utility.copy()

        for s in range(num_states):
            # Calculate the summation portion of the Bellman equation
            summation = np.sum(env.transition_model[s] * agent.utility, axis=0)
            # Use summation to calculate temporary utility value for current state
            temp_utility[s] = env.rewards[s] + agent.gamma * np.max(summation)
            # Determine whether delta should be updated
            utility_abs_diff = np.abs(temp_utility[s] - agent.utility[s])
            if utility_abs_diff > delta:
                delta = utility_abs_diff

        # Terminate if the maximum error of the utility for any state is less
        # than the allowed amount
        if delta < eps * (1 - agent.gamma) / agent.gamma:
            break

    # Use utility function to determine policy
    for s in range(num_states):
        policy[s] = np.argmax(
            np.sum(env.transition_model[s] * agent.utility, axis=0))

    return policy
