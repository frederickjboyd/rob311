# part2.py: Project 4 Part 2 script
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
  - Complete the policy_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def policy_iteration(env: mdp_env, agent: mdp_agent, max_iter=1000) -> np.ndarray:
    """
    policy_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 657). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs-
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        max_iter: Max iterations for the algorithm

    Outputs -
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    # np.random.seed(1)  # TODO: Remove this

    policy = np.random.randint(len(env.actions), size=(len(env.states)))
    agent.utility = np.zeros([len(env.states), 1])

    # Define variables
    num_states = len(env.states)

    for i in range(max_iter):
        # Reset unchanged
        unchanged = True

        # Evaluate the new utility function for the updated policy
        agent.utility = policy_evaluation(policy, agent.utility, env, agent)

        for s in range(num_states):
            # Calculate the maximum utility for any possible action
            summation_with_action = np.sum(
                env.transition_model[s] * agent.utility, axis=0)
            maximum_with_action = np.max(summation_with_action)

            # Calculate the utility using the given policy
            summation_with_policy = np.sum(np.multiply(
                env.transition_model[s, :, policy[s]], np.squeeze(agent.utility)))

            # If we can achieve a greater utility value with another action,
            # then change the policy to that
            if maximum_with_action > summation_with_policy:
                # Update the policy with the best action
                policy[s] = np.argmax(summation_with_action)
                # Since we have updated the policy, it is not unchanged
                unchanged = False

        # If we have not updated the policy, we can terminate the loop
        if unchanged:
            break

    return policy


def policy_evaluation(policy, utility_func, mdp, agent):
    """Helper function to evaluate the utility of a particular policy"""
    # Variable where we will store updated utility function using new policy
    new_utility_func = np.empty_like(utility_func)

    # Calculate utility for each possible state
    for s in range(len(mdp.states)):
        summation = np.sum(np.multiply(
            mdp.transition_model[s, :, policy[s]], np.squeeze(utility_func)))
        new_utility_func[s] = mdp.rewards[s] + agent.gamma * summation

    return new_utility_func
