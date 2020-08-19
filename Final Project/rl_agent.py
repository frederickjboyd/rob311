import random
import gym
import math
import numpy as np


"""
Agent Description:

This agent uses a variation of the Monte Carlo method to learn an optimal 
policy. Initially, it randomly explores the least seen action to figure out 
which ones are ideal. In order to "learn" the best actions, it keep a history of
every state and action it has seen along with a corresponding count and value 
for that particular state-action pair. The calculation of a state-action pair's 
value takes into account the previous value, the number of times the pair has 
been seen, and the total reward for that particular episode. The reward for each
episode is discounted over time because actions taken towards the end of the 
episode that caused the agent to fail should be penalized. However, if other 
actions in that episode resulted in the reward being very high, then those 
should be counted more heavily.

Over time, the agent figures out which actions it should take when it sees a 
state. After a certain number of episodes, the agent stops randomly exploring 
actions and only takes the most valued action for a particular state. Since 
there is an element of randomness to the project, this cutoff is to ensure that 
the agent enough chances to meet the requirements of the project. Additionally, 
states are discretized by rounding each number to the nearest 0.5.
"""


class CartPoleAgent:

    def __init__(self, observation_space, action_space):
        # Store observation space and action space.
        self.observation_space = observation_space
        self.action_space = action_space

        # Constants
        # Decay values related to the random exploration
        # Smaller numbers correspond to less exploration
        self.DECAY_X = 0.02
        self.DECAY_Y = 20
        # String constants to avoid typos
        self.COUNT = "count"
        self.VALUE = "value"
        self.STATE = "state"
        self.ACTION = "action"

        # Variables
        # Value decreases as the number of episodes increases
        self.decay = 0
        # Keep track of all states seen and corresponding actions taken for the
        # current episode
        self.curr_episode_state_actions = []
        # Cumulative score for the current episode
        self.curr_episode_rewards = 0
        # Approximate history of all past states seen
        # For all possible actions, each state has a corresponding count of the
        # number of times that action has been taken, as well as a score that
        # quantifies the usefulness of taking that action for the particular
        # state
        self.history = {}
        # Keep track of how many episodes have occurred
        self.episode_count = 0

    def action(self, state):
        """Choose an action from set of possible actions."""
        discretized_state = self.discretize_state(state)

        # Random placeholders for what will become the least explored action
        # and the most valued action
        least_explored_action = self.action_space.sample()
        max_value_action = self.action_space.sample()

        # Check if we have seen this state before - if not, then create a new
        # entry for it
        if discretized_state not in self.history:
            self.history[discretized_state] = [None] * self.action_space.n

            for i in range(self.action_space.n):
                self.history[discretized_state][i] = {
                    self.COUNT: 0, self.VALUE: 0}

        # Get history of current state
        curr_state_history = self.history[discretized_state]

        # Actually find the least explored and most valued actions
        for a in range(self.action_space.n):
            if curr_state_history[a][self.COUNT] < curr_state_history[least_explored_action][self.COUNT]:
                least_explored_action = a
            if curr_state_history[a][self.VALUE] > curr_state_history[max_value_action][self.VALUE]:
                max_value_action = a

        # Determine whether we explore (i.e. learn) or take the best known
        # action as of now
        if random.randint(0, 100) < self.decay and self.episode_count < 700:
            return least_explored_action
        else:
            return max_value_action

        return action

    def reset(self):
        """Reset the agent, if desired."""
        # Update values and counts for states and actions seen in the episode
        for step, s_a_pair in enumerate(self.curr_episode_state_actions):
            self.update_monte_carlo(
                s_a_pair[self.STATE], s_a_pair[self.ACTION], self.curr_episode_rewards - step)

        # Reset variables
        self.curr_episode_rewards = 0
        self.episode_count += 1
        self.curr_episode_state_actions = []

        # Update decay
        self.decay = max(-self.episode_count * self.DECAY_X +
                         self.DECAY_Y, 10 / (self.episode_count + 1))

    def update(self, state, action, reward, state_next, terminal):
        """Update the agent internally after an action is taken."""
        discretized_state = self.discretize_state(state)
        self.curr_episode_rewards += reward
        # Keep track of current state and corresponding action
        self.curr_episode_state_actions.append(
            {self.STATE: discretized_state, self.ACTION: action})

    def update_monte_carlo(self, discretized_state, action, reward):
        """Helper function to update the reward and count values in history."""
        # Get the count and value numbers for the given state-action pair
        s_a_pair = self.history[discretized_state][action]
        # Update value
        s_a_pair[self.VALUE] = (
            s_a_pair[self.VALUE] * s_a_pair[self.COUNT] + reward) / (s_a_pair[self.COUNT] + 1)
        # Increment count
        s_a_pair[self.COUNT] += 1

    def discretize_state(self, state) -> str:
        """
        Helper function to discretize states. Reduces the number of possible 
        states from a continuous observation space to one that is easier to 
        work with by rounding numbers seen in each observation.
        """
        discretized_state = ""

        for x in state:
            # Round each number to the nearest 0.5
            rounded = round(x * 2) / 2
            # Convert to string and concatenate with the discretized state
            discretized_state += str(rounded)

        return discretized_state
