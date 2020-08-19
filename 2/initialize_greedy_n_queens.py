import numpy as np
from support import plot_n_queens_solution
# WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    # Set random seed for numpy
    # np.random.seed(311)

    greedy_init = np.zeros(N, dtype=int)
    # First queen goes in a random spot
    greedy_init[0] = np.random.randint(0, N)

    # Keep track of all previous queen positions
    previous_queen_positions = [greedy_init[0]]
    # Create 2D array to keep track of how many conflicts are in each square
    board = np.zeros((N, N))

    # Update board for first queen
    board[greedy_init[0]][0] = np.inf
    update_board(greedy_init, 0, N, board)

    # First queen has already been placed so we start iterating at 1
    for col in range(1, N):
        # Get all conflicts for current column
        col_conflicts = board[:, col]

        # Find minimum
        min_conflicts = min(col_conflicts)

        # Loop over column conflicts to get indices of minimum number of conflicts
        min_indices = []
        for i in range(len(col_conflicts)):
            if col_conflicts[i] == min_conflicts:
                min_indices.append(i)

        # Choose a random column index containing the minimum number of conflicts
        random_index = np.random.choice(min_indices)

        # Update position of queen for current column
        greedy_init[col] = random_index

        update_board(greedy_init, col, N, board)

        # Add queen to list of previous queen positions
        previous_queen_positions.append(greedy_init[col] * N + col)

    return greedy_init


def update_board(greedy_init, col, N, board):
    """
    Helper function for initialize_greedy_n_queens to find the number of conflicts in a column
    """
    # Get row
    row = greedy_init[col]

    # Check all possible moves for queen
    # Do not need to account for up/down because we never place more than one queen in a column
    for i in range(1, N):
        # Left
        if 0 <= col - i < N:
            board[row][col - i] += 1
        # Right
        if 0 <= col + i < N:
            board[row][col + i] += 1
        # Top left
        if 0 <= row - i < N and 0 <= col - i < N:
            board[row - i][col - i] += 1
        # Top right
        if 0 <= row - i < N and 0 <= col + i < N:
            board[row - i][col + i] += 1
        # Bottom left
        if 0 <= row + i < N and 0 <= col - i < N:
            board[row + i][col - i] += 1
        # Bottom right
        if 0 <= row + i < N and 0 <= col + i < N:
            board[row + i][col + i] += 1


if __name__ == '__main__':
    assignment = initialize_greedy_n_queens(10)
    plot_n_queens_solution(assignment)
