import numpy as np
# WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """

    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000

    # Initialize board to keep track of conflicts
    board = np.zeros((N, N), dtype=int)
    for i in range(N):
        update_board(solution, i, N, board)

    for idx in range(max_steps):
        # Check if we have found a solution
        valid_solution = check_solution(solution, board, N)
        # If yes, then return it
        if valid_solution:
            print('Found Solution!', idx)
            return solution, idx

        col_conflict_indices = []
        for col, row in enumerate(solution):
            # Check if a queen's current position has conflicts
            if board[row][col] != 0:
                col_conflict_indices.append(col)

        # We do not have a valid solution, so pick a random column containing a conflict
        random_col = np.random.choice(col_conflict_indices)

        # Get the minimum number of conflicts from that column
        random_col_conflicts = board[:, random_col]
        min_conflicts = min(random_col_conflicts)

        # Pick a random row containing the minimum number of conflicts
        random_row = np.random.choice(
            [i for i, e in enumerate(random_col_conflicts) if min_conflicts == e])

        # Decrement board conflict values for queen that we will move
        update_board(solution, random_col, N, board, False)

        # Update position of queen
        solution[random_col] = random_row

        # Increment board conflict values for queen's new position
        update_board(solution, random_col, N, board, True)

        # Keep track of number of iterations
        num_steps += 1

    print('initialization:', initialization)

    print('board:', board)

    return [], -1


def update_board(curr_solution, col, N, board, increment=True):
    """
    Helper function for min_conflicts_n_queens to find the number of conflicts in a column
    """
    # Get queen's row
    row = curr_solution[col]

    # Determine whether to add or subtract 1
    n = 1 if increment == True else -1

    # Check all possible moves for queen
    # Do not need to account for up/down because we never place more than one queen in a column
    for i in range(1, N):
        # Left
        if 0 <= col - i < N:
            board[row][col - i] += n
        # Right
        if 0 <= col + i < N:
            board[row][col + i] += n
        # Top left
        if 0 <= row - i < N and 0 <= col - i < N:
            board[row - i][col - i] += n
        # Top right
        if 0 <= row - i < N and 0 <= col + i < N:
            board[row - i][col + i] += n
        # Bottom left
        if 0 <= row + i < N and 0 <= col - i < N:
            board[row + i][col - i] += n
        # Bottom right
        if 0 <= row + i < N and 0 <= col + i < N:
            board[row + i][col + i] += n


def check_solution(curr_solution, board, N):
    """
    Helper function for min_conflicts_n_queens to check if the current solution is valid

    Also returns the column indices where conflicts exist
    """
    # Check that no conflicts exist for each queen's current position
    # for col, row in enumerate(curr_solution):
    #     if board[row][col] != 0:
    #         return False

    for col in range(N):
        if not np.isin(0, board[:, col]):
            return False

    return True


if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 1000
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    # Plot the initial greedy assignment
    plot_n_queens_solution(assignment_initial)

    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # Plot the solution produced by your algorithm
    plot_n_queens_solution(assignment_solved)
