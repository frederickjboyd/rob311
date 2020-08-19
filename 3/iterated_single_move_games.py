from abc import ABC, abstractmethod
import numpy as np


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """

    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """

    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """

        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000) -> (int, int):
    """

    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print(
                    "WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]

    return p1_score, p2_score


class StudentAgent(IteratedGamePlayer):
    """
    At a high level, we aim to predict the type of agent that are currently up against, 
    then play appropriately to counter that agent's moves. Since we understand how most 
    of the agents work, we know what they will play and we can play a move that will 
    counter that.

    In order to do this, we keep a counter of the number of times the opponent's 
    behaviour aligns with each agent. For example, whenever we see the opponent play 
    a 0 (rock), we add 1 to the first_move counter.

    From there, we use the number of rounds that has been played to calculate the 
    probability of the opponent being each agent. In most cases, this is simply 
    a division of a counter's value by the number of rounds that has been played. 

    Continuing the previous example, if we see that the opponent is consistently 
    playing 0s (rocks), then the first_move counter's value should be the same as 
    the number of rounds played. As such, when we divide the two numbers we get a 
    probability of 1 that the opponent is the first_move agent.

    One other note is that counters and probabilities are not calculated for the 
    Nash equilibrium and Markov process agents. In the event that the probabilities 
    indicate that the agent is one of these, then we use a relatively simple strategy
    (which can be found in more detail through the link below). If we have won the 
    previous round, we play that same move again. If we have lost the previous 
    round, we assume our opponent will play the same move again so we play the 
    counter of their previous move.

    https://arstechnica.com/science/2014/05/win-at-rock-paper-scissors-by-knowing-thy-opponent/
    """

    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)

        self.game_matrix = game_matrix
        self.N = game_matrix.shape[0]

        # Define constants for agents we will go up against
        # This will help us avoid typos in strings we need to use constantly
        self.FIRST_MOVE = 'FirstMove'
        self.COPYCAT = 'Copycat'
        self.UNIFORM = 'Uniform'
        self.GOLDFISH = 'Goldfish'
        self.NASH_EQUILIBRIUM = 'NashEquilibrium'
        self.RANDOM_MARKOV = 'RandomMarkov'

        # Dictionary to keep track of the type of agent our opponent's moves correspond to
        # e.g. every time our opponent chooses the first move, we increment FIRST_MOVE by 1
        self.opponent_agent_moves = {
            self.FIRST_MOVE: 0,
            self.COPYCAT: 0,
            self.UNIFORM: [0] * self.N,  # Number of times each move was played
            self.GOLDFISH: 0,
            self.NASH_EQUILIBRIUM: 0,
            self.RANDOM_MARKOV: 0
        }

        # Dictionary to keep track of probabilities corresponding to opponent agent
        self.opponent_agent_type_probabilities = {
            self.FIRST_MOVE: 0.0,
            self.COPYCAT: 0.0,
            self.UNIFORM: 0.0,
            self.GOLDFISH: 0.0,
            self.NASH_EQUILIBRIUM: 0.0,
            self.RANDOM_MARKOV: 0.0
        }

        # Accumulator to keep track of number of rounds played
        self.num_rounds_played = 0

        # Keep track of our last move to check if opponent agent is COPYCAT
        self.my_last_move = np.random.randint(0, self.N)

        # Keep track of opponent's last move and whether we won or not
        self.opponent_last_move = np.random.randint(0, self.N)
        self.last_round_won = False

        # Extract a, b, c values for later use
        self.a = self.game_matrix[1, 0]
        self.b = self.game_matrix[0, 2]
        self.c = self.game_matrix[2, 1]

    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        most_likely_agent = max(self.opponent_agent_type_probabilities,
                                key=self.opponent_agent_type_probabilities.get)

        threshold = 0.99

        # Counter specific agents
        # FIRST_MOVE
        if most_likely_agent == self.FIRST_MOVE and self.opponent_agent_type_probabilities[self.FIRST_MOVE] > threshold:
            move = 1  # Player paper to counter rock
        # COPYCAT
        elif most_likely_agent == self.COPYCAT and self.opponent_agent_type_probabilities[self.COPYCAT] > threshold:
            move = self.counter_move(self.my_last_move)
        # GOLDFISH
        elif most_likely_agent == self.GOLDFISH and self.opponent_agent_type_probabilities[self.GOLDFISH] > threshold:
            # Goldfish agent will counter our previous move
            # So we want to counter the counter of our previous move
            previous_move_counter = self.counter_move(self.my_last_move)
            move = self.counter_move(previous_move_counter)
        # UNIFORM
        elif most_likely_agent == self.UNIFORM and self.opponent_agent_type_probabilities[self.UNIFORM] > threshold:
            uniform_prob = 1 / self.N
            # Find the largest probable score that we could get
            # Multiply probability elementwise for each row
            score_rock = uniform_prob * (-self.a + self.b)
            score_paper = uniform_prob * (self.a - self.c)
            score_scissors = uniform_prob * (-self.b + self.c)
            move = np.argmax([score_rock, score_paper, score_scissors])
        # NASH_EQUILIBRIUM or RANDOM_MARKOV
        else:
            # Play same move if we won last game
            if self.last_round_won:
                move = self.my_last_move
            # Otherwise, assume that opponent will play the same move as last round
            else:
                move = self.counter_move(self.opponent_last_move)

        return move

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        # my_move: move that we just played
        # other_move: move that opponent just played

        # Update results if opponent agent is FIRST_MOVE
        if other_move == 0:
            self.opponent_agent_moves[self.FIRST_MOVE] += 1

        # Update results if opponent agent is COPYCAT
        if other_move == self.my_last_move:
            self.opponent_agent_moves[self.COPYCAT] += 1

        # Update results if opponent agent is UNIFORM
        self.opponent_agent_moves[self.UNIFORM][other_move] += 1

        # Update results if opponent is GOLDFISH
        if other_move == self.counter_move(self.my_last_move):
            self.opponent_agent_moves[self.GOLDFISH] += 1

        # Calculate new probabilities
        self.num_rounds_played += 1
        # FIRST_MOVE
        self.opponent_agent_type_probabilities[self.FIRST_MOVE] = self.opponent_agent_moves[
            self.FIRST_MOVE] / self.num_rounds_played
        # COPYCAT
        self.opponent_agent_type_probabilities[self.COPYCAT] = self.opponent_agent_moves[self.COPYCAT] / \
            self.num_rounds_played
        # UNIFORM
        num_rock = self.opponent_agent_moves[self.UNIFORM][0]
        num_paper = self.opponent_agent_moves[self.UNIFORM][1]
        num_scissors = self.opponent_agent_moves[self.UNIFORM][2]
        expected = self.num_rounds_played / 3.0
        prob_rock = ((num_rock - expected) ** 2) / (expected ** 2)
        prob_paper = ((num_paper - expected) ** 2) / (expected ** 2)
        prob_scissors = ((num_scissors - expected) ** 2) / (expected ** 2)
        # prob_uniform should never be > 1
        prob_uniform = 1 - ((prob_rock + prob_paper + prob_scissors) / 3)
        self.opponent_agent_type_probabilities[self.UNIFORM] = prob_uniform
        # GOLDFISH
        self.opponent_agent_type_probabilities[self.GOLDFISH] = self.opponent_agent_moves[self.GOLDFISH] / \
            self.num_rounds_played

        # Keep track of our last move so we can compare it the next time update_results is called
        self.my_last_move = my_move

        # Keep track of other variables so we can use them next time as well
        self.opponent_last_move = other_move
        self.last_round_won = self.round_won(my_move, other_move)

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        # Reset opponent_agent_moves
        for key in self.opponent_agent_moves:
            if key == self.UNIFORM:
                self.opponent_agent_moves[key] = [0] * self.N
            else:
                self.opponent_agent_moves[key] = 0

        # Reset opponent_agent_type_probabilities
        for key in self.opponent_agent_type_probabilities:
            self.opponent_agent_type_probabilities[key] = 0.0

        # Reset other miscellaneous variables
        self.num_rounds_played = 0
        self.my_last_move = np.random.randint(0, self.N)
        self.opponent_last_move = np.random.randint(0, self.N)
        self.last_round_won = False

    def round_won(self, my_move: int, opponents_move: int) -> bool:
        """
        Helper function that determines who won a round when given our move and the opponent's move
        """
        my_score = self.game_matrix[my_move, opponents_move]
        opponents_score = self.game_matrix[opponents_move, my_move]

        if my_score > opponents_score:
            return True
        else:
            return False

    def counter_move(self, move: int) -> int:
        """
        Helper function to find the counter of a given move
        """
        if move < 0 or move >= self.N:
            ValueError('Invalid move given to counter_move!')

        # Paper beats rock
        if move == 0:
            return 1
        # Scissors beat paper
        elif move == 1:
            return 2
        # Rock beats scissors
        else:
            return 0


if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0 
    because rock loses to paper.
    """
    game_matrix = np.array([[0.0, -1.0, 1.0],
                            [1.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    # uniform_score, first_move_score = play_game(
    #     uniform_player, first_move_player, game_matrix)

    # print("Uniform player's score: {:}".format(uniform_score))
    # print("First-move player's score: {:}".format(first_move_score))

    copycat_player = CopycatPlayer(game_matrix)

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score, first_move_score = play_game(
        student_player, first_move_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("First-move player's score: {:}".format(first_move_score))

    student_player.reset()

    student_score, copycat_score = play_game(
        student_player, copycat_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("Copycat player's score: {:}".format(copycat_score))

    student_player.reset()

    student_score, uniform_score = play_game(
        student_player, uniform_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("Uniform player's score: {:}".format(uniform_score))

    student_player.reset()
