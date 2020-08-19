import numpy as np
# DO NOT ADD TO OR MODIFY ANY IMPORT STATEMENTS


def dt_entropy(goal, examples):
    """
    Compute entropy over discrete random variable for decision trees.
    Utility function to compute the entropy (which is always over the 'decision'
    variable, which is the last column in the examples).

    :param goal: Decision variable (e.g., WillWait), cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the entropy of the decision variable, given examples.
    """

    entropy = 0.0
    N = examples.shape[0]  # Number of examples
    M = examples.shape[1]  # Number of columns
    last_col = examples[:, M-1]
    num_goal_values = len(goal[1])

    for v in range(num_goal_values):
        p_v = np.sum(last_col == v) / N
        # Avoid NaN examples by treating the log2(0.0) = 0
        if p_v != 0:
            entropy += p_v * np.log2(p_v)

    return -entropy


def dt_cond_entropy(attribute, col_idx, goal, examples):
    """
    Compute the conditional entropy for attribute. Utility function to compute the conditional entropy (which is always
    over the 'decision' variable or goal), given a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the conditional entropy, given the attribute and examples.
    """

    cond_entropy = 0.0
    N = examples.shape[0]  # Number of examples
    M = examples.shape[1]  # Number of columns
    last_col = examples[:, M-1]
    num_goal_values = len(goal[1])
    attribute_col = examples[:, col_idx]
    num_attribute_values = len(attribute[1])

    for n in range(num_attribute_values):
        # Get examples corresponding to each possible value for specified attribute
        attribute_examples_subset = examples[examples[:, col_idx] == n]
        num_attribute_examples = len(attribute_examples_subset)
        # Get labels corresponding for each example in the subset
        attribute_examples_labels = attribute_examples_subset[:, M-1]
        # Accumulator for inner loop
        accum = 0.0

        for v in range(num_goal_values):
            # Number of positive examples for attribute subset and each label
            num_positive_examples = np.sum(attribute_examples_labels == v)
            # Treat log2(0.0) = 0
            if num_attribute_examples != 0 and num_positive_examples != 0:
                p = num_positive_examples / num_attribute_examples
                accum -= p * np.log2(p)

        cond_entropy += (num_attribute_examples / N) * accum

    return cond_entropy


def dt_info_gain(attribute, col_idx, goal, examples):
    """
    Compute information gain for attribute.
    Utility function to compute the information gain after splitting on attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the information gain, given the attribute and examples.

    """

    info_gain = dt_entropy(goal, examples) - \
        dt_cond_entropy(attribute, col_idx, goal, examples)

    return info_gain


def dt_intrinsic_info(attribute, col_idx, examples):
    """
    Compute the intrinsic information for attribute.
    Utility function to compute the intrinsic information of a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the intrinsic information for the attribute and examples.
    """

    intrinsic_info = 0.0
    N = examples.shape[0]
    # Extract column corresponding to specified attribute
    attribute_values = examples[:, col_idx]
    num_attribute_values = len(attribute[1])

    for n in range(num_attribute_values):
        # Find number of occurrences for each possible value of specified attribute
        num_attribute_subset = np.sum(attribute_values == n)
        p = num_attribute_subset / N
        # Treat log2(0.0) = 0
        if p != 0:
            intrinsic_info -= p * np.log2(p)

    return intrinsic_info


def dt_gain_ratio(attribute, col_idx, goal, examples):
    """
    Compute information gain ratio for attribute.
    Utility function to compute the gain ratio after splitting on attribute. Note that this is just the information
    gain divided by the intrinsic information.
    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the gain ratio, given the attribute and examples.
    """

    gain_ratio = 0.0
    info_gain = dt_info_gain(attribute, col_idx, goal, examples)
    intrinsic_info = dt_intrinsic_info(attribute, col_idx, examples)
    # Avoid NaN examples by treating 0.0/0.0 = 0.0
    if intrinsic_info != 0:
        gain_ratio = info_gain / intrinsic_info

    return gain_ratio


def learn_decision_tree(parent, attributes, goal, examples, score_fun):
    """
    Recursively learn a decision tree from training data.
    Learn a decision tree from training data, using the specified scoring function to determine which attribute to split
    on at each step. This is an implementation of the algorithm on pg. 702 of AIMA.

    :param parent: Parent node in tree (or None if first call of this algorithm).
    :param attributes: Attributes avaialble for splitting at this node.
    :param goal: Goal, decision variable (classes/labels).
    :param examples: Subset of examples that reach this point in the tree.
    :param score_fun: Scoring function used (dt_info_gain or dt_gain_ratio)
    :return: Root node of tree structure.
    """

    node = None
    N = examples.shape[0]  # Number of examples
    M = examples.shape[1]  # Number of columns
    num_attributes = len(attributes)
    last_col = examples[:, M-1]

    # 1. Do any examples reach this point?
    if N == 0:
        if parent:
            node = TreeNode(parent, None, examples, True,
                            plurality_value(goal, parent.examples))
        else:
            return None

    # 2. Or do all examples have the same class/label? If so, we're done!
    elif len(set(last_col)) == 1:
        node = TreeNode(parent, None, examples, True, last_col[0])

    # 3. No attributes left? Choose the majority class/label.
    elif num_attributes == 0:
        node = TreeNode(parent, None, examples, True,
                        plurality_value(goal, examples))

    # 4. Otherwise, need to choose an attribute to split on, but which one? Use score_fun and loop over attributes!
    else:
        # Best score?
        best_attribute_index = -1
        max_score = np.NINF

        for index, attribute in enumerate(attributes):
            score = score_fun(attribute, index, goal, examples)
            # NOTE: to pass the Autolab tests, when breaking ties you should always select the attribute with the smallest (i.e.
            # leftmost) column index!
            # Keep track of best score and corresponding attribute index
            if score > max_score:
                max_score = score
                best_attribute_index = index

        # Now that we know index of best attribute, get it from the list
        best_attribute = attributes[best_attribute_index]

        # Create new node for best-scoring attribute
        node = TreeNode(parent, best_attribute, examples, False, 0)

        # Now, recurse down each branch (operating on a subset of examples below).
        num_best_attribute_values = len(best_attribute[1])

        for best_attribute_value in range(num_best_attribute_values):
            # Get examples for each possible value of best_attribute
            modified_examples = examples[examples[:,
                                                  best_attribute_index] == best_attribute_value]

            # Remove attribute column from modified examples array
            modified_examples = np.delete(
                modified_examples, best_attribute_index, 1)
            # Remove attribute we just look at from attributes list
            attributes_subset = np.delete(attributes, best_attribute_index, 0)

            subtree = learn_decision_tree(
                node, attributes_subset, goal, modified_examples, score_fun)

            # You should append to node.branches in this recursion
            node.branches.append(subtree)

    return node


def plurality_value(goal: tuple, examples: np.ndarray) -> int:
    """
    Utility function to pick class/label from mode of examples (see AIMA pg. 702).
    :param goal: Tuple representing the goal
    :param examples: (n, m) array of examples, each row is an example.
    :return: index of label representing the mode of example labels.
    """
    vals = np.zeros(len(goal[1]))

    # Get counts of number of examples in each possible attribute class first.
    for i in range(len(goal[1])):
        vals[i] = sum(examples[:, -1] == i)

    return np.argmax(vals)


class TreeNode:
    """
    Class representing a node in a decision tree.
    When parent == None, this is the root of a decision tree.
    """

    def __init__(self, parent, attribute, examples, is_leaf, label):
        # Parent node in the tree
        self.parent = parent
        # Attribute that this node splits on
        self.attribute = attribute
        # Examples used in training
        self.examples = examples
        # Boolean representing whether this is a leaf in the decision tree
        self.is_leaf = is_leaf
        # Label of this node (important for leaf nodes that determine classification output)
        self.label = label
        # List of nodes
        self.branches = []

    def query(self, attributes: np.ndarray, goal, query: np.ndarray) -> (int, str):
        """
        Query the decision tree that self is the root of at test time.

        :param attributes: Attributes available for splitting at this node
        :param goal: Goal, decision variable (classes/labels).
        :param query: A test query which is a (n,) array of attribute values, same format as examples but with the final
                      class label).
        :return: label_val, label_txt: integer and string representing the label index and label name.
        """
        node = self
        while not node.is_leaf:
            b = node.get_branch(attributes, query)
            node = node.branches[b]

        return node.label, goal[1][node.label]

    def get_branch(self, attributes: list, query: np.ndarray):
        """
        Find attributes in a set of attributes and determine which branch to use (return index of that branch)

        :param attributes: list of attributes
        :param query: A test query which is a (n,) array of attribute values.
        :return:
        """
        for i in range(len(attributes)):
            if self.attribute[0] == attributes[i][0]:
                return query[i]
        # Return None if that attribute can't be found
        return None

    def count_tree_nodes(self, root=True) -> int:
        """
        Count the number of decision nodes in a decision tree.
        :param root: boolean indicating if this is the root of a decision tree (needed for recursion base case)
        :return: number of nodes in the tree
        """
        num = 0
        for branch in self.branches:
            num += branch.count_tree_nodes(root=False) + 1
        return num + root


if __name__ == '__main__':
    # Example use of a decision tree from AIMA's restaurant problem on page (pg. 698)
    # Each attribute is a tuple of 2 elements: the 1st is the attribute name (a string), the 2nd is a tuple of options
    a0 = ('Alternate', ('No', 'Yes'))
    a1 = ('Bar', ('No', 'Yes'))
    a2 = ('Fri-Sat', ('No', 'Yes'))
    a3 = ('Hungry', ('No', 'Yes'))
    a4 = ('Patrons', ('None', 'Some', 'Full'))
    a5 = ('Price', ('$', '$$', '$$$'))
    a6 = ('Raining', ('No', 'Yes'))
    a7 = ('Reservation', ('No', 'Yes'))
    a8 = ('Type', ('French', 'Italian', 'Thai', 'Burger'))
    a9 = ('WaitEstimate', ('0-10', '10-30', '30-60', '>60'))
    attributes = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    # The goal is a tuple of 2 elements: the 1st is the decision's name, the 2nd is a tuple of options
    goal = ('WillWait', ('No', 'Yes'))

    # Let's input the training data (12 examples in Figure 18.3, AIMA pg. 700)
    # Each row is an example we will use for training: 10 features/attributes and 1 outcome (the last element)
    # The first 10 columns are the attributes with 0-indexed indices representing the value of the attribute
    # For example, the leftmost column represents the attribute 'Alternate': 0 is 'No', 1 is 'Yes'
    # Another example: the 3rd last column is 'Type': 0 is 'French', 1 is 'Italian', 2 is 'Thai', 3 is 'Burger'
    # The 11th and final column is the label corresponding to the index of the goal 'WillWait': 0 is 'No', 1 is 'Yes'
    examples = np.array([[1, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1],
                         [1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 0],
                         [0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 1],
                         [1, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1],
                         [1, 0, 1, 0, 2, 2, 0, 1, 0, 3, 0],
                         [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                         [0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1],
                         [0, 1, 1, 0, 2, 0, 1, 0, 3, 3, 0],
                         [1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                         [1, 1, 1, 1, 2, 0, 0, 0, 3, 2, 1]])

    # entropy_test = dt_entropy(goal, examples)
    # print('entropy_test:', entropy_test)
    # cond_entropy_test = dt_cond_entropy(a2, 2, goal, examples)
    # print('cond_entropy_test:', cond_entropy_test)

    # Build your decision tree using dt_info_gain as the score function
    tree = learn_decision_tree(None, attributes, goal, examples, dt_info_gain)
    # Query the tree with an unseen test example: it should be classified as 'Yes'
    test_query = np.array([0, 0, 1, 1, 2, 0, 0, 0, 2, 3])
    _, test_class = tree.query(attributes, goal, test_query)
    print("Result of query: {:}".format(test_class))

    # Repeat with dt_gain_ratio:
    tree_gain_ratio = learn_decision_tree(
        None, attributes, goal, examples, dt_gain_ratio)
    # Query this new tree: it should also be classified as 'Yes'
    _, test_class = tree_gain_ratio.query(attributes, goal, test_query)
    print("Result of query with gain ratio as score: {:}".format(test_class))
