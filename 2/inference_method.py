from collections import deque
from support import definite_clause

# THIS IS THE TEMPLATE FILE
# WARNING: DO NOT CHANGE THE NAME OF FILE OR THE FUNCTION SIGNATURE


def pl_fc_entails(symbols_list: list, KB_clauses: list, known_symbols: list, query: int) -> bool:
    """
    pl_fc_entails function executes the Propositional Logic forward chaining algorithm (AIMA pg 258).
    It verifies whether the Knowledge Base (KB) entails the query
        Inputs
        ---------
            symbols_list  - a list of symbol(s) (have to be integers) used for this inference problem
            KB_clauses    - a list of definite_clause(s) composed using the numbers present in symbols_list
            known_symbols - a list of symbol(s) from the symbols_list that are known to be true in the KB (facts)
            query         - a single symbol that needs to be inferred

            Note: Definitely check out the test below. It will clarify a lot of your questions.

        Outputs
        ---------
        return - boolean value indicating whether KB entails the query
    """

    # Initialize variables
    inferred = dict.fromkeys(symbols_list, False)
    count = []
    for clause in KB_clauses:
        count.append(len(clause.body))  # Number of symbols in clause's premise

    while len(known_symbols) != 0:
        # Remove symbol from front of list
        p = known_symbols.pop(0)
        # Check if we have satisfied query
        if p == query:
            return True
        # Check if we have already seen this symbol
        if inferred[p] == False:
            inferred[p] = True
            # Get all the premises from the KB
            for i, clause in enumerate(KB_clauses):
                if p in clause.body:
                    count[i] -= 1
                    # If we have gone through all the premises, add conclusion to queue
                    if count[i] == 0:
                        known_symbols.append(clause.conclusion)

    return False


# SAMPLE TEST
if __name__ == '__main__':

    # Symbols used in this inference problem (Has to be Integers)
    symbols = [1, 2, 9, 4, 5]

    # Clause a: 1 and 2 => 9
    # Clause b: 9 and 4 => 5
    # Clause c: 1 => 4
    KB = [definite_clause([1, 2], 9), definite_clause(
        [9, 4], 5), definite_clause([1], 4)]

    # Known Symbols 1, 2
    known_symbols = [1, 2]

    # Does KB entail 5?
    entails = pl_fc_entails(symbols, KB, known_symbols, 5)

    print("Sample Test: " + ("Passed" if entails == True else "Failed"))
