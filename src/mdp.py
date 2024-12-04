from src.node import Node

def initialize_nodes():
    """
    Initializes the nodes based on the predefined structure and returns them in a dictionary.
    :return: A dictionary mapping node IDs to Node objects
    """
    # Create nodes
    node0 = Node(0, ('R', 'U', '8p'))
    node1 = Node(1, ('T', 'U', '10p'))
    node2 = Node(2, ('R', 'U', '10p'))
    node3 = Node(3, ('R', 'D', '10p'))
    node4 = Node(4, ('R', 'U', '8a'))
    node5 = Node(5, ('R', 'D', '8a'))
    node6 = Node(6, ('T', 'U', '10a'))
    node7 = Node(7, ('R', 'U', '10a'))
    node8 = Node(8, ('R', 'D', '10a'))
    node9 = Node(9, ('T', 'D', '10a'))
    node10 = Node(10, ('_', '_', '11a'), is_terminal=True)

    # Define transitions
    transitions = {
        (0, 'P', 1): 1, (0, 'R', 2): 1, (0, 'S', 3): 1,
        (1, 'R', 4): 1, (1, 'P', 7): 1,
        (2, 'R', 4): 1, (2, 'P', 4): 0.5, (2, 'P', 7): 0.5, (2, 'S', 5): 1,
        (3, 'R', 5): 1, (3, 'P', 5): 0.5, (3, 'P', 8): 0.5,
        (4, 'P', 6): 1, (4, 'R', 7): 1, (4, 'S', 8): 1,
        (5, 'R', 8): 1, (5, 'P', 9): 1,
        (6, 'P', 10): 1, (6, 'R', 10): 1, (6, 'S', 10): 1,
        (7, 'P', 10): 1, (7, 'R', 10): 1, (7, 'S', 10): 1,
        (8, 'P', 10): 1, (8, 'R', 10): 1, (8, 'S', 10): 1,
        (9, 'P', 10): 1, (9, 'R', 10): 1, (9, 'S', 10): 1
    }

    # Define rewards
    rewards = {
        (0, 'P', 1): 2, (0, 'R', 2): 0, (0, 'S', 3): -1,
        (1, 'R', 4): 0, (1, 'P', 7): 2,
        (2, 'R', 4): 0, (2, 'P', 4): 2, (2, 'P', 7): 2, (2, 'S', 5): -1,
        (3, 'R', 5): 0, (3, 'P', 5): 2, (3, 'P', 8): 2,
        (4, 'P', 6): 2, (4, 'R', 7): 0, (4, 'S', 8): -1,
        (5, 'R', 8): 0, (5, 'P', 9): 2,
        (6, 'P', 10): -1, (6, 'R', 10): -1, (6, 'S', 10): -1,
        (7, 'P', 10): 0, (7, 'R', 10): 0, (7, 'S', 10): 0,
        (8, 'P', 10): 4, (8, 'R', 10): 4, (8, 'S', 10): 4,
        (9, 'P', 10): 3, (9, 'R', 10): 3, (9, 'S', 10): 3,
    }

    for node in [node0, node1, node2, node3, node4, node5, node6, node7, node8, node9, node10]:
        node.rewards = {key: value for key, value in rewards.items() if key[0] == node.id}
        node.transitions = {key: value for key, value in transitions.items() if key[0] == node.id}

        # Initialize Q-values for each possible action (set to 0)
        possible_actions = list(set(action for _, action, _ in node.transitions.keys()))
        for action in possible_actions:
            node.set_q_value(action, 0.0)  # Set Q-value for each action to 0 initially

    return {
        0: node0, 1: node1, 2: node2, 3: node3, 4: node4,
        5: node5, 6: node6, 7: node7, 8: node8, 9: node9, 10: node10,
    }