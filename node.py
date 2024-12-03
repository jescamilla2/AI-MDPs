import random

class Node:
    def __init__(self, id, state, transitions=None, rewards=None, is_terminal=False):
        """
        Node constructor for MDP representation.
        :param id: Unique identifier for the node.
        :param state: A tuple representing the state of the node (e.g., ('R', 'U', '8p')).
        :param transitions: A dictionary where keys are (current_node, action, next_node) and values are probabilities.
        :param rewards: A dictionary where keys are (current_node, action, next_node) and values are rewards.
        :param is_terminal: Boolean flag to specify if the node is terminal.
        """
        self.id = id
        self.state = state
        self.transitions = transitions or {}
        self.rewards = rewards or {}
        self.value = 0  # Initialize value to 0
        self.policy = None # Store the optimal policy (action)
        self.q_values = {} # Initialize Q-values dictionary
        self.is_terminal_state = is_terminal  # Flag to indicate if this node is terminal

    def update_value(self, new_value):
        """
        Updates the value of the node.
        :param new_value: The new value to be set for the node
        :return: None
        """
        self.value = new_value

    def get_possible_actions(self):
        """
        Extract unique actions from the transitions dictionary
        :return: list of unique actions available from this state
        """
        actions = {key[1] for key in self.transitions.keys()}
        return list(actions)

    def select_action(self):
        """
        Randomly select an action with equal probability among possible actions.
        :return: Selected action.
        """
        # Extract unique actions available from the transitions
        actions = {key[1] for key in self.transitions.keys() if key[0] == self.id}
        return random.choice(list(actions))

    def get_next_node(self, action):
        """
        Determine the next node based on action and transition probabilities.
        :param action: The chosen action.
        :return: The ID of the next node.
        """
        # Filter transitions relevant to the current node and action
        relevant_transitions = {(key[2], prob) for key, prob in self.transitions.items() if key[0] == self.id and key[1] == action}
        rand = random.random()
        cumulative_prob = 0
        for next_node, prob in relevant_transitions:
            cumulative_prob += prob
            if rand < cumulative_prob:
                return next_node
        return next_node  # Default to the last node if cumulative probabilities don't match

    def get_next_state_value(self, action, nodes, discount_factor):
        """
        Calculate the expected value for a given action.
        :param action: The action for which to compute the expected value.
        :param nodes: A dictionary of all nodes by their ID, used to access the value of the next state.
        :return: The expected value for taking the given action.
        """
        total_value = 0

        # Iterate through transitions to find relevant (current_state, action, next_state) tuples
        for (current_state, act, next_state), prob in self.transitions.items():
            if act == action:
                # Get the reward for this transition
                if (current_state, action, next_state) in self.rewards:
                    reward = self.rewards[(current_state, action, next_state)]
                else:
                    reward = 0 # Default reward if not found

                # Get the value of the next state, default to 0 if the next state is not found
                next_state_value = nodes[next_state].value if next_state in nodes else 0

                # Update total value based on Bellman equation
                if prob > 0:
                    total_value += prob * (reward + discount_factor * next_state_value)  # Using discount factor 0.99

        return total_value

    def is_terminal(self):
        """
        Check if this node is a terminal state.
        :return: True if terminal, False otherwise.
        """
        return self.is_terminal_state

    # Q-learning specific methods:
    def q_value(self, action):
        """
        Retrieve the Q-value for a given state-action pair.
        :param action: The action for which to get the Q-value.
        :return: The Q-value for the (state, action) pair. Defaults to 0 if the action is not initialized.
        """
        return self.q_values.get(action, 0)  # Returns 0 if action is not found in q_values

    def set_q_value(self, action, value):
        """
        Set the Q-value for a given state-action pair.
        :param action: The action for which to set the Q-value.
        :param value: The new Q-value to assign to the (state, action) pair.
        :return: None
        """
        self.q_values[action] = value

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


