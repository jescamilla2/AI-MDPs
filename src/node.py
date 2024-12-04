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




