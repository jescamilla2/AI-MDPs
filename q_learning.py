import random
# Import your Node class and initialization function
from node import Node, initialize_nodes

def q_learning(nodes, episodes=1000, alpha=0.2, gamma=0.99, threshold=0.001):
    """
    Performs Q-learning to find the optimal policy.
    :param nodes: Dictionary of nodes (states) indexed by their IDs.
    :param episodes: Number of episodes to run the Q-learning algorithm.
    :param alpha: Learning rate.
    :param gamma: Discount factor.
    :param threshold: Stopping criteria for the maximum Q-value change.
    :return: None. Updates Q-values and policies in place.
    """
    max_change = float('inf')  # Initialize the maximum Q-value change
    iteration = 0  # Track the number of episodes

    while max_change > threshold and iteration < episodes:
        iteration += 1
        max_change = 0  # Reset the max change for this episode

        # Select a random initial state
        current_node = random.choice(list(nodes.values()))

        # Run through an episode (usually until a terminal state is reached)
        while not current_node.is_terminal():  # Assuming `is_terminal()` determines if it's a terminal state
            # Select action based on random equiprobable policy
            action = random.choice(current_node.get_possible_actions())

            # Debug: print selected action
            print(f"Action chosen for state {current_node.state[0]}{current_node.state[1]} {current_node.state[2]:3s}: {action}")

            # Get the next state (Node object) from the action
            next_node_id = current_node.get_next_node(action)  # Fetch the ID of the next node
            next_node = nodes[next_node_id]  # Use the ID to get the Node object

            # Fetch the reward for the transition
            reward = current_node.rewards.get((current_node.id, action, next_node_id), 0)  # Default reward is 0

            # Q-learning update rule
            possible_actions = next_node.get_possible_actions()
            max_next_q = max((next_node.q_value(a) for a in possible_actions), default=0)

            old_q_value = current_node.q_value(action)

            # Update Q-value for the state-action pair
            new_q_value = old_q_value + alpha * (reward + gamma * max_next_q - old_q_value)

            # Set the new Q-value
            current_node.set_q_value(action, new_q_value)

            # Track the change in Q-value for convergence check
            q_value_change = abs(old_q_value - new_q_value)
            max_change = max(max_change, q_value_change)

            # Print the details of the update for debugging
            print(f"Episode {iteration:2}, Node ({current_node.state[0]}{current_node.state[1]} {current_node.state[2]:>3s}), Action: {action}, "
                  f"Old Q: {old_q_value:>7.4f}, New Q: {new_q_value:>7.4f}, Reward: {reward:>7.4f}, "
                  f"Next State Q: {max_next_q:.4f}")

            # Transition to the next state
            current_node = next_node

        # Decrease the learning rate alpha after each episode
        alpha *= 0.995

        # Print debugging info for the episode
        print(f"Episode {iteration:2} complete, Max Q-value change: {max_change:.4f}")

    # After all episodes, print the final Q-values and optimal policy
    print("\nFinal Q-values and Optimal Policies:")
    for node in nodes.values():
        print(f"Node ({node.state[0]}{node.state[1]} {node.state[2]:>3s}):")
        for action in node.get_possible_actions():
            print(f"  Action: {action}, Q-value: {node.q_value(action)}")

    # Print the optimal policy for each state (action with highest Q-value)
    print("\nOptimal Policies:")
    for node in nodes.values():
        possible_actions = node.get_possible_actions()
        if not possible_actions:
            print(f"Node ({node.state[0]}{node.state[1]} {node.state[2]:3s}): No possible actions available.")
            continue  # Skip this state if no actions are defined

        optimal_action = max(possible_actions, key=lambda action: node.q_value(action))
        print(f"Node ({node.state[0]}{node.state[1]} {node.state[2]:3s}): Optimal Action = {optimal_action}")


# Main method to run Q-learning
if __name__ == "__main__":
    """
    Main method to set up the MDP, run Q-learning, and display results.
    """
    # Step 1: Initialize nodes and their transitions/rewards
    nodes = initialize_nodes()

    # Step 2: Run Q-learning
    print("Starting Q-learning...\n")
    q_learning(nodes)

    # Step 3: Display final results
    print("\nFinal Results After Q-learning:")
    print("-" * 50)
    for node in nodes.values():
        print(f"Node ({node.state[0]}{node.state[1]} {node.state[2]:>3s}): Q-values: {node.q_values}")
    print("-" * 50)
