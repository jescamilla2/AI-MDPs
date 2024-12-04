

def value_iteration(nodes, threshold=0.001, discount_factor=0.99):
    """
    Performs value iteration to find the optimal policy.
    :param nodes: Dictionary of nodes (states) indexed by their IDs.
    :param threshold: Stopping criteria for the maximum value change.
    :param discount_factor: Discount factor (gamma) for future rewards.
    :return: None. Updates node values and policies in place.
    """
    iterations = 0  # Count the number of iterations
    max_change = float('inf')  # Initialize maximum change in value

    while max_change > threshold:
        iterations += 1
        max_change = 0  # Reset the maximum change per iteration

        # Iterate over each node to update its value
        for node in nodes.values():
            old_value = node.value  # Store the old value
            best_action_value = -float('inf')  # Initialize best action value
            best_action = None  # Track the action that gives the best value

            action_values = {}  # To store action values for printing

            # Evaluate all possible actions from the current node
            for action in node.get_possible_actions():
                # Compute the value for this action
                action_value = node.get_next_state_value(action, nodes, discount_factor)

                # Store the action value
                action_values[action] = action_value

                # Check if this action is the best so far
                if action_value > best_action_value:
                    best_action_value = action_value
                    best_action = action

            # If no valid action exists, keep the value constant
            if best_action_value == -float('inf'):
                best_action_value = old_value

            # Update the node's value and policy
            node.value = best_action_value
            node.policy = best_action

            # Calculate the change in value for this node
            value_change = abs(old_value - node.value)
            max_change = max(max_change, value_change)

            # Print updates (for debugging)
            action_value_str = ", ".join([f"{action}: {val:>7.4f}" for action, val in action_values.items()])
            print(f"Node ({node.state[0]}{node.state[1]} {node.state[2]:>3s}): Old Value: {old_value:>7.4f}, New Value: {node.value:>7.4f}, "
                  f"Action Values{{ {action_value_str:34s} }}, Optimal Action: {best_action}, Action Value: {best_action_value:>7.4f}")

        # Debugging output for each iteration
        print(f"Iteration {iterations} - Max Value Change: {max_change:.4f}\n")

    # Final results
    print("Final Value Iteration Results:")
    for node in nodes.values():
        print(f"Node ({node.state[0]}{node.state[1]} {node.state[2]:>3s}): Value = {node.value:>7.4f}, Optimal Action = {node.policy}")

    print(f"\nTotal Iterations: {iterations}")


