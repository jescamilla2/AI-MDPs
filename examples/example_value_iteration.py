from src.value_iteration import value_iteration
from src.mdp import initialize_nodes

# Main method to run Value Iteration
if __name__ == "__main__":
    """
    Main method to set up the MDP, run value iteration, and display results.
    """
    # Step 1: Initialize nodes and their transitions/rewards
    nodes = initialize_nodes()

    # Step 2: Run value iteration
    print("Starting Value Iteration...\n")
    value_iteration(nodes)

    # Step 3: Display final results
    print("\nFinal Results After Value Iteration:")
    print("-" * 50)
    for node in nodes.values():
        print(f"Node ({node.state[0]}{node.state[1]} {node.state[2]:>3s}): Value = {node.value:>7.4f}, Optimal Action = {node.policy}")
    print("-" * 50)