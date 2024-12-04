from src.q_learning import q_learning
from src.mdp import initialize_nodes

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
